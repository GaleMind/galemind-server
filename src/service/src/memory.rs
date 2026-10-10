//! Reclaiming memory when the server is holding too much of it.
//!
//! The server's resident memory is mostly not its own doing: loaded sessions
//! are a few megabytes, an execution's intermediates are tens, and underneath
//! both the allocator keeps what it has freed rather than returning it. That
//! is efficient and bounded -- it plateaus, it is not a leak -- but a plateau
//! is still a number, and a deployment with a memory limit cares about the
//! number rather than about whose fault it is.
//!
//! So there is a high-water mark. Above it the server reclaims, in two steps,
//! cheapest first:
//!
//! 1. **Ask the allocator for its pages back.** Nothing is lost: these are
//!    pages already freed, which the allocator was holding in case they were
//!    wanted again. The cost is page faults when the work returns, which is
//!    why it is not done after every request.
//! 2. **Unload the models**, if that was not enough. This does lose
//!    something: the next request for a model pays its load again, which for
//!    a heavy model is tens of milliseconds. It happens only when a purge
//!    left the process still above the mark.
//!
//! What it deliberately does not do is refuse requests. A server over its
//! memory mark is not a server that cannot answer, and the mechanism for too
//! much work in flight is already the queue.

use crate::ResourceSampler;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Returns an allocator's retained pages to the operating system.
///
/// Implemented by whoever installed the global allocator, which is the binary
/// -- so this is a seam rather than a call. The platform allocator cannot do
/// it at all (glibc needs `malloc_trim`, which is FFI), which is what
/// [`NoReclaim`] reports.
pub trait AllocatorReclaim: Send + Sync {
    /// How this appears in the log and the metrics.
    fn name(&self) -> &'static str;

    /// Hands back what the allocator is holding.
    ///
    /// Called from a blocking context, because purging is a syscall per
    /// region and not something to do on an async worker.
    fn reclaim(&self) -> Reclaim;
}

/// What asking an allocator for its pages achieved.
///
/// Three outcomes rather than a boolean, because they call for different
/// things to be said. "There is no way to ask this allocator" is a build to
/// change; "the allocator refused" is a transient condition to report with
/// its reason; and a success that freed nothing means the memory is in use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reclaim {
    /// This allocator has no way to return pages, whatever is asked.
    Unsupported,
    /// The allocator was asked and did what it could.
    Done,
    /// The allocator refused, for a reason worth logging.
    Refused(String),
}

/// An allocator that cannot be asked, which is the platform's own.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoReclaim;

impl AllocatorReclaim for NoReclaim {
    fn name(&self) -> &'static str {
        "system"
    }

    fn reclaim(&self) -> Reclaim {
        Reclaim::Unsupported
    }
}

/// When to reclaim, and how hard.
#[derive(Debug, Clone, Copy)]
pub struct MemoryGuardConfig {
    /// Resident bytes above which a reclaim is triggered. `None` never
    /// reclaims.
    pub high_water_bytes: Option<u64>,
    /// How long after a reclaim before another may run.
    ///
    /// Without this, a server whose steady state is above the mark reclaims
    /// on every sample and spends itself returning pages it immediately
    /// faults back in.
    pub cooldown: Duration,
    /// Whether to unload models when a purge was not enough.
    pub may_unload: bool,
}

impl Default for MemoryGuardConfig {
    fn default() -> Self {
        Self {
            high_water_bytes: None,
            cooldown: Duration::from_secs(30),
            may_unload: true,
        }
    }
}

/// Watches resident memory and reclaims above the mark.
pub struct MemoryGuard {
    config: MemoryGuardConfig,
    allocator: Arc<dyn AllocatorReclaim>,
    sampler: ResourceSampler,
    /// Reclaims run, published as `gm_memory_reclaim_count`.
    reclaims: AtomicU64,
    /// Reclaims that went on to unload models.
    unloads: AtomicU64,
    /// When the last one ran, for the cooldown.
    last: std::sync::Mutex<Option<Instant>>,
}

/// What a check decided.
///
/// A decision rather than an outcome: unloading models is async and the
/// check is not, so the guard says what is needed and the caller does it.
/// That also keeps the guard from logging a count it would have to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reclaimed {
    /// Below the mark, or no mark configured.
    NotNeeded,
    /// Above the mark, but the previous reclaim was too recent.
    InCooldown,
    /// The allocator was asked and that was enough.
    Purged { before: u64, after: u64 },
    /// The allocator was asked and the process is still above the mark, so
    /// the models should go too.
    StillOver { before: u64, after: u64 },
}

impl MemoryGuard {
    pub fn new(config: MemoryGuardConfig, allocator: Arc<dyn AllocatorReclaim>) -> Self {
        Self {
            config,
            allocator,
            sampler: ResourceSampler::new(),
            reclaims: AtomicU64::new(0),
            unloads: AtomicU64::new(0),
            last: std::sync::Mutex::new(None),
        }
    }

    /// The configured mark, for the metrics.
    pub fn high_water_bytes(&self) -> u64 {
        self.config.high_water_bytes.unwrap_or(0)
    }

    pub fn reclaim_count(&self) -> u64 {
        self.reclaims.load(Ordering::Relaxed)
    }

    pub fn unload_count(&self) -> u64 {
        self.unloads.load(Ordering::Relaxed)
    }

    /// Resident bytes now, as the guard reads them.
    pub fn resident_bytes(&self) -> u64 {
        self.sampler.sample().memory_bytes
    }

    /// Checks the mark and asks the allocator for its pages if it has been
    /// crossed.
    ///
    /// Returns [`Reclaimed::StillOver`] when that was not enough, which is
    /// the caller's cue to unload the models and to report back with
    /// [`MemoryGuard::record_unload`].
    pub fn check(&self) -> Reclaimed {
        let Some(mark) = self.config.high_water_bytes else {
            return Reclaimed::NotNeeded;
        };

        let before = self.resident_bytes();
        if before <= mark {
            return Reclaimed::NotNeeded;
        }

        // A steady state above the mark is a reason to reclaim occasionally,
        // not continuously.
        {
            let mut last = match self.last.lock() {
                Ok(last) => last,
                // A poisoned lock means a previous check panicked; reclaiming
                // is not worth propagating that.
                Err(e) => e.into_inner(),
            };
            if let Some(at) = *last {
                if at.elapsed() < self.config.cooldown {
                    debug!(
                        resident_mb = before / (1024 * 1024),
                        "above the memory mark, but a reclaim ran recently"
                    );
                    return Reclaimed::InCooldown;
                }
            }
            *last = Some(Instant::now());
        }

        self.reclaims.fetch_add(1, Ordering::Relaxed);
        match self.allocator.reclaim() {
            Reclaim::Done => {}
            // Said once per reclaim rather than once per sample: a server
            // given a mark it has no way to honour should know why nothing
            // is happening, and should be told what would fix it.
            Reclaim::Unsupported => warn!(
                allocator = self.allocator.name(),
                resident_mb = before / (1024 * 1024),
                mark_mb = mark / (1024 * 1024),
                "over the memory mark and this allocator cannot return pages; \
                 build with the jemalloc feature"
            ),
            // A different thing entirely: the allocator can, and this time
            // would not. Usually transient, so it is reported rather than
            // treated as a misconfiguration.
            Reclaim::Refused(reason) => warn!(
                allocator = self.allocator.name(),
                resident_mb = before / (1024 * 1024),
                reason = %reason,
                "the allocator refused to return pages"
            ),
        }
        let after = self.resident_bytes();

        let returned = before.saturating_sub(after);
        if after <= mark || !self.config.may_unload {
            info!(
                allocator = self.allocator.name(),
                before_mb = before / (1024 * 1024),
                after_mb = after / (1024 * 1024),
                returned_mb = returned / (1024 * 1024),
                mark_mb = mark / (1024 * 1024),
                "reclaimed memory"
            );
            return Reclaimed::Purged { before, after };
        }

        // Still over. Worth distinguishing in the log, because the two cases
        // call for different actions: a purge that returned nothing means
        // the memory is in use rather than retained, and no amount of
        // reclaiming will help -- the batch is too large or the mark is too
        // low. One that returned something and is still over means the
        // server is simply working above its limit.
        if returned == 0 {
            warn!(
                resident_mb = after / (1024 * 1024),
                mark_mb = mark / (1024 * 1024),
                "over the memory mark and nothing could be returned: this \
                 memory is in use, not retained"
            );
        } else {
            warn!(
                before_mb = before / (1024 * 1024),
                after_mb = after / (1024 * 1024),
                returned_mb = returned / (1024 * 1024),
                mark_mb = mark / (1024 * 1024),
                "reclaimed memory and stayed over the mark"
            );
        }
        Reclaimed::StillOver { before, after }
    }

    /// Records that the caller unloaded `models` after a
    /// [`Reclaimed::StillOver`].
    ///
    /// Reported rather than inferred: the guard cannot unload anything
    /// itself, so counting its own advice would overstate what happened.
    pub fn record_unload(&self, models: usize) {
        if models > 0 {
            self.unloads.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reports whatever a test wants, and counts the asking.
    struct Fake {
        works: bool,
        calls: AtomicU64,
    }

    impl AllocatorReclaim for Fake {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn reclaim(&self) -> Reclaim {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.works {
                Reclaim::Done
            } else {
                Reclaim::Refused("the fake refuses".to_string())
            }
        }
    }

    fn guard(mark: Option<u64>, cooldown: Duration) -> (Arc<Fake>, MemoryGuard) {
        let allocator = Arc::new(Fake {
            works: true,
            calls: AtomicU64::new(0),
        });
        let guard = MemoryGuard::new(
            MemoryGuardConfig {
                high_water_bytes: mark,
                cooldown,
                may_unload: true,
            },
            Arc::clone(&allocator) as Arc<dyn AllocatorReclaim>,
        );
        (allocator, guard)
    }

    #[test]
    fn no_mark_means_nothing_happens() {
        let (allocator, guard) = guard(None, Duration::ZERO);
        assert_eq!(guard.check(), Reclaimed::NotNeeded);
        assert_eq!(allocator.calls.load(Ordering::Relaxed), 0);
        assert_eq!(guard.reclaim_count(), 0);
        assert_eq!(guard.high_water_bytes(), 0);
    }

    /// A mark this process is nowhere near must not trigger anything, which
    /// is the case every healthy server is in.
    #[test]
    fn below_the_mark_nothing_happens() {
        let (allocator, guard) = guard(Some(64 * 1024 * 1024 * 1024), Duration::ZERO);
        assert_eq!(guard.check(), Reclaimed::NotNeeded);
        assert_eq!(allocator.calls.load(Ordering::Relaxed), 0);
    }

    /// A mark of one byte is always crossed, which is how the action is
    /// tested without having to allocate hundreds of megabytes.
    #[test]
    fn above_the_mark_the_allocator_is_asked_first() {
        let (allocator, guard) = guard(Some(1), Duration::ZERO);
        let outcome = guard.check();

        assert_eq!(allocator.calls.load(Ordering::Relaxed), 1);
        assert_eq!(guard.reclaim_count(), 1);
        // This process stays well above one byte, so a purge cannot bring it
        // under and the caller is told to unload.
        assert!(
            matches!(outcome, Reclaimed::StillOver { .. }),
            "{outcome:?}"
        );

        // The count follows what the caller reports, not what was advised.
        assert_eq!(guard.unload_count(), 0);
        guard.record_unload(0);
        assert_eq!(guard.unload_count(), 0, "nothing was unloaded");
        guard.record_unload(3);
        assert_eq!(guard.unload_count(), 1);
    }

    #[test]
    fn unloading_can_be_forbidden_and_then_only_the_allocator_is_asked() {
        let allocator = Arc::new(Fake {
            works: true,
            calls: AtomicU64::new(0),
        });
        let guard = MemoryGuard::new(
            MemoryGuardConfig {
                high_water_bytes: Some(1),
                cooldown: Duration::ZERO,
                may_unload: false,
            },
            Arc::clone(&allocator) as Arc<dyn AllocatorReclaim>,
        );

        let outcome = guard.check();
        assert!(
            matches!(outcome, Reclaimed::Purged { .. }),
            "models must be kept when unloading is forbidden: {outcome:?}"
        );
        assert_eq!(guard.unload_count(), 0);
    }

    /// The cooldown is what keeps a server whose steady state is over the
    /// mark from spending itself purging and faulting the same pages.
    #[test]
    fn a_reclaim_is_not_repeated_until_the_cooldown_passes() {
        let (allocator, guard) = guard(Some(1), Duration::from_secs(3600));

        assert!(matches!(guard.check(), Reclaimed::StillOver { .. }));
        for _ in 0..5 {
            assert_eq!(guard.check(), Reclaimed::InCooldown);
        }
        assert_eq!(
            allocator.calls.load(Ordering::Relaxed),
            1,
            "the allocator is asked once, not once per sample"
        );
        assert_eq!(guard.reclaim_count(), 1);
    }

    /// An allocator that cannot return pages still gets the mark honoured as
    /// far as it can be -- and says so, rather than failing silently.
    #[test]
    fn an_allocator_that_cannot_reclaim_still_unloads() {
        let allocator = Arc::new(Fake {
            works: false,
            calls: AtomicU64::new(0),
        });
        let guard = MemoryGuard::new(
            MemoryGuardConfig {
                high_water_bytes: Some(1),
                cooldown: Duration::ZERO,
                may_unload: true,
            },
            allocator,
        );
        assert!(matches!(guard.check(), Reclaimed::StillOver { .. }));
    }

    /// The three outcomes are distinguished because they call for different
    /// things to be said: a build to change, a reason to report, or nothing.
    #[test]
    fn the_platform_allocator_reports_that_it_cannot() {
        assert_eq!(NoReclaim.name(), "system");
        assert_eq!(NoReclaim.reclaim(), Reclaim::Unsupported);
        assert_ne!(
            Reclaim::Unsupported,
            Reclaim::Refused("transient".to_string()),
            "a missing implementation is not a refusal"
        );
    }

    #[test]
    fn resident_memory_is_readable() {
        let (_, guard) = guard(None, Duration::ZERO);
        assert!(guard.resident_bytes() > 0);
    }
}
