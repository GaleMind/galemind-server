//! Clock implementations.

use crate::traits::Clock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The real clock.
#[derive(Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_millis(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    fn unix_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_default()
    }
}

/// A clock that only moves when told to, for tests.
#[derive(Debug, Default)]
pub struct TestClock {
    millis: AtomicU64,
}

impl TestClock {
    pub fn new(start_millis: u64) -> Self {
        Self {
            millis: AtomicU64::new(start_millis),
        }
    }

    /// Moves time forward.
    pub fn advance_millis(&self, delta: u64) {
        self.millis.fetch_add(delta, Ordering::SeqCst);
    }

    pub fn advance_secs(&self, delta: u64) {
        self.advance_millis(delta * 1_000);
    }
}

impl Clock for TestClock {
    fn now_millis(&self) -> u64 {
        self.millis.load(Ordering::SeqCst)
    }

    fn unix_millis(&self) -> u64 {
        self.now_millis()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_test_clock_moves_only_when_advanced() {
        let clock = TestClock::new(1_000);
        assert_eq!(clock.now_millis(), 1_000);
        clock.advance_secs(3_600);
        assert_eq!(clock.now_millis(), 3_601_000);
    }

    #[test]
    fn the_system_clock_never_goes_backwards() {
        let clock = SystemClock::new();
        let first = clock.now_millis();
        assert!(clock.now_millis() >= first);
        assert!(clock.unix_millis() > 1_600_000_000_000);
    }
}
