//! How large a batch to form and how long to wait for it.

use crate::samples::RecentSamples;
use std::time::Duration;

/// What the dispatcher has recently observed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrafficStats {
    /// Requests arriving per second, averaged over the recent window.
    pub arrival_rate_per_sec: f64,
    /// How long one batch has recently taken to execute.
    pub batch_latency: Duration,
    /// Rows in the recent batches.
    pub mean_batch_rows: f64,
    /// Requests waiting in the queue right now.
    pub queue_depth: usize,
    /// Largest batch the model itself permits.
    pub model_max_batch_size: usize,
}

/// Decides the batch window and size.
///
/// Replaceable so that a deployment can trade latency for throughput
/// differently, or pin both values for predictable benchmarking.
pub trait BatchPolicy: Send + Sync {
    /// How long to keep collecting after the first request arrives.
    fn window(&self, stats: &TrafficStats) -> Duration;

    /// Most requests to put in one batch.
    fn max_batch(&self, stats: &TrafficStats) -> usize;
}

/// Fixed window and size, for benchmarks and tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedBatchPolicy {
    pub window: Duration,
    pub max_batch: usize,
}

impl BatchPolicy for FixedBatchPolicy {
    fn window(&self, _stats: &TrafficStats) -> Duration {
        self.window
    }

    fn max_batch(&self, stats: &TrafficStats) -> usize {
        self.max_batch.min(stats.model_max_batch_size).max(1)
    }
}

/// Limits within which the adaptive policy moves.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveBatchConfig {
    /// Shortest window: how long a lone request may be delayed in the hope of
    /// company.
    pub min_window: Duration,
    /// Longest window, which caps the latency batching can add.
    pub max_window: Duration,
    /// Ceiling on batch size, regardless of what the model would allow.
    pub max_batch_size: usize,
    /// How many recent batches the averages cover.
    pub sample_window: usize,
}

impl Default for AdaptiveBatchConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_micros(200),
            max_window: Duration::from_millis(5),
            max_batch_size: 32,
            sample_window: 64,
        }
    }
}

/// Adjusts the batch window and size to the traffic actually arriving.
///
/// Batch size follows Little's law: the number of requests expected to arrive
/// while one batch executes is the arrival rate times the batch latency, so
/// that is the batch size which keeps the backend busy without making anyone
/// wait for a batch that will not fill. Quiet traffic therefore yields batches
/// of one or two and near-minimum windows, while heavy traffic grows both up to
/// their ceilings.
///
/// The window never exceeds the time a batch takes to execute: waiting longer
/// than that adds latency to the request already in hand without raising
/// throughput, because the backend would have been free again by then.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveBatchPolicy {
    config: AdaptiveBatchConfig,
}

impl AdaptiveBatchPolicy {
    pub fn new(config: AdaptiveBatchConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &AdaptiveBatchConfig {
        &self.config
    }

    fn ceiling(&self, stats: &TrafficStats) -> usize {
        self.config
            .max_batch_size
            .min(stats.model_max_batch_size)
            .max(1)
    }
}

impl Default for AdaptiveBatchPolicy {
    fn default() -> Self {
        Self::new(AdaptiveBatchConfig::default())
    }
}

impl BatchPolicy for AdaptiveBatchPolicy {
    fn window(&self, stats: &TrafficStats) -> Duration {
        // Requests already waiting need no waiting for.
        if stats.queue_depth >= self.ceiling(stats) {
            return Duration::ZERO;
        }

        let rate = stats.arrival_rate_per_sec;
        if !rate.is_finite() || rate <= 0.0 {
            return self.config.min_window;
        }

        // Time for the rest of a full batch to turn up at the current rate.
        let missing = self.ceiling(stats).saturating_sub(stats.queue_depth.max(1)) as f64;
        let fill = Duration::from_secs_f64((missing / rate).clamp(0.0, 3_600.0));

        // Never wait longer than a batch takes to run.
        let budget = if stats.batch_latency.is_zero() {
            self.config.max_window
        } else {
            stats.batch_latency
        };

        fill.min(budget)
            .clamp(self.config.min_window, self.config.max_window)
    }

    fn max_batch(&self, stats: &TrafficStats) -> usize {
        let ceiling = self.ceiling(stats);
        let rate = stats.arrival_rate_per_sec;
        if !rate.is_finite() || rate <= 0.0 {
            return 1;
        }

        // Little's law: in-flight requests = arrival rate x service time.
        let latency = if stats.batch_latency.is_zero() {
            self.config.min_window
        } else {
            stats.batch_latency
        };
        let expected = (rate * latency.as_secs_f64()).ceil();
        let expected = if expected.is_finite() {
            expected as usize
        } else {
            ceiling
        };

        expected.max(stats.queue_depth).clamp(1, ceiling)
    }
}

/// Rolling averages feeding a [`BatchPolicy`].
#[derive(Debug)]
pub struct TrafficMeter {
    arrival_intervals_ms: RecentSamples,
    batch_latencies_ms: RecentSamples,
    batch_rows: RecentSamples,
    last_arrival_millis: Option<u64>,
}

impl TrafficMeter {
    pub fn new(sample_window: usize) -> Self {
        Self {
            arrival_intervals_ms: RecentSamples::new(sample_window),
            batch_latencies_ms: RecentSamples::new(sample_window),
            batch_rows: RecentSamples::new(sample_window),
            last_arrival_millis: None,
        }
    }

    /// Records that a request arrived at `now_millis`.
    pub fn record_arrival(&mut self, now_millis: u64) {
        if let Some(previous) = self.last_arrival_millis {
            self.arrival_intervals_ms
                .push(now_millis.saturating_sub(previous) as f64);
        }
        self.last_arrival_millis = Some(now_millis);
    }

    /// Records how long a batch of `rows` rows took.
    pub fn record_batch(&mut self, rows: usize, elapsed: Duration) {
        self.batch_latencies_ms
            .push(elapsed.as_secs_f64() * 1_000.0);
        self.batch_rows.push(rows as f64);
    }

    /// The current picture of traffic.
    pub fn stats(&self, queue_depth: usize, model_max_batch_size: usize) -> TrafficStats {
        let arrival_rate_per_sec = match self.arrival_intervals_ms.mean() {
            // A burst can produce a mean interval of zero; treat it as the
            // highest rate we can express rather than dividing by zero.
            Some(mean) if mean > 0.0 => 1_000.0 / mean,
            Some(_) => f64::MAX,
            None => 0.0,
        };

        TrafficStats {
            arrival_rate_per_sec,
            batch_latency: Duration::from_secs_f64(
                self.batch_latencies_ms.mean().unwrap_or(0.0) / 1_000.0,
            ),
            mean_batch_rows: self.batch_rows.mean().unwrap_or(0.0),
            queue_depth,
            model_max_batch_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(rate: f64, latency_ms: u64, queue_depth: usize) -> TrafficStats {
        TrafficStats {
            arrival_rate_per_sec: rate,
            batch_latency: Duration::from_millis(latency_ms),
            mean_batch_rows: 0.0,
            queue_depth,
            model_max_batch_size: 32,
        }
    }

    #[test]
    fn with_no_traffic_history_the_policy_waits_the_minimum_and_batches_one() {
        let policy = AdaptiveBatchPolicy::default();
        let idle = stats(0.0, 0, 0);
        assert_eq!(policy.window(&idle), policy.config().min_window);
        assert_eq!(policy.max_batch(&idle), 1);
    }

    #[test]
    fn heavy_traffic_grows_the_batch_to_the_ceiling() {
        let policy = AdaptiveBatchPolicy::default();
        // 50k requests/s with a 2ms batch: about 100 arrive per batch, so the
        // ceiling binds.
        let busy = stats(50_000.0, 2, 0);
        assert_eq!(policy.max_batch(&busy), 32);
    }

    #[test]
    fn light_traffic_keeps_batches_small_for_latency() {
        let policy = AdaptiveBatchPolicy::default();
        // 20 requests/s with a 1ms batch: a batch of one is right.
        assert_eq!(policy.max_batch(&stats(20.0, 1, 0)), 1);
        // 2000 requests/s with a 2ms batch: about four arrive per batch.
        assert_eq!(policy.max_batch(&stats(2_000.0, 2, 0)), 4);
    }

    #[test]
    fn the_window_never_exceeds_the_time_a_batch_takes_to_run() {
        let policy = AdaptiveBatchPolicy::new(AdaptiveBatchConfig {
            max_window: Duration::from_millis(50),
            ..Default::default()
        });
        // Rate so low that filling a batch would take 31 seconds, but a batch
        // runs in 3ms, so waiting beyond 3ms only adds latency.
        assert_eq!(policy.window(&stats(1.0, 3, 0)), Duration::from_millis(3));
    }

    #[test]
    fn the_window_stays_inside_its_bounds() {
        let policy = AdaptiveBatchPolicy::new(AdaptiveBatchConfig {
            min_window: Duration::from_millis(1),
            max_window: Duration::from_millis(4),
            ..Default::default()
        });
        assert_eq!(
            policy.window(&stats(1_000_000.0, 0, 0)),
            Duration::from_millis(1)
        );
        assert_eq!(policy.window(&stats(0.5, 100, 0)), Duration::from_millis(4));
    }

    #[test]
    fn a_queue_already_holding_a_full_batch_is_not_made_to_wait() {
        let policy = AdaptiveBatchPolicy::default();
        assert_eq!(policy.window(&stats(100.0, 1, 32)), Duration::ZERO);
        assert_eq!(policy.max_batch(&stats(100.0, 1, 10)), 10);
    }

    #[test]
    fn the_models_own_limit_caps_the_batch() {
        let policy = AdaptiveBatchPolicy::default();
        let capped = TrafficStats {
            model_max_batch_size: 4,
            ..stats(50_000.0, 2, 0)
        };
        assert_eq!(policy.max_batch(&capped), 4);
    }

    #[test]
    fn a_fixed_policy_reports_what_it_was_given() {
        let policy = FixedBatchPolicy {
            window: Duration::from_millis(7),
            max_batch: 8,
        };
        assert_eq!(policy.window(&stats(0.0, 0, 0)), Duration::from_millis(7));
        assert_eq!(policy.max_batch(&stats(0.0, 0, 0)), 8);
    }

    #[test]
    fn the_meter_turns_arrival_intervals_into_a_rate() {
        let mut meter = TrafficMeter::new(8);
        for tick in 0..5 {
            meter.record_arrival(tick * 10);
        }
        let stats = meter.stats(0, 32);
        assert!(
            (stats.arrival_rate_per_sec - 100.0).abs() < 1e-6,
            "{stats:?}"
        );
    }

    #[test]
    fn the_meter_averages_recent_batch_latency() {
        let mut meter = TrafficMeter::new(8);
        meter.record_batch(4, Duration::from_millis(2));
        meter.record_batch(4, Duration::from_millis(4));
        let stats = meter.stats(1, 32);
        assert_eq!(stats.batch_latency, Duration::from_millis(3));
        assert_eq!(stats.mean_batch_rows, 4.0);
        assert_eq!(stats.queue_depth, 1);
    }

    #[test]
    fn a_simultaneous_burst_does_not_divide_by_zero() {
        let mut meter = TrafficMeter::new(8);
        meter.record_arrival(5);
        meter.record_arrival(5);
        let stats = meter.stats(0, 32);
        assert_eq!(stats.arrival_rate_per_sec, f64::MAX);
        assert_eq!(AdaptiveBatchPolicy::default().max_batch(&stats), 32);
    }
}
