//! A fixed-size window of recent measurements.

/// Keeps the most recent `capacity` samples and nothing older.
///
/// Overwriting the oldest sample is exactly right here, which is why this
/// structure belongs to the statistics and not to the request queue: a queue
/// that drops its oldest entry loses a request, while a statistics window that
/// drops its oldest entry simply forgets stale history.
#[derive(Debug)]
pub struct RecentSamples {
    samples: Vec<f64>,
    capacity: usize,
    next: usize,
}

impl RecentSamples {
    /// A window of `capacity` samples. A capacity of zero is raised to one so
    /// the window always holds the latest measurement.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            next: 0,
        }
    }

    /// Records a sample, forgetting the oldest once the window is full.
    pub fn push(&mut self, sample: f64) {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
        } else {
            self.samples[self.next] = sample;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Mean of the window, or `None` while no sample has been recorded.
    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
    }

    /// Largest sample in the window.
    pub fn max(&self) -> Option<f64> {
        self.samples
            .iter()
            .copied()
            .fold(None, |acc, s| Some(acc.map_or(s, |best: f64| best.max(s))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_window_has_no_mean() {
        let samples = RecentSamples::new(4);
        assert!(samples.is_empty());
        assert_eq!(samples.mean(), None);
        assert_eq!(samples.max(), None);
    }

    #[test]
    fn the_mean_covers_only_what_is_held() {
        let mut samples = RecentSamples::new(4);
        samples.push(1.0);
        samples.push(3.0);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples.mean(), Some(2.0));
    }

    #[test]
    fn old_samples_are_forgotten_once_the_window_is_full() {
        let mut samples = RecentSamples::new(3);
        for value in [10.0, 10.0, 10.0, 1.0, 1.0, 1.0] {
            samples.push(value);
        }
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.mean(), Some(1.0));
        assert_eq!(samples.max(), Some(1.0));
    }

    #[test]
    fn a_zero_capacity_window_still_holds_the_latest_sample() {
        let mut samples = RecentSamples::new(0);
        samples.push(5.0);
        samples.push(7.0);
        assert_eq!(samples.mean(), Some(7.0));
    }
}
