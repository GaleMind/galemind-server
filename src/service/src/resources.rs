//! CPU and memory use of the server process, sampled over time.
//!
//! Utilisation is a rate, so it only means anything between two readings: each
//! sample is compared with the previous one. A server therefore samples on an
//! interval and the metrics endpoint reports the newest reading, which is what
//! Triton calls a per-interval metric.
//!
//! The readings come from `/proc`. Where it is unavailable the snapshot simply
//! reports zeros rather than failing a metrics scrape.

use std::sync::Mutex;
use std::time::Instant;

/// One reading of process and machine resource use.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ResourceSnapshot {
    /// Share of the machine's CPU the process used since the previous sample,
    /// where 1.0 is every core fully busy.
    pub cpu_utilization: f64,
    /// Resident memory of the process.
    pub memory_bytes: u64,
    /// Memory the machine has in total.
    pub memory_total_bytes: u64,
    /// `memory_bytes / memory_total_bytes`, or 0 when the total is unknown.
    pub memory_utilization: f64,
    /// Cores the process may use.
    pub cpu_count: usize,
    /// Seconds since the sampler was created.
    pub uptime_secs: f64,
}

#[derive(Debug, Clone, Copy)]
struct Reading {
    at: Instant,
    cpu_secs: f64,
}

/// Samples resource use, remembering the previous reading so that CPU time can
/// be turned into a utilisation.
pub struct ResourceSampler {
    started: Instant,
    cpu_count: usize,
    clock_ticks_per_sec: f64,
    previous: Mutex<Option<Reading>>,
}

impl ResourceSampler {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            cpu_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            // The kernel reports process CPU time in clock ticks. 100 per
            // second is the near-universal value on Linux, and getting it
            // wrong only scales the reading.
            clock_ticks_per_sec: 100.0,
            previous: Mutex::new(None),
        }
    }

    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }

    /// Takes a reading, measuring CPU use since the previous one.
    pub fn sample(&self) -> ResourceSnapshot {
        let now = Instant::now();
        let cpu_secs = self.process_cpu_seconds();
        let memory_bytes = read_file("/proc/self/status")
            .and_then(|text| parse_vm_rss_bytes(&text))
            .unwrap_or(0);
        let memory_total_bytes = read_file("/proc/meminfo")
            .and_then(|text| parse_mem_total_bytes(&text))
            .unwrap_or(0);

        let cpu_utilization = {
            let mut previous = self.previous.lock().ok();
            let last = previous.as_ref().and_then(|slot| **slot);
            if let Some(slot) = previous.as_mut() {
                **slot = Some(Reading { at: now, cpu_secs });
            }

            match last {
                Some(last) => {
                    let elapsed = now.duration_since(last.at).as_secs_f64();
                    cpu_rate(cpu_secs - last.cpu_secs, elapsed, self.cpu_count)
                }
                // The first reading has nothing to compare against.
                None => 0.0,
            }
        };

        ResourceSnapshot {
            cpu_utilization,
            memory_bytes,
            memory_total_bytes,
            memory_utilization: if memory_total_bytes > 0 {
                memory_bytes as f64 / memory_total_bytes as f64
            } else {
                0.0
            },
            cpu_count: self.cpu_count,
            uptime_secs: self.started.elapsed().as_secs_f64(),
        }
    }

    fn process_cpu_seconds(&self) -> f64 {
        read_file("/proc/self/stat")
            .and_then(|text| parse_cpu_ticks(&text))
            .map(|ticks| ticks / self.clock_ticks_per_sec)
            .unwrap_or(0.0)
    }
}

impl Default for ResourceSampler {
    fn default() -> Self {
        Self::new()
    }
}

fn read_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// CPU utilisation from the CPU seconds used over a wall-clock interval.
fn cpu_rate(cpu_secs: f64, elapsed_secs: f64, cpu_count: usize) -> f64 {
    if elapsed_secs <= 0.0 || cpu_count == 0 || !cpu_secs.is_finite() {
        return 0.0;
    }
    // A reading can only be clamped, not trusted beyond its bounds: counters
    // can appear to go backwards across a sampling gap.
    (cpu_secs / elapsed_secs / cpu_count as f64).clamp(0.0, 1.0)
}

/// Reads user plus system time, in clock ticks, from `/proc/self/stat`.
///
/// The process name is in parentheses and may itself contain spaces, so fields
/// are counted from the closing parenthesis rather than from the start.
fn parse_cpu_ticks(stat: &str) -> Option<f64> {
    let rest = stat.rsplit_once(')')?.1;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // After the name, field 1 is the state, so utime and stime are 12 and 13.
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// Reads resident memory from `/proc/self/status`, which reports kibibytes.
fn parse_vm_rss_bytes(status: &str) -> Option<u64> {
    parse_kib_field(status, "VmRSS:")
}

/// Reads total memory from `/proc/meminfo`, which reports kibibytes.
fn parse_mem_total_bytes(meminfo: &str) -> Option<u64> {
    parse_kib_field(meminfo, "MemTotal:")
}

fn parse_kib_field(text: &str, prefix: &str) -> Option<u64> {
    text.lines()
        .find(|line| line.starts_with(prefix))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|kib| kib * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ticks_are_read_past_a_process_name_containing_spaces() {
        let stat = "42 (my server (x)) S 1 42 42 0 -1 4194304 100 0 0 0 250 50 0 0 20 0";
        assert_eq!(parse_cpu_ticks(stat), Some(300.0));
    }

    #[test]
    fn a_truncated_stat_line_yields_nothing_rather_than_a_wrong_number() {
        assert_eq!(parse_cpu_ticks("42 (x) S 1 2 3"), None);
        assert_eq!(parse_cpu_ticks("no parenthesis here"), None);
    }

    #[test]
    fn memory_fields_are_read_as_bytes() {
        let status = "Name:\tgalemind\nVmPeak:\t  900 kB\nVmRSS:\t  4096 kB\n";
        assert_eq!(parse_vm_rss_bytes(status), Some(4_194_304));

        let meminfo = "MemTotal:       16384 kB\nMemFree:         1024 kB\n";
        assert_eq!(parse_mem_total_bytes(meminfo), Some(16_777_216));
    }

    #[test]
    fn a_missing_memory_field_yields_nothing() {
        assert_eq!(parse_vm_rss_bytes("Name:\tgalemind\n"), None);
    }

    #[test]
    fn utilisation_is_cpu_seconds_over_wall_seconds_across_all_cores() {
        // Two cores busy for one second out of four available: a quarter.
        assert!((cpu_rate(2.0, 1.0, 4) - 0.5).abs() < 1e-9);
        assert!((cpu_rate(1.0, 1.0, 4) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn utilisation_is_clamped_and_never_divides_by_zero() {
        assert_eq!(cpu_rate(5.0, 0.0, 4), 0.0);
        assert_eq!(cpu_rate(1.0, 1.0, 0), 0.0);
        assert_eq!(cpu_rate(-1.0, 1.0, 4), 0.0, "a counter going backwards");
        assert_eq!(cpu_rate(100.0, 1.0, 1), 1.0, "cannot exceed every core");
        assert_eq!(cpu_rate(f64::NAN, 1.0, 1), 0.0);
    }

    #[test]
    fn the_first_sample_reports_no_utilisation_because_it_has_no_predecessor() {
        let sampler = ResourceSampler::new();
        let first = sampler.sample();
        assert_eq!(first.cpu_utilization, 0.0);
        assert!(first.cpu_count >= 1);
    }

    #[test]
    fn a_later_sample_reports_real_memory_use_on_this_machine() {
        let sampler = ResourceSampler::new();
        sampler.sample();
        let second = sampler.sample();

        assert!(second.memory_bytes > 0, "resident memory should be known");
        assert!(second.memory_total_bytes >= second.memory_bytes);
        assert!((0.0..=1.0).contains(&second.memory_utilization));
        assert!((0.0..=1.0).contains(&second.cpu_utilization));
        assert!(second.uptime_secs >= 0.0);
    }
}
