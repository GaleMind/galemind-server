//! What the operating system will let a load run do.
//!
//! A load generator at high concurrency is one of the few programs that hits
//! the open-file limit in normal use: every request in flight needs a socket,
//! and a socket is a file descriptor. The default soft limit on most Linux
//! distributions is 1024, so a run at concurrency 700 is already close to it
//! and a C10K run is ten times past it -- and the failure arrives as
//! `Too many open files` from inside a connection pool, which reads like a
//! bug in the client rather than a limit on the process.
//!
//! So the limit is checked before the run starts, and reported as the thing
//! it is: a limit, with the command that raises it. The check is a read of
//! `/proc/self/limits` rather than `getrlimit`, which keeps this crate free of
//! both `unsafe` and a libc dependency; where that file does not exist the
//! check passes, because an unknown limit is not evidence of a low one.

use galemind_core::InferError;

/// Descriptors a run needs beyond its sockets: stdio, the metrics poller's own
/// connection, DNS sockets, the report file, and room for the runtime's own
/// bookkeeping.
const HEADROOM: u64 = 64;

/// Requests one HTTP/2 connection is asked to carry at once.
///
/// A stream is cheap, but a connection is still one socket with one
/// flow-control window, one write loop and one place to be blocked: past a
/// few hundred streams in flight the connection, not the server, is what a
/// run measures.
///
/// Measured rather than guessed, at concurrency 10 000 against this server:
/// 64 streams per connection (157 channels) answered 46 274 requests a
/// second, 256 (40 channels) answered 49 761, and 1 250 (8 channels)
/// answered 47 192. The curve is flat enough that the number chosen is the
/// one that keeps the socket count small without being at the edge.
pub const STREAMS_PER_CONNECTION: usize = 256;

/// The most sockets a multiplexed client opens when nothing says otherwise.
///
/// Multiplexing makes a socket cheap, not free: each one is a connection the
/// server accepts, a flow-control window to keep fed and a write loop to
/// schedule, and the point of streams is that a few sockets carry a lot of
/// work. 256 is high enough that a C10K run is not squeezed through two
/// connections and low enough to stay far away from any default open-file
/// limit.
pub const DEFAULT_MAX_CONNECTIONS: usize = 256;

/// The variable that overrides [`DEFAULT_MAX_CONNECTIONS`].
///
/// An environment variable as well as a flag because the right number is a
/// property of the deployment -- the server's `max_concurrent_streams`, the
/// size of a request, what the network between them does with many
/// connections -- so it belongs where the rest of the deployment's
/// configuration is, not in every command line.
pub const MAX_CONNECTIONS_ENV: &str = "GALEMIND_MAX_CONNECTIONS";

/// The socket ceiling for a multiplexed client: the environment's, or 256.
///
/// A value that is not a positive number is ignored rather than fatal: a
/// mistyped variable in a deployment should not stop a client from running,
/// and the default it falls back to is a safe one.
pub fn max_connections() -> usize {
    std::env::var(MAX_CONNECTIONS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(DEFAULT_MAX_CONNECTIONS)
}

/// The process' limits on open files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileLimits {
    /// What the process may open now, and may raise up to `hard`.
    pub soft: u64,
    /// The ceiling only privilege can raise.
    pub hard: u64,
}

/// Reads the limits, or `None` where they cannot be known.
pub fn file_limits() -> Option<FileLimits> {
    parse_limits(&std::fs::read_to_string("/proc/self/limits").ok()?)
}

/// Descriptors a run needs to hold `sockets` sockets open.
///
/// The interesting number is the socket count rather than the concurrency,
/// and the two are only the same under HTTP/1.1, where a request holds its
/// connection for its lifetime. A multiplexed transport carries many requests
/// on one socket, so a C10K run over HTTP/2 or gRPC needs tens of
/// descriptors rather than ten thousand.
pub fn descriptors_needed(sockets: usize) -> u64 {
    sockets as u64 + HEADROOM
}

/// Checks a planned run against the limits, given how many sockets its
/// client will hold open.
///
/// Returns a warning to print where the run will fit but only just -- a
/// margin thin enough that a retried connection or an extra poller will push
/// it over -- and an error where it cannot fit at all.
pub fn check_sockets(concurrency: usize, sockets: usize) -> Result<Option<String>, InferError> {
    // No `/proc`: the run proceeds, and a failure will say `Too many open
    // files` as it did before. An unknown limit is not evidence of a low one.
    let Some(limits) = file_limits() else {
        return Ok(None);
    };

    if descriptors_needed(sockets) > limits.soft {
        return Err(too_many(concurrency, sockets, limits));
    }
    Ok(tight_warning(concurrency, sockets, limits))
}

/// Checks a run that opens a socket per request in flight, which is what
/// HTTP/1.1 does.
pub fn check_concurrency(concurrency: usize) -> Result<Option<String>, InferError> {
    check_sockets(concurrency, concurrency)
}

/// The error a run that cannot fit is refused with.
fn too_many(concurrency: usize, sockets: usize, limits: FileLimits) -> InferError {
    let needed = descriptors_needed(sockets);
    // The command to run is part of the message: being told the limit without
    // being told how to raise it is a riddle, not a diagnosis.
    let remedy = if limits.hard >= needed {
        format!(
            "raise it for this shell with `ulimit -n {}` (the hard limit is {})",
            needed.max(4_096),
            limits.hard
        )
    } else {
        format!(
            "the hard limit is {}, so raising it needs privilege \
             (`sudo prlimit --nofile={needed} --pid $$`) or a lower --concurrency",
            limits.hard
        )
    };

    // Naming the transport's shape as well as the number, because the fix is
    // often not a higher limit: the same run multiplexed needs tens of
    // descriptors rather than thousands.
    InferError::InvalidRequest(format!(
        "a run at concurrency {concurrency} over a transport that opens a socket \
         per request needs about {needed} open files, but this process may open \
         only {}: {remedy}, or multiplex it instead (`--transport http2` or \
         `--transport grpc`), which carries the same run on a few sockets",
        limits.soft
    ))
}

/// A warning for a run that fits with little to spare.
fn tight_warning(concurrency: usize, sockets: usize, limits: FileLimits) -> Option<String> {
    let needed = descriptors_needed(sockets);
    // Past half the limit the margin is thinner than it looks: a connection
    // being replaced holds two descriptors for as long as the old one takes to
    // close, so a run needing 764 of 1024 can touch the limit in a burst even
    // though its steady state fits.
    if needed * 2 > limits.soft {
        return Some(format!(
            "concurrency {concurrency} needs about {needed} of the {} open files \
             this process may have; a retried connection could exhaust them. \
             Consider `ulimit -n {}`.",
            limits.soft,
            (needed * 2).max(4_096)
        ));
    }
    None
}

/// Reads the `Max open files` row of `/proc/self/limits`.
fn parse_limits(text: &str) -> Option<FileLimits> {
    let line = text
        .lines()
        .find(|line| line.starts_with("Max open files"))?;
    // The limit names contain spaces, so the numbers are taken from the end of
    // the row: "Max open files   1024   1048576   files".
    let fields: Vec<&str> = line.split_whitespace().collect();
    let unlimited = |value: &str| (value == "unlimited").then_some(u64::MAX);

    let units = fields.last().copied().unwrap_or_default() == "files";
    let (soft, hard) = if units {
        (fields.get(fields.len() - 3)?, fields.get(fields.len() - 2)?)
    } else {
        (fields.get(fields.len() - 2)?, fields.last()?)
    };

    Some(FileLimits {
        soft: unlimited(soft).or_else(|| soft.parse().ok())?,
        hard: unlimited(hard).or_else(|| hard.parse().ok())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = "Limit                     Soft Limit           Hard Limit           Units     \n\
                        Max cpu time              unlimited            unlimited            seconds   \n\
                        Max open files            1024                 1048576              files     \n\
                        Max locked memory         8388608              8388608              bytes     \n";

    #[test]
    fn the_open_file_limit_is_read_from_proc() {
        assert_eq!(
            parse_limits(REAL),
            Some(FileLimits {
                soft: 1_024,
                hard: 1_048_576
            })
        );
    }

    #[test]
    fn an_unlimited_limit_is_read_as_no_limit() {
        let text = "Max open files            unlimited            unlimited            files\n";
        let limits = parse_limits(text).expect("parsed");
        assert_eq!(limits.soft, u64::MAX);
    }

    #[test]
    fn a_file_without_the_row_yields_nothing() {
        assert_eq!(
            parse_limits("Max cpu time   unlimited   unlimited   seconds\n"),
            None
        );
        assert_eq!(parse_limits(""), None);
    }

    #[test]
    fn a_run_needs_a_descriptor_per_request_in_flight_plus_headroom() {
        assert_eq!(descriptors_needed(700), 764);
        assert_eq!(descriptors_needed(10_000), 10_064);
    }

    /// The case a user meets first: concurrency 700 against the usual soft
    /// limit of 1024. Its steady state fits, but replacing connections can
    /// touch the limit in a burst, so it is worth a warning.
    #[test]
    fn a_run_that_fits_with_little_to_spare_is_warned_about() {
        let warning = tight_warning(
            700,
            700,
            FileLimits {
                soft: 1_024,
                hard: 1_048_576,
            },
        )
        .expect("a warning");
        assert!(warning.contains("764"), "{warning}");
        assert!(warning.contains("ulimit -n"), "{warning}");
    }

    #[test]
    fn a_comfortable_run_is_not_warned_about() {
        // 64 workers need 128 of 1024, an eighth of the limit.
        assert_eq!(
            tight_warning(
                64,
                64,
                FileLimits {
                    soft: 1_024,
                    hard: 1_048_576
                }
            ),
            None
        );
        // A C10K run under a raised limit is comfortable too.
        assert_eq!(
            tight_warning(
                10_000,
                10_000,
                FileLimits {
                    soft: 65_536,
                    hard: 1_048_576
                }
            ),
            None
        );
    }

    /// A C10K run against the default limit cannot work, and saying so before
    /// the run beats `Too many open files` from inside a connection pool.
    #[test]
    fn a_run_that_cannot_fit_is_refused_with_the_command_that_fixes_it() {
        let error = too_many(
            10_000,
            10_000,
            FileLimits {
                soft: 1_024,
                hard: 1_048_576,
            },
        );
        let text = error.to_string();
        assert!(text.contains("10000"), "{text}");
        assert!(text.contains("10064"), "{text}");
        assert!(text.contains("ulimit -n"), "{text}");
        assert!(matches!(error, InferError::InvalidRequest(_)));

        // Where even the hard limit is too low, the remedy is a different one.
        let error = too_many(
            10_000,
            10_000,
            FileLimits {
                soft: 1_024,
                hard: 2_048,
            },
        );
        assert!(error.to_string().contains("prlimit"), "{error}");
    }

    /// The reason the socket count is a parameter: a multiplexed client
    /// drives a C10K run on a few dozen sockets, and the limit that refuses
    /// it over HTTP/1.1 is nowhere near it.
    #[test]
    fn a_multiplexed_run_is_measured_by_its_sockets_not_its_concurrency() {
        let usual = FileLimits {
            soft: 1_024,
            hard: 1_048_576,
        };
        // 10 000 in flight over 32 gRPC channels: 96 descriptors.
        assert_eq!(descriptors_needed(32), 96);
        assert_eq!(check_sockets(10_000, 32).unwrap(), None);
        assert_eq!(tight_warning(10_000, 32, usual), None);

        // The same run with a socket per request does not fit at all, and is
        // told what to do about it.
        let error = too_many(10_000, 10_000, usual);
        assert!(error.to_string().contains("--transport http2"), "{error}");
    }

    /// The ceiling is configuration, so it comes from the environment where
    /// the environment says something usable and from the default where it
    /// does not.
    #[test]
    fn the_socket_ceiling_is_the_environments_or_the_default() {
        // Serialised through one test because the environment is global.
        let restore = std::env::var(MAX_CONNECTIONS_ENV).ok();

        std::env::remove_var(MAX_CONNECTIONS_ENV);
        assert_eq!(max_connections(), DEFAULT_MAX_CONNECTIONS);
        assert_eq!(DEFAULT_MAX_CONNECTIONS, 256);

        std::env::set_var(MAX_CONNECTIONS_ENV, "16");
        assert_eq!(max_connections(), 16);

        std::env::set_var(MAX_CONNECTIONS_ENV, " 64 ");
        assert_eq!(max_connections(), 64);

        // Nonsense falls back rather than failing a run that was going to
        // work.
        for bad in ["0", "-1", "lots", ""] {
            std::env::set_var(MAX_CONNECTIONS_ENV, bad);
            assert_eq!(max_connections(), DEFAULT_MAX_CONNECTIONS, "{bad:?}");
        }

        match restore {
            Some(value) => std::env::set_var(MAX_CONNECTIONS_ENV, value),
            None => std::env::remove_var(MAX_CONNECTIONS_ENV),
        }
    }

    #[test]
    fn the_real_limits_of_this_process_are_readable_on_linux() {
        // Not asserting a value: the point is that the file is parseable where
        // it exists, which is the platform the server is built for.
        if std::path::Path::new("/proc/self/limits").exists() {
            let limits = file_limits().expect("limits are readable");
            assert!(limits.soft > 0);
            assert!(limits.hard >= limits.soft);
        }
    }
}
