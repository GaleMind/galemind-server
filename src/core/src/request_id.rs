//! Identifiers that follow one request through the server's log.
//!
//! A request is handled by a transport handler, then the service, then a
//! batching dispatcher, then a backend -- four places, on different threads,
//! interleaved with every other request in flight. Without a shared
//! identifier the log of a busy server records what happened but not to whom,
//! which is the one question worth asking when a single request was slow.
//!
//! Ids are generated here rather than taken from a dependency because the
//! requirement is narrow: unique within a process' logs, cheap enough to mint
//! per request, and short enough to read. A per-process prefix plus a counter
//! satisfies all three. Where a caller supplies its own id -- the protocol's
//! `id` field, or an `X-Request-Id` header put on by a gateway -- that id is
//! used instead, so one identifier spans the whole system.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Longest caller-supplied id that will be used as given.
///
/// An id ends up in log lines and in a response header, so an unbounded one
/// from a caller is a way to bloat both.
pub const MAX_REQUEST_ID_LEN: usize = 64;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Distinguishes two runs of the server, and two servers behind one load
/// balancer, so that ids can be compared across collected logs.
fn prefix() -> &'static str {
    use std::sync::OnceLock;
    static PREFIX: OnceLock<String> = OnceLock::new();
    PREFIX.get_or_init(|| {
        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        // The process id alone repeats within a day on a busy host, and the
        // start time alone repeats across hosts that boot together.
        let mixed = start ^ ((std::process::id() as u64) << 32);
        base36(mixed & 0xffff_ffff)
    })
}

/// A new id, unique for the lifetime of this process.
pub fn next_request_id() -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{}", prefix(), base36(n))
}

/// Uses a caller's id when it is usable, and mints one otherwise.
///
/// A caller's id is kept only if it is printable ASCII and short: it is echoed
/// in a response header and written to the log, and neither should carry a
/// newline or a control character from a request.
pub fn request_id_or_new(supplied: Option<&str>) -> String {
    match supplied {
        Some(id) if is_usable(id) => id.to_string(),
        _ => next_request_id(),
    }
}

/// Whether an id can be logged and echoed as it stands.
pub fn is_usable(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_graphic() || b == b' ' || b == b'_')
}

/// Lower-case base 36, which packs an id into the fewest readable characters.
fn base36(mut value: u64) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if value == 0 {
        return "0".to_string();
    }
    let mut out = Vec::with_capacity(13);
    while value > 0 {
        out.push(DIGITS[(value % 36) as usize]);
        value /= 36;
    }
    out.reverse();
    String::from_utf8(out).expect("base 36 digits are ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn generated_ids_do_not_repeat() {
        let ids: HashSet<String> = (0..1_000).map(|_| next_request_id()).collect();
        assert_eq!(ids.len(), 1_000);
    }

    #[test]
    fn generated_ids_share_the_processs_prefix() {
        let first = next_request_id();
        let second = next_request_id();
        let prefix = |id: &str| id.split('-').next().unwrap().to_string();
        assert_eq!(prefix(&first), prefix(&second));
        assert_ne!(first, second);
    }

    #[test]
    fn a_callers_id_is_kept_so_one_id_spans_the_whole_system() {
        assert_eq!(
            request_id_or_new(Some("gateway-7f3a")),
            "gateway-7f3a".to_string()
        );
    }

    /// An id reaches the log and a response header, so a control character or
    /// an unbounded string from a caller is replaced rather than passed on.
    #[test]
    fn an_unusable_caller_id_is_replaced_rather_than_logged() {
        for supplied in [
            "",
            "line\nbreak",
            "tab\there",
            "\u{1b}[31mred",
            &"x".repeat(MAX_REQUEST_ID_LEN + 1),
        ] {
            let id = request_id_or_new(Some(supplied));
            assert_ne!(id, supplied, "{supplied:?} must not be used as given");
            assert!(is_usable(&id), "the replacement is usable: {id:?}");
        }
    }

    #[test]
    fn an_id_at_the_length_limit_is_still_accepted() {
        let id = "x".repeat(MAX_REQUEST_ID_LEN);
        assert_eq!(request_id_or_new(Some(&id)), id);
    }

    #[test]
    fn base_36_counts_up_the_way_a_reader_expects() {
        assert_eq!(base36(0), "0");
        assert_eq!(base36(35), "z");
        assert_eq!(base36(36), "10");
        assert_eq!(base36(1_295), "zz");
    }
}
