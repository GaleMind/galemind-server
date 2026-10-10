//! Listening sockets, with a backlog big enough for a connection storm.
//!
//! Shared by both transports because both have the same problem: a server
//! that is fine at a hundred connections and drops them at ten thousand, for
//! a reason that is in neither the server's code nor its logs.
//!
//! # The accept queue
//!
//! A listening socket has a queue of connections the kernel has completed the
//! handshake for and the process has not yet accepted. Its size is the
//! `backlog` argument to `listen`, and when it is full the kernel drops
//! further connection attempts -- silently, as far as the server is
//! concerned, while the client sees a connection that never completes or is
//! refused.
//!
//! `tokio::net::TcpListener::bind` asks for 1024. That is ample for a server
//! whose clients arrive steadily, and not ample for one that ten thousand
//! clients connect to at once: measured here, a run of ten processes opening
//! a thousand connections each cost 139 632 `ListenOverflows` and failed 97%
//! of its requests, with the server idle at 4% CPU. The queue, not the
//! server, was the bottleneck -- and the counter that says so,
//! `TcpExtListenOverflows` in `nstat`, is nowhere near where anyone looks
//! first.
//!
//! So the backlog is ours to choose and is raised to
//! [`DEFAULT_LISTEN_BACKLOG`]. The kernel caps whatever is asked for at
//! `/proc/sys/net/core/somaxconn` -- commonly 4096 on current kernels and 128
//! on older ones -- so a deployment expecting a storm raises that too:
//!
//! ```text
//! sysctl -w net.core.somaxconn=16384
//! ```

use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpSocket};

/// Connections the kernel may hold for us before refusing more.
///
/// Four thousand is the common `somaxconn`, so this asks for what a current
/// kernel will actually grant and leaves raising both to a deployment that
/// needs more.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 4_096;

/// Binds a listening socket with an explicit accept-queue depth.
///
/// Equivalent to `TcpListener::bind` but for the backlog, which is the point.
/// `SO_REUSEADDR` is set because `TcpListener::bind` sets it, and a restart
/// that cannot rebind its own address while the old sockets drain is a
/// restart that fails.
pub async fn bind_with_backlog(address: SocketAddr, backlog: u32) -> std::io::Result<TcpListener> {
    let socket = if address.is_ipv4() {
        TcpSocket::new_v4()?
    } else {
        TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    socket.bind(address)?;
    // At least one: a backlog of zero is a listener that accepts nothing,
    // which is never what a configured value meant.
    socket.listen(backlog.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The socket has to work, whatever the backlog: the queue depth is not
    /// visible from inside the process, so what is asserted is that binding
    /// and connecting still do what they did.
    #[tokio::test]
    async fn a_listener_with_a_raised_backlog_still_accepts_connections() {
        let listener = bind_with_backlog("127.0.0.1:0".parse().unwrap(), 8_192)
            .await
            .expect("binds");
        let address = listener.local_addr().expect("an address");

        let client = tokio::spawn(async move { tokio::net::TcpStream::connect(address).await });
        let (_accepted, from) = listener.accept().await.expect("accepts");
        assert_eq!(from.ip(), address.ip());
        client.await.expect("the task").expect("connects");
    }

    #[tokio::test]
    async fn a_backlog_of_zero_is_raised_to_something_that_listens() {
        let listener = bind_with_backlog("127.0.0.1:0".parse().unwrap(), 0)
            .await
            .expect("binds");
        let address = listener.local_addr().expect("an address");
        let client = tokio::spawn(async move { tokio::net::TcpStream::connect(address).await });
        listener.accept().await.expect("accepts");
        client.await.expect("the task").expect("connects");
    }

    #[tokio::test]
    async fn an_ipv6_address_binds_too() {
        match bind_with_backlog("[::1]:0".parse().unwrap(), 128).await {
            Ok(listener) => {
                assert!(listener.local_addr().expect("an address").is_ipv6());
            }
            // A machine without IPv6 is not a failing test.
            Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::AddrNotAvailable, "{e}"),
        }
    }
}
