//! systemd supervision: `READY=1` once Luma serves, `WATCHDOG=1` while it does.
//!
//! With `Type=notify` and `WatchdogSec=` in the unit, systemd knows Luma is up
//! only after the WAL replay finished and the listener is bound, and restarts
//! it when the heartbeat stops. The heartbeat is sent only after a real request
//! to `/v1/health` succeeds over loopback, through the same listener and
//! runtime that serve clients: a timer thread on its own would keep beating
//! while the request path is wedged, which is exactly the case it is for.
//!
//! The protocol is one datagram to the socket named by `$NOTIFY_SOCKET`; no
//! dependency needed. Outside systemd (no variable) every call is a no-op.

/// Send one notification. Errors are logged, never fatal: a missing
/// supervisor must not stop the server.
pub fn notify(state: &str) {
    #[cfg(unix)]
    {
        let Some(path) = std::env::var_os("NOTIFY_SOCKET") else {
            return;
        };
        let sent = std::os::unix::net::UnixDatagram::unbound()
            .and_then(|socket| socket.send_to(state.as_bytes(), &path));
        if let Err(e) = sent {
            tracing::warn!("sd_notify {state:?} failed: {e}");
        }
    }
    #[cfg(not(unix))]
    let _ = state;
}

/// Tell systemd the server is ready, and start the watchdog heartbeat if the
/// unit asked for one (`$WATCHDOG_USEC`). `probe` is the plain-HTTP listener
/// to check before each beat; `None` (TLS) beats from the runtime alone.
pub fn ready(probe: Option<std::net::SocketAddr>) {
    notify("READY=1");
    let Some(usec) = std::env::var("WATCHDOG_USEC")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    else {
        return;
    };
    // Twice per period, as sd_watchdog_enabled(3) recommends.
    let period = std::time::Duration::from_micros(usec / 2);
    let target = probe.map(loopback);
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(period);
        loop {
            tick.tick().await;
            let alive = match target {
                Some(target) => serves(target).await,
                None => true,
            };
            if alive {
                notify("WATCHDOG=1");
            } else {
                tracing::warn!("watchdog: /v1/health did not answer; withholding the heartbeat");
            }
        }
    });
}

/// The address to reach our own listener: a wildcard bind is reached on
/// loopback.
fn loopback(addr: std::net::SocketAddr) -> std::net::SocketAddr {
    let mut target = addr;
    if addr.ip().is_unspecified() {
        target.set_ip(match addr {
            std::net::SocketAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
            std::net::SocketAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
        });
    }
    target
}

/// Whether `GET /v1/health` answers 200 within five seconds.
async fn serves(target: std::net::SocketAddr) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let attempt = async {
        let mut stream = tokio::net::TcpStream::connect(target).await.ok()?;
        stream
            .write_all(b"GET /v1/health HTTP/1.0\r\nHost: localhost\r\n\r\n")
            .await
            .ok()?;
        let mut head = [0u8; 12];
        stream.read_exact(&mut head).await.ok()?;
        // "HTTP/1.x 200"
        Some(&head[9..] == b"200")
    };
    matches!(
        tokio::time::timeout(std::time::Duration::from_secs(5), attempt).await,
        Ok(Some(true))
    )
}
