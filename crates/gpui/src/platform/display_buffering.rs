//! Device-local Metal buffering preference and per-window stall detection.
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// Presentation policy for native Metal windows. Other backends ignore this preference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DisplayBuffering {
    /// Start with two buffers and promote after repeated drawable-acquisition stalls.
    #[default]
    Auto,
    /// Always request two buffers.
    Double,
    /// Always request three buffers.
    Triple,
}

static DISPLAY_BUFFERING: AtomicU8 = AtomicU8::new(0);

/// Set the process-wide preference. Refresh windows to apply it on their next draw.
pub fn set_display_buffering(mode: DisplayBuffering) {
    DISPLAY_BUFFERING.store(mode as u8, Ordering::Relaxed);
}

/// Read the process-wide preference without allocating or performing I/O.
pub fn display_buffering() -> DisplayBuffering {
    match DISPLAY_BUFFERING.load(Ordering::Relaxed) {
        1 => DisplayBuffering::Double,
        2 => DisplayBuffering::Triple,
        _ => DisplayBuffering::Auto,
    }
}

/// Constant-size per-window policy. Auto promotion lasts until the mode changes or window closes.
/// Six waits over 20 ms within two seconds of active drawing trigger promotion. The first frame
/// after an idle gap (>250 ms) is ignored, so wakeup and isolated resize stalls do not accumulate.
#[derive(Debug)]
pub struct DisplayBufferingState {
    mode: DisplayBuffering,
    promoted: bool,
    last_frame: Option<Instant>,
    interval_start: Option<Instant>,
    stalls: u8,
}

impl DisplayBufferingState {
    /// Start a new window at the requested policy.
    pub fn new(mode: DisplayBuffering) -> Self {
        Self {
            mode,
            promoted: false,
            last_frame: None,
            interval_start: None,
            stalls: 0,
        }
    }

    /// Apply an explicit policy change, resetting Auto's observation history.
    pub fn set_mode(&mut self, mode: DisplayBuffering) {
        if self.mode != mode {
            *self = Self::new(mode);
        }
    }

    /// Requested drawable count, always supported by CAMetalLayer (two or three).
    pub fn drawable_count(&self) -> u64 {
        if self.mode == DisplayBuffering::Triple || self.promoted {
            3
        } else {
            2
        }
    }

    /// Record only the time blocked acquiring a drawable, not CPU rendering or idle time.
    pub fn observe_wait(&mut self, now: Instant, wait: Duration) {
        if self.mode != DisplayBuffering::Auto || self.promoted {
            return;
        }
        let idle = self
            .last_frame
            .is_none_or(|last| now.saturating_duration_since(last) > Duration::from_millis(250));
        // Exclude time spent blocked from the next frame's idle-gap calculation.
        self.last_frame = Some(now + wait);
        if idle
            || self
                .interval_start
                .is_none_or(|start| now.duration_since(start) > Duration::from_secs(2))
        {
            self.interval_start = Some(now);
            self.stalls = 0;
        }
        if !idle && wait > Duration::from_millis(20) {
            self.stalls += 1;
            self.promoted = self.stalls >= 6;
        }
    }
}
