/// A failed presentation must survive GPUI clearing its own `needs_present` flag.
/// Stop after ten failures so a permanently unavailable surface cannot spin forever.
#[derive(Default)]
pub(crate) struct FrameRetry {
    failures: u8,
    pending: Option<bool>,
}

impl FrameRetry {
    pub fn record(&mut self, presented: bool, needs_redraw: bool, device_lost: bool) -> bool {
        if presented {
            self.failures = 0;
            self.pending = None;
        } else {
            self.failures = self.failures.saturating_add(1);
        }
        if device_lost || self.failures >= 10 {
            self.pending = None;
        } else if !presented || needs_redraw {
            self.pending = Some(self.pending.unwrap_or(false) || needs_redraw);
        }
        self.pending.is_some()
    }

    /// `Some(force_render)` retries presentation; `None` is an ordinary frame request.
    pub fn take_request(&mut self) -> Option<bool> {
        self.pending.take()
    }
}

#[cfg(test)]
mod tests {
    use super::FrameRetry;

    #[test]
    fn failed_present_retries_without_waiting_for_input() {
        let mut retry = FrameRetry::default();
        assert!(retry.record(false, false, false));
        assert_eq!(retry.take_request(), Some(false));
        assert!(!retry.record(true, false, false));
        assert_eq!(retry.take_request(), None);
    }

    #[test]
    fn atlas_recovery_bypasses_cached_sprites() {
        let mut retry = FrameRetry::default();
        assert!(retry.record(false, true, false));
        assert!(retry.record(false, false, false));
        assert_eq!(retry.take_request(), Some(true));
    }

    #[test]
    fn persistent_failure_is_bounded_and_success_resets_the_budget() {
        let mut retry = FrameRetry::default();
        for _ in 0..9 {
            assert!(retry.record(false, false, false));
            assert_eq!(retry.take_request(), Some(false));
        }
        assert!(!retry.record(false, false, false));
        assert_eq!(retry.take_request(), None);
        assert!(!retry.record(true, false, false));
        assert!(retry.record(false, false, false));
    }

    #[test]
    fn device_loss_stops_pending_retries() {
        let mut retry = FrameRetry::default();
        assert!(retry.record(false, true, false));
        assert!(!retry.record(false, false, true));
        assert_eq!(retry.take_request(), None);
    }
}
