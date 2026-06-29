//! Out-of-band queue for compositing remote-window video frames.
//!
//! Base gpui's `PaintSurface` carries no payload on Windows and lives in the read-only zed checkout,
//! so instead of routing frames through the scene we let an element push them here during paint and
//! have `DirectXRenderer::draw` drain + composite them on top of the scene each frame. Same thread
//! (paint and platform draw run sequentially per frame), so a thread-local queue is sufficient.

use std::cell::RefCell;
use std::sync::Arc;

use gpui::{Bounds, ScaledPixels};

/// One remote-window frame to composite this draw: a BGRA image and where to place it (device px).
pub struct RemoteSurface {
    /// Destination rectangle on the swapchain, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub width: u32,
    pub height: u32,
    /// Bytes per row of `bgra` (may exceed width*4).
    pub stride: u32,
    pub bgra: Arc<Vec<u8>>,
}

thread_local! {
    static PENDING: RefCell<Vec<RemoteSurface>> = const { RefCell::new(Vec::new()) };
}

/// Queue a frame to be composited on the next `DirectXRenderer::draw`. Call from an element's paint.
pub fn push_surface(surface: RemoteSurface) {
    PENDING.with(|p| p.borrow_mut().push(surface));
}

/// Drain the queued frames (called by the renderer).
pub(crate) fn take_surfaces() -> Vec<RemoteSurface> {
    PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()))
}
