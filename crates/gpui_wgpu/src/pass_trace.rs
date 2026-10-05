//! Record of the passes and copies encoded for a frame, for relating a frame to the GPU jobs a
//! driver runs for it. Recording is off unless a caller asks for it.
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ENTRIES: Mutex<Vec<String>> = Mutex::new(Vec::new());
static TRANSFERS: Mutex<[(u32, u64); TRANSFER_KINDS]> = Mutex::new([(0, 0); TRANSFER_KINDS]);

/// What a copy command is for. Each copy region (and each zero-fill region of a lazily
/// initialized resource) is one transfer job on PowerVR, whose Mesa driver kicks the
/// transfer queue once per region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    /// Glyph and image tiles copied into atlas textures.
    Atlas,
    /// The zero fill wgpu gives an atlas texture before its first partial copy.
    AtlasInit,
    /// Pixels a rotating target missed, copied from the target on screen.
    CopyForward,
    /// Instance data and uniforms copied from the upload ring.
    Buffer,
    /// Rendered rows copied into a scanout buffer.
    Scanout,
    Other,
}

const TRANSFER_KINDS: usize = 6;
const TRANSFER_NAMES: [&str; TRANSFER_KINDS] =
    ["atlas", "atlas_init", "copy_forward", "buffer", "scanout", "other"];

/// Records everything encoded until [`finish`].
pub fn begin() {
    ENTRIES.lock().unwrap().clear();
    *TRANSFERS.lock().unwrap() = [(0, 0); TRANSFER_KINDS];
    ACTIVE.store(true, Ordering::Relaxed);
}

/// The recorded entries, followed by a `transfers` line when the frame encoded any copy.
pub fn finish() -> Vec<String> {
    ACTIVE.store(false, Ordering::Relaxed);
    let mut entries = std::mem::take(&mut *ENTRIES.lock().unwrap());
    let transfers = std::mem::replace(&mut *TRANSFERS.lock().unwrap(), [(0, 0); TRANSFER_KINDS]);
    if let Some(line) = transfer_summary(&transfers) {
        entries.push(line);
    }
    entries
}

pub fn note(entry: impl FnOnce() -> String) {
    if ACTIVE.load(Ordering::Relaxed) {
        ENTRIES.lock().unwrap().push(entry());
    }
}

/// Counts `regions` copy regions of `bytes` in total.
pub fn transfer(kind: Transfer, regions: u32, bytes: u64) {
    if ACTIVE.load(Ordering::Relaxed) && regions != 0 {
        let index = kind as usize;
        let mut transfers = TRANSFERS.lock().unwrap();
        transfers[index].0 += regions;
        transfers[index].1 += bytes;
    }
}

fn transfer_summary(transfers: &[(u32, u64); TRANSFER_KINDS]) -> Option<String> {
    let total: u32 = transfers.iter().map(|(regions, _)| regions).sum();
    if total == 0 {
        return None;
    }
    let mut line = format!("transfers regions={total}");
    for (name, (regions, bytes)) in TRANSFER_NAMES.iter().zip(transfers) {
        line.push_str(&format!(" {name}={regions}/{bytes}"));
    }
    Some(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_summary_lists_every_kind() {
        let mut transfers = [(0, 0); TRANSFER_KINDS];
        assert_eq!(transfer_summary(&transfers), None);
        transfers[Transfer::Atlas as usize] = (2, 4096);
        transfers[Transfer::CopyForward as usize] = (1, 64);
        assert_eq!(
            transfer_summary(&transfers).unwrap(),
            "transfers regions=3 atlas=2/4096 atlas_init=0/0 copy_forward=1/64 buffer=0/0 scanout=0/0 other=0/0"
        );
    }
}
