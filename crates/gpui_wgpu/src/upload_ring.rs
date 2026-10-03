use std::sync::{Arc, atomic::{AtomicU8, Ordering}};

pub(crate) struct UploadRing {
    slots: [Slot; 3],
    capacity: u64,
}
struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
}
impl UploadRing {
    pub(crate) fn new(device: &wgpu::Device, capacity: u64) -> Self {
        Self { capacity, slots: std::array::from_fn(|_| Slot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpui_persistent_upload"), size: capacity,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }),
            state: Arc::new(AtomicU8::new(0)),
        }) }
    }
    pub(crate) fn capacity(&self) -> u64 { self.capacity }
    pub(crate) fn acquire(&self) -> Option<usize> {
        self.slots.iter().position(|slot| slot.state.load(Ordering::Acquire) == 0)
    }
    pub(crate) fn write(&self, index: usize, offset: u64, bytes: &[u8]) -> anyhow::Result<()> {
        let slot = self.slots.get(index).ok_or_else(|| anyhow::anyhow!("Invalid upload ring slot"))?;
        anyhow::ensure!(slot.state.load(Ordering::Acquire) == 0, "Upload slot is not mapped");
        let start = usize::try_from(offset)?;
        let end = start.checked_add(bytes.len()).ok_or_else(|| anyhow::anyhow!("Upload range overflow"))?;
        anyhow::ensure!(end as u64 <= self.capacity, "Upload range exceeds ring capacity");
        // Whole-buffer mappings avoid Vulkan/wgpu subrange alignment restrictions.
        let mut mapped = slot.buffer.get_mapped_range_mut(..);
        anyhow::ensure!(end <= mapped.len(), "Upload range exceeds mapping");
        mapped.slice(start..end).copy_from_slice(bytes);
        Ok(())
    }
    pub(crate) fn buffer(&self, index: usize) -> &wgpu::Buffer { &self.slots[index].buffer }
    pub(crate) fn unmap(&self, index: usize) { self.slots[index].buffer.unmap(); }
    pub(crate) fn submitted(&self, index: usize) {
        let slot = &self.slots[index];
        slot.state.store(1, Ordering::Release);
        let state = slot.state.clone();
        slot.buffer.map_async(wgpu::MapMode::Write, .., move |result| {
            // A failed mapping cannot be reused; device recovery replaces the ring.
            state.store(if result.is_ok() { 0 } else { 2 }, Ordering::Release);
            if let Err(error) = result { log::warn!("GPUI upload ring mapping failed: {error}"); }
        });
    }
    pub(crate) fn failed(&self) -> bool {
        self.slots.iter().any(|slot| slot.state.load(Ordering::Acquire) == 2)
    }
}
