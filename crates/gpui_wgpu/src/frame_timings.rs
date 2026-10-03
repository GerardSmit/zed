use std::sync::{Arc, atomic::{AtomicU8, Ordering}};
use std::cell::Cell;

const MAX_PASSES: usize = 32;
const QUERIES_PER_FRAME: u32 = 3 + MAX_PASSES as u32 * 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuPassTimings {
    pub samples: u64,
    pub total_nanoseconds: u64,
    pub maximum_nanoseconds: u64,
}

#[derive(Clone, Copy)]
pub(crate) enum PassKind { Path = 1, Layer = 2, Blur = 3 }

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuFrameTimings {
    pub samples: u64,
    pub total_nanoseconds: u64,
    pub maximum_nanoseconds: u64,
    pub intermediate_nanoseconds: u64,
    pub root_nanoseconds: u64,
    pub paths: GpuPassTimings,
    pub layers: GpuPassTimings,
    pub blur: GpuPassTimings,
    pub unmeasured_passes: u64,
}

struct Slot {
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    // 0 = reusable, 1 = mapping, 2 = mapped, 3 = mapping failed.
    state: Arc<AtomicU8>,
    kinds: Cell<[u8; MAX_PASSES]>,
    count: Cell<u32>,
    unmeasured: Cell<u64>,
}

pub(crate) struct FrameTimings {
    queries: wgpu::QuerySet,
    slots: [Slot; 3],
    period: f64,
    completed: GpuFrameTimings,
    active: Cell<Option<usize>>,
}

impl FrameTimings {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let period = f64::from(queue.get_timestamp_period());
        if !device.features().contains(features) || !period.is_finite() || period <= 0.0 { return None; }
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpui_frame_timestamps"), ty: wgpu::QueryType::Timestamp, count: QUERIES_PER_FRAME * 3,
            }),
            slots: std::array::from_fn(|_| Slot {
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpui_timestamp_resolve"), size: 768,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpui_timestamp_readback"), size: u64::from(QUERIES_PER_FRAME) * 8,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(0)),
                kinds: Cell::new([0; MAX_PASSES]), count: Cell::new(0), unmeasured: Cell::new(0),
            }),
            period, completed: GpuFrameTimings::default(), active: Cell::new(None),
        })
    }

    fn reap(&mut self) {
        for slot in &self.slots {
            match slot.state.load(Ordering::Acquire) {
                2 => {
                    let bytes = slot.readback.get_mapped_range(..);
                    if let (Some(start), Some(middle), Some(end)) = (bytes.get(..8), bytes.get(8..16), bytes.get(16..24)) {
                        if let (Ok(start), Ok(middle), Ok(end)) = (start.try_into(), middle.try_into(), end.try_into()) {
                            let start = u64::from_le_bytes(start);
                            let middle = u64::from_le_bytes(middle);
                            let end = u64::from_le_bytes(end);
                            // Reject invalid/disjoint results instead of manufacturing a duration.
                            if let Some(ticks) = end.checked_sub(start) {
                                let duration = ticks as f64 * self.period;
                                if duration.is_finite() && duration <= u64::MAX as f64 && (start..=end).contains(&middle) {
                                    let duration = duration as u64;
                                    self.completed.samples = self.completed.samples.saturating_add(1);
                                    self.completed.total_nanoseconds = self.completed.total_nanoseconds.saturating_add(duration);
                                    self.completed.maximum_nanoseconds = self.completed.maximum_nanoseconds.max(duration);
                                    self.completed.intermediate_nanoseconds = self.completed.intermediate_nanoseconds.saturating_add(((middle - start) as f64 * self.period) as u64);
                                    self.completed.root_nanoseconds = self.completed.root_nanoseconds.saturating_add(((end - middle) as f64 * self.period) as u64);
                                    self.completed.unmeasured_passes = self.completed.unmeasured_passes.saturating_add(slot.unmeasured.get());
                                    for (index, kind) in slot.kinds.get().iter().take(slot.count.get() as usize).enumerate() {
                                        let offset = (3 + index * 2) * 8;
                                        let pair = &bytes[offset..offset + 16];
                                        let mut first = [0; 8];
                                        let mut last = [0; 8];
                                        first.copy_from_slice(&pair[..8]);
                                        last.copy_from_slice(&pair[8..]);
                                        let first = u64::from_le_bytes(first);
                                        let last = u64::from_le_bytes(last);
                                        if first < start || last > end { continue; }
                                        let Some(ticks) = last.checked_sub(first) else { continue; };
                                        let duration = (ticks as f64 * self.period) as u64;
                                        let category = match kind {
                                            1 => &mut self.completed.paths,
                                            2 => &mut self.completed.layers,
                                            3 => &mut self.completed.blur,
                                            _ => continue,
                                        };
                                        category.samples = category.samples.saturating_add(1);
                                        category.total_nanoseconds = category.total_nanoseconds.saturating_add(duration);
                                        category.maximum_nanoseconds = category.maximum_nanoseconds.max(duration);
                                    }
                                }
                            }
                        }
                    }
                    drop(bytes);
                    slot.readback.unmap();
                    slot.state.store(0, Ordering::Release);
                }
                3 => {
                    log::warn!("GPUI GPU timestamp readback failed");
                    slot.readback.unmap();
                    slot.state.store(0, Ordering::Release);
                }
                _ => {}
            }
        }
    }

    pub(crate) fn begin(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<usize> {
        self.reap();
        self.active.set(None);
        let index = self.slots.iter().position(|slot| slot.state.load(Ordering::Acquire) == 0)?;
        self.slots[index].count.set(0);
        self.slots[index].unmeasured.set(0);
        self.active.set(Some(index));
        encoder.write_timestamp(&self.queries, index as u32 * QUERIES_PER_FRAME);
        Some(index)
    }

    pub(crate) fn intermediate_end(&self, index: usize, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.queries, index as u32 * QUERIES_PER_FRAME + 1);
    }

    pub(crate) fn end(&self, index: usize, encoder: &mut wgpu::CommandEncoder) {
        let slot = &self.slots[index];
        let query = index as u32 * QUERIES_PER_FRAME;
        let count = 3 + slot.count.get() * 2;
        encoder.write_timestamp(&self.queries, query + 2);
        encoder.resolve_query_set(&self.queries, query..query + count, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, u64::from(count) * 8);
        self.active.set(None);
    }

    pub(crate) fn begin_pass(&self, kind: PassKind, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        let index = self.active.get()?;
        let slot = &self.slots[index];
        let count = slot.count.get();
        if count as usize >= MAX_PASSES {
            slot.unmeasured.set(slot.unmeasured.get().saturating_add(1));
            return None;
        }
        let mut kinds = slot.kinds.get();
        kinds[count as usize] = kind as u8;
        slot.kinds.set(kinds);
        slot.count.set(count + 1);
        let query = index as u32 * QUERIES_PER_FRAME + 3 + count * 2;
        encoder.write_timestamp(&self.queries, query);
        Some(query)
    }

    pub(crate) fn end_pass(&self, query: u32, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.queries, query + 1);
    }

    pub(crate) fn cancel(&self) { self.active.set(None); }

    pub(crate) fn submitted(&self, index: usize) {
        let slot = &self.slots[index];
        slot.state.store(1, Ordering::Release);
        let state = slot.state.clone();
        slot.readback.map_async(wgpu::MapMode::Read, .., move |result| {
            state.store(if result.is_ok() { 2 } else { 3 }, Ordering::Release);
        });
    }

    pub(crate) fn take(&mut self) -> GpuFrameTimings {
        self.reap();
        std::mem::take(&mut self.completed)
    }
}
