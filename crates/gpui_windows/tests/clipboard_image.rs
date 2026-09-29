#![cfg(target_os = "windows")]

use gpui::{ClipboardEntry, ClipboardItem, Image, ImageFormat, Platform};
use gpui_windows::WindowsPlatform;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use windows::Win32::{
    Foundation::HGLOBAL,
    System::{
        DataExchange::{
            CloseClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
            RegisterClipboardFormatW,
        },
        Memory::{GlobalLock, GlobalSize, GlobalUnlock},
        Ole::{CF_DIB, CF_DIBV5},
    },
};

struct OpenedClipboard;

impl OpenedClipboard {
    fn open() -> Self {
        unsafe { OpenClipboard(None).expect("clipboard should open after writing") };
        Self
    }

    fn read(&self, format: u32) -> Vec<u8> {
        let handle =
            unsafe { GetClipboardData(format).expect("clipboard format should be readable") };
        let global = HGLOBAL(handle.0);
        let length = unsafe { GlobalSize(global) };
        let ptr = unsafe { GlobalLock(global) } as *const u8;
        assert!(!ptr.is_null());
        let bytes = unsafe { std::slice::from_raw_parts(ptr, length) }.to_vec();
        unsafe { GlobalUnlock(global).expect("clipboard data should unlock") };
        bytes
    }
}

impl Drop for OpenedClipboard {
    fn drop(&mut self) {
        unsafe { CloseClipboard().expect("clipboard should close") };
    }
}

#[test]
fn png_copy_exposes_standard_bitmaps_and_preserves_png_roundtrip() {
    let image = DynamicImage::ImageRgba8(ImageBuffer::from_fn(2, 2, |column, row| {
        match (column, row) {
            (0, 0) => Rgba([255, 0, 0, 128]),
            (1, 0) => Rgba([0, 255, 0, 255]),
            (0, 1) => Rgba([0, 0, 255, 255]),
            _ => Rgba([255, 255, 0, 64]),
        }
    }));
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let item = ClipboardItem {
        entries: vec![ClipboardEntry::Image(Image::from_bytes(
            ImageFormat::Png,
            png.clone(),
        ))],
    };
    let platform = WindowsPlatform::new(true).unwrap();
    platform.write_to_clipboard(item.clone());

    {
        let clipboard = OpenedClipboard::open();
        let png_format = unsafe { RegisterClipboardFormatW(windows::core::w!("PNG")) };
        assert_ne!(png_format, 0);
        assert!(unsafe { IsClipboardFormatAvailable(png_format).is_ok() });
        assert!(unsafe { IsClipboardFormatAvailable(CF_DIB.0 as u32).is_ok() });
        assert!(unsafe { IsClipboardFormatAvailable(CF_DIBV5.0 as u32).is_ok() });
        assert_eq!(clipboard.read(png_format), png);
        let dib = clipboard.read(CF_DIB.0 as u32);
        assert_eq!(u32::from_le_bytes(dib[0..4].try_into().unwrap()), 40);
        assert_eq!(i32::from_le_bytes(dib[8..12].try_into().unwrap()), 2);
        assert_eq!(&dib[40..48], &[255, 0, 0, 255, 0, 255, 255, 64]);
        let mut bmp = Vec::with_capacity(14 + dib.len());
        bmp.extend_from_slice(b"BM");
        bmp.extend_from_slice(&(14 + dib.len() as u32).to_le_bytes());
        bmp.extend_from_slice(&[0; 4]);
        bmp.extend_from_slice(&54u32.to_le_bytes());
        bmp.extend_from_slice(&dib);
        let decoded = image::load_from_memory_with_format(&bmp, image::ImageFormat::Bmp).unwrap();
        assert_eq!(decoded.get_pixel(1, 0).0[..3], [0, 255, 0]);
        assert_eq!(decoded.get_pixel(0, 1).0[..3], [0, 0, 255]);
        let dibv5 = clipboard.read(CF_DIBV5.0 as u32);
        assert_eq!(u32::from_le_bytes(dibv5[0..4].try_into().unwrap()), 124);
        assert_eq!(&dibv5[52..56], &0xff000000u32.to_le_bytes());
        assert_eq!(&dibv5[132..136], &[0, 0, 255, 128]);
    }
    assert_eq!(platform.read_from_clipboard(), Some(item));
}
