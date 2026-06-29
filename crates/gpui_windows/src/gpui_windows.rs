#![cfg(target_os = "windows")]

mod clipboard;
mod destination_list;
mod direct_manipulation;
mod direct_write;
mod directx_atlas;
mod directx_devices;
mod directx_renderer;
mod dispatcher;
mod display;
mod events;
mod keyboard;
mod platform;
pub mod remote_surface;
mod system_settings;
mod util;
mod vsync;
mod window;
mod wrapper;

pub(crate) use clipboard::*;
pub(crate) use destination_list::*;
pub(crate) use direct_write::*;
pub(crate) use directx_atlas::*;
pub(crate) use directx_devices::*;
pub(crate) use directx_renderer::*;
pub(crate) use dispatcher::*;
pub(crate) use display::*;
pub(crate) use events::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
pub(crate) use system_settings::*;
pub(crate) use util::*;
pub(crate) use vsync::*;
pub(crate) use window::*;
pub(crate) use wrapper::*;

pub use platform::WindowsPlatform;
pub use remote_surface::{RemoteSurface, push_surface};

pub(crate) use windows::Win32::Foundation::HWND;

/// Set the Windows immersive dark-mode flag (the DWM-painted title bar / window frame) on every
/// top-level window owned by this process. Lets the OS frame follow the active ced theme's
/// light/dark kind instead of only the OS-wide system setting. No-op failures are ignored.
pub fn set_app_windows_dark_mode(dark: bool) {
    use windows::Win32::Foundation::LPARAM;
    use windows::Win32::Graphics::Dwm::{DWMWA_USE_IMMERSIVE_DARK_MODE, DwmSetWindowAttribute};
    use windows::Win32::System::Threading::GetCurrentProcessId;
    use windows::Win32::UI::WindowsAndMessaging::{EnumWindows, GetWindowThreadProcessId};
    use windows::core::BOOL;

    unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let dark = lparam.0 != 0;
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
        if pid == unsafe { GetCurrentProcessId() } {
            let flag: BOOL = dark.into();
            unsafe {
                let _ = DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    &flag as *const _ as _,
                    std::mem::size_of::<BOOL>() as u32,
                );
            }
        }
        BOOL(1) // continue enumeration
    }

    unsafe {
        let _ = EnumWindows(Some(enum_proc), LPARAM(dark as isize));
    }
}
