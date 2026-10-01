#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

fn main() {
    #[cfg(target_os = "macos")]
    macos_build::run();
}

#[cfg(target_os = "macos")]
mod macos_build {
    use std::{
        env,
        path::{Path, PathBuf},
    };

    use cbindgen::Config;

    pub fn run() {
        let header_path = generate_shader_bindings();

        #[cfg(feature = "runtime_shaders")]
        {
            emit_stitched_shaders(&header_path);
            translate_wgsl_shaders();
        }
        #[cfg(not(feature = "runtime_shaders"))]
        {
            let shader_path = Path::new("./src/shaders.metal");
            println!("cargo:rerun-if-changed={}", shader_path.display());
            compile_metal_library(shader_path, Some(&header_path), "shaders");
            compile_metal_library(&translate_wgsl_shaders(), None, "wgsl_shaders");
        }
    }

    /// Writes the Metal translation of the primitives drawn from `gpui_wgpu`'s WGSL, and the Rust
    /// constants the renderer binds them with, to `OUT_DIR`.
    fn translate_wgsl_shaders() -> PathBuf {
        use gpui_shader_build::{SHAPES, Target, translate};

        for path in gpui_shader_build::wgsl_source_paths() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let translation = match translate(Target::Metal, &[SHAPES]) {
            Ok(translation) => translation,
            Err(error) => {
                println!("cargo::error=WGSL shader translation failed:\n{error}");
                std::process::exit(1);
            }
        };
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let source_path = out_dir.join("wgsl_shaders.metal");
        std::fs::write(&source_path, &translation.source).unwrap();
        std::fs::write(
            out_dir.join("wgsl_shaders.rs"),
            translation.rust_constants(),
        )
        .unwrap();
        source_path
    }

    fn generate_shader_bindings() -> PathBuf {
        let output_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("scene.h");

        let gpui_dir = find_gpui_crate_dir();

        let mut config = Config {
            include_guard: Some("SCENE_H".into()),
            language: cbindgen::Language::C,
            no_includes: true,
            ..Default::default()
        };
        config.export.include.extend([
            "Bounds".into(),
            "Corners".into(),
            "Edges".into(),
            "Size".into(),
            "Pixels".into(),
            "PointF".into(),
            "Hsla".into(),
            "ContentMask".into(),
            "ContentFade".into(),
            "Uniforms".into(),
            "AtlasTile".into(),
            "PathRasterizationInputIndex".into(),
            "PathVertex_ScaledPixels".into(),
            "PathRasterizationVertex".into(),
            "ShadowInputIndex".into(),
            "Shadow".into(),
            "QuadInputIndex".into(),
            "Underline".into(),
            "UnderlineInputIndex".into(),
            "Quad".into(),
            "BorderStyle".into(),
            "SpriteInputIndex".into(),
            "MonochromeSprite".into(),
            "PolychromeSprite".into(),
            "PathSprite".into(),
            "SurfaceInputIndex".into(),
            "SurfaceBounds".into(),
            "TransformationMatrix".into(),
        ]);
        config.no_includes = true;
        config.enumeration.prefix_with_name = true;

        let mut builder = cbindgen::Builder::new();

        let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

        // Source files from gpui that define types used in shaders
        let gpui_src_paths = [
            gpui_dir.join("src/scene.rs"),
            gpui_dir.join("src/geometry.rs"),
            gpui_dir.join("src/color.rs"),
            gpui_dir.join("src/window.rs"),
            gpui_dir.join("src/platform.rs"),
        ];

        // Source files from this crate
        let local_src_paths = [crate_dir.join("src/metal_renderer.rs")];

        for src_path in gpui_src_paths.iter().chain(local_src_paths.iter()) {
            println!("cargo:rerun-if-changed={}", src_path.display());
            builder = builder.with_src(src_path);
        }

        builder
            .with_config(config)
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(&output_path);

        output_path
    }

    /// Locate the gpui crate directory relative to this crate.
    fn find_gpui_crate_dir() -> PathBuf {
        gpui::GPUI_MANIFEST_DIR.into()
    }

    /// To enable runtime compilation, we need to "stitch" the shaders file with the generated header
    /// so that it is self-contained.
    #[cfg(feature = "runtime_shaders")]
    fn emit_stitched_shaders(header_path: &Path) {
        fn stitch_header(header: &Path, shader_path: &Path) -> std::io::Result<PathBuf> {
            let header_contents = std::fs::read_to_string(header)?;
            let shader_contents = std::fs::read_to_string(shader_path)?;
            let stitched_contents = format!("{header_contents}\n{shader_contents}");
            let out_path =
                PathBuf::from(env::var("OUT_DIR").unwrap()).join("stitched_shaders.metal");
            std::fs::write(&out_path, stitched_contents)?;
            Ok(out_path)
        }
        let shader_source_path = "./src/shaders.metal";
        let shader_path = PathBuf::from(shader_source_path);
        stitch_header(header_path, &shader_path).unwrap();
        println!("cargo:rerun-if-changed={shader_source_path}");
    }

    /// Compiles `shader_path` into `OUT_DIR/{library_name}.metallib`.
    #[cfg(not(feature = "runtime_shaders"))]
    fn compile_metal_library(shader_path: &Path, header_path: Option<&Path>, library_name: &str) {
        use std::process::{self, Command};
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let air_output_path = out_dir.join(format!("{library_name}.air"));
        let metallib_output_path = out_dir.join(format!("{library_name}.metallib"));

        let mut command = Command::new("xcrun");
        command
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-gline-tables-only",
                "-mmacosx-version-min=10.15.7",
                "-MO",
                "-c",
            ])
            .arg(shader_path);
        if let Some(header_path) = header_path {
            command.arg("-include").arg(header_path);
        }
        let output = command.arg("-o").arg(&air_output_path).output().unwrap();

        if !output.status.success() {
            println!(
                "cargo::error=metal shader compilation failed for {}:\n{}",
                shader_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
            process::exit(1);
        }

        let output = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib"])
            .arg(air_output_path)
            .arg("-o")
            .arg(metallib_output_path)
            .output()
            .unwrap();

        if !output.status.success() {
            println!(
                "cargo::error=metallib compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            process::exit(1);
        }
    }
}
