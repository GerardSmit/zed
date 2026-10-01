//! Translates GPUI's WGSL shaders into Metal Shading Language and HLSL while a native renderer
//! builds, so the native renderers draw from the same WGSL that `gpui_wgpu` runs.
//!
//! [`metal_slots`] and [`direct3d_registers`] are the only definition of where the native
//! renderers bind each WGSL resource: the translators bind to them, and
//! [`Translation::rust_constants`] hands them to the renderer through a generated file.

use std::{fmt, path::PathBuf};

use naga::{
    AddressSpace, Module, ResourceBinding, ShaderStage, TypeInner,
    back::{hlsl, msl},
    compact::KeepUnused,
    proc::{BoundsCheckPolicies, BoundsCheckPolicy, Layouter},
    valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
};

/// `gpui_wgpu`'s storage-buffer shader variant: the same files, concatenated in the same order.
pub const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("../../gpui_wgpu/src/shaders.wgsl"),
    include_str!("../../gpui_wgpu/src/shaders_storage.wgsl"),
);

/// The files behind [`STORAGE_BUFFER_SHADERS`], for `cargo:rerun-if-changed`.
pub fn wgsl_source_paths() -> [PathBuf; 2] {
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../gpui_wgpu/src");
    [
        source_dir.join("shaders.wgsl"),
        source_dir.join("shaders_storage.wgsl"),
    ]
}

/// WGSL's `@group(0) @binding(0) var<uniform> globals: GlobalParams`.
const GLOBALS: ResourceBinding = ResourceBinding {
    group: 0,
    binding: 0,
};

/// WGSL's `@group(1) @binding(0)`, which holds every primitive's instance array.
const INSTANCES: ResourceBinding = ResourceBinding {
    group: 1,
    binding: 0,
};

/// Metal buffer indices, the same in the vertex and the fragment stage.
pub mod metal_slots {
    pub const GLOBALS: u8 = 0;
    pub const INSTANCES: u8 = 1;
    /// naga reads the byte length of every runtime-sized array an entry point uses from this
    /// buffer, whatever the bounds-check policy.
    pub const BUFFER_SIZES: u8 = 2;
}

/// Direct3D 11 registers, clear of the hand-written shaders' `b0`, `b1` and `t0`.
pub mod direct3d_registers {
    /// `t1`, where the renderer binds every primitive's instance buffer.
    pub const INSTANCES: u32 = 1;
    /// `b2`.
    pub const GLOBALS: u32 = 2;
    /// `b3`, naga's `NagaConstants`. Direct3D starts `SV_InstanceID` at zero for every draw, so
    /// WGSL's `instance_index` adds the `first_instance` it finds here.
    pub const SPECIAL_CONSTANTS: u32 = 3;
}

/// A primitive drawn as instanced quads, whose vertex and fragment entry points both read the
/// primitive's instances from `@group(1) @binding(0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitive {
    /// The prefix of the primitive's generated Rust constants.
    pub name: &'static str,
    pub vertex_entry_point: &'static str,
    pub fragment_entry_point: &'static str,
}

pub const SHAPES: Primitive = Primitive {
    name: "SHAPES",
    vertex_entry_point: "vs_shape",
    fragment_entry_point: "fs_shape",
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Metal Shading Language 2.2, the newest version macOS 10.15 runs.
    Metal,
    /// HLSL for shader model 5.0, which every Direct3D 11 device of feature level 11_0 runs.
    Direct3D11,
}

/// One translated source holding the entry points of a set of primitives.
#[derive(Debug)]
pub struct Translation {
    pub target: Target,
    pub source: String,
    pub primitives: Vec<TranslatedPrimitive>,
    /// The size of WGSL's `GlobalParams`, which the renderer binds at the globals slot.
    pub globals_size: u32,
}

#[derive(Debug)]
pub struct TranslatedPrimitive {
    pub primitive: Primitive,
    /// The vertex entry point as the translated source names it.
    pub vertex_entry_point: String,
    /// The fragment entry point as the translated source names it.
    pub fragment_entry_point: String,
    /// The byte stride of the primitive's WGSL instance array.
    pub instance_stride: u32,
}

#[derive(Debug)]
pub struct TranslationError(String);

impl fmt::Display for TranslationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TranslationError {}

/// Translates the entry points of `primitives` in [`STORAGE_BUFFER_SHADERS`] into one source for
/// `target`.
pub fn translate(
    target: Target,
    primitives: &[Primitive],
) -> Result<Translation, TranslationError> {
    let mut module = naga::front::wgsl::parse_str(STORAGE_BUFFER_SHADERS)
        .map_err(|error| TranslationError(error.emit_to_string(STORAGE_BUFFER_SHADERS)))?;
    validate(&module)?;

    let requested: Vec<(&str, ShaderStage)> = primitives
        .iter()
        .flat_map(|primitive| {
            [
                (primitive.vertex_entry_point, ShaderStage::Vertex),
                (primitive.fragment_entry_point, ShaderStage::Fragment),
            ]
        })
        .collect();
    for (name, stage) in &requested {
        let found = module
            .entry_points
            .iter()
            .any(|entry_point| entry_point.name == *name && entry_point.stage == *stage);
        if !found {
            return Err(TranslationError(format!(
                "the WGSL has no {stage:?} entry point named `{name}`"
            )));
        }
    }
    module
        .entry_points
        .retain(|entry_point| requested.iter().any(|(name, _)| entry_point.name == *name));
    // The other primitives' instance arrays share `@group(1) @binding(0)`, so everything the
    // requested entry points do not reach has to leave the module before it is written.
    naga::compact::compact(&mut module, KeepUnused::No);
    let info = validate(&module)?;

    let (source, entry_point_names) = match target {
        Target::Metal => write_msl(&module, &info)?,
        Target::Direct3D11 => write_hlsl(&module, &info)?,
    };

    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|error| TranslationError(format!("cannot lay out the WGSL types: {error}")))?;
    let globals = module
        .global_variables
        .iter()
        .find(|(_, global)| {
            global.binding == Some(GLOBALS) && global.space == AddressSpace::Uniform
        })
        .ok_or_else(|| {
            TranslationError("the requested entry points do not read the globals".to_string())
        })?;
    let globals_size = layouter[globals.1.ty].size;

    let entry_point_name = |name: &str| -> Result<(usize, String), TranslationError> {
        module
            .entry_points
            .iter()
            .position(|entry_point| entry_point.name == name)
            .and_then(|index| Some((index, entry_point_names.get(index)?.clone())))
            .ok_or_else(|| TranslationError(format!("`{name}` was not translated")))
    };
    let primitives = primitives
        .iter()
        .map(|primitive| {
            let (vertex_index, vertex_entry_point) =
                entry_point_name(primitive.vertex_entry_point)?;
            let (_, fragment_entry_point) = entry_point_name(primitive.fragment_entry_point)?;
            Ok(TranslatedPrimitive {
                primitive: *primitive,
                vertex_entry_point,
                fragment_entry_point,
                instance_stride: instance_stride(&module, &info, vertex_index)?,
            })
        })
        .collect::<Result<Vec<_>, TranslationError>>()?;

    Ok(Translation {
        target,
        source,
        primitives,
        globals_size,
    })
}

impl Translation {
    /// Rust constants for the renderer to `include!`: the binding slots, the entry point names
    /// and the sizes the renderer's own types must match.
    pub fn rust_constants(&self) -> String {
        let mut constants = String::new();
        match self.target {
            Target::Metal => {
                push_constant(
                    &mut constants,
                    "GLOBALS_BUFFER_INDEX",
                    "u64",
                    metal_slots::GLOBALS,
                );
                push_constant(
                    &mut constants,
                    "INSTANCES_BUFFER_INDEX",
                    "u64",
                    metal_slots::INSTANCES,
                );
                push_constant(
                    &mut constants,
                    "BUFFER_SIZES_BUFFER_INDEX",
                    "u64",
                    metal_slots::BUFFER_SIZES,
                );
            }
            Target::Direct3D11 => {
                push_constant(
                    &mut constants,
                    "INSTANCES_REGISTER",
                    "u32",
                    direct3d_registers::INSTANCES,
                );
                push_constant(
                    &mut constants,
                    "GLOBALS_REGISTER",
                    "u32",
                    direct3d_registers::GLOBALS,
                );
                push_constant(
                    &mut constants,
                    "SPECIAL_CONSTANTS_REGISTER",
                    "u32",
                    direct3d_registers::SPECIAL_CONSTANTS,
                );
            }
        }
        push_constant(&mut constants, "GLOBALS_SIZE", "usize", self.globals_size);
        for translated in &self.primitives {
            let name = translated.primitive.name;
            push_constant(
                &mut constants,
                &format!("{name}_VERTEX_ENTRY_POINT"),
                "&str",
                format!("{:?}", translated.vertex_entry_point),
            );
            push_constant(
                &mut constants,
                &format!("{name}_FRAGMENT_ENTRY_POINT"),
                "&str",
                format!("{:?}", translated.fragment_entry_point),
            );
            push_constant(
                &mut constants,
                &format!("{name}_INSTANCE_STRIDE"),
                "usize",
                translated.instance_stride,
            );
        }
        constants
    }
}

fn push_constant(constants: &mut String, name: &str, ty: &str, value: impl fmt::Display) {
    constants.push_str(&format!("pub const {name}: {ty} = {value};\n"));
}

fn validate(module: &Module) -> Result<ModuleInfo, TranslationError> {
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(module)
        .map_err(|error| TranslationError(error.emit_to_string(STORAGE_BUFFER_SHADERS)))
}

fn unchecked_bounds() -> BoundsCheckPolicies {
    BoundsCheckPolicies {
        index: BoundsCheckPolicy::Unchecked,
        buffer: BoundsCheckPolicy::Unchecked,
        image_load: BoundsCheckPolicy::Unchecked,
        binding_array: BoundsCheckPolicy::Unchecked,
    }
}

fn write_msl(
    module: &Module,
    info: &ModuleInfo,
) -> Result<(String, Vec<String>), TranslationError> {
    let buffer = |slot| msl::BindTarget {
        buffer: Some(slot),
        ..msl::BindTarget::default()
    };
    let resources = msl::EntryPointResources {
        resources: msl::BindingMap::from([
            (GLOBALS, buffer(metal_slots::GLOBALS)),
            (INSTANCES, buffer(metal_slots::INSTANCES)),
        ]),
        immediates_buffer: None,
        sizes_buffer: Some(metal_slots::BUFFER_SIZES),
    };
    let options = msl::Options {
        lang_version: (2, 2),
        per_entry_point_map: module
            .entry_points
            .iter()
            .map(|entry_point| (entry_point.name.clone(), resources.clone()))
            .collect(),
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: unchecked_bounds(),
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
    };
    let (source, translation_info) =
        msl::write_string(module, info, &options, &msl::PipelineOptions::default())
            .map_err(|error| TranslationError(format!("cannot write MSL: {error}")))?;
    let entry_point_names = translation_info
        .entry_point_names
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| TranslationError(format!("cannot write an MSL entry point: {error}")))?;
    Ok((source, entry_point_names))
}

fn write_hlsl(
    module: &Module,
    info: &ModuleInfo,
) -> Result<(String, Vec<String>), TranslationError> {
    let register = |register| hlsl::BindTarget {
        register,
        ..hlsl::BindTarget::default()
    };
    let options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_0,
        binding_map: hlsl::BindingMap::from([
            (GLOBALS, register(direct3d_registers::GLOBALS)),
            (INSTANCES, register(direct3d_registers::INSTANCES)),
        ]),
        fake_missing_bindings: false,
        special_constants_binding: Some(register(direct3d_registers::SPECIAL_CONSTANTS)),
        zero_initialize_workgroup_memory: false,
        restrict_indexing: false,
        force_loop_bounding: false,
        ray_query_initialization_tracking: false,
        ..hlsl::Options::default()
    };
    let mut source = String::new();
    let reflection = hlsl::Writer::new(&mut source, &options, &hlsl::PipelineOptions::default())
        .write(module, info, None)
        .map_err(|error| TranslationError(format!("cannot write HLSL: {error}")))?;
    let entry_point_names = reflection
        .entry_point_names
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| TranslationError(format!("cannot write an HLSL entry point: {error}")))?;
    Ok((
        declare_special_constants_for_shader_model_5_0(source)?,
        entry_point_names,
    ))
}

/// naga declares its special constants as a `ConstantBuffer<T>`, which needs shader model 5.1,
/// even when it writes shader model 5.0.
fn declare_special_constants_for_shader_model_5_0(
    source: String,
) -> Result<String, TranslationError> {
    let register = direct3d_registers::SPECIAL_CONSTANTS;
    let declaration =
        format!("ConstantBuffer<NagaConstants> _NagaConstants: register(b{register});");
    if !source.contains(&declaration) {
        return Err(TranslationError(format!(
            "the HLSL does not declare `{declaration}`"
        )));
    }
    Ok(source.replacen(
        &declaration,
        &format!(
            "cbuffer NagaConstantsBuffer : register(b{register}) {{ NagaConstants _NagaConstants; }}"
        ),
        1,
    ))
}

fn instance_stride(
    module: &Module,
    info: &ModuleInfo,
    entry_point_index: usize,
) -> Result<u32, TranslationError> {
    let entry_point = info.get_entry_point(entry_point_index);
    let mut strides = module
        .global_variables
        .iter()
        .filter(|(handle, global)| {
            global.binding == Some(INSTANCES) && !entry_point[*handle].is_empty()
        })
        .map(|(_, global)| match module.types[global.ty].inner {
            TypeInner::Array { stride, .. } => Some(stride),
            _ => None,
        });
    match (strides.next(), strides.next()) {
        (Some(Some(stride)), None) => Ok(stride),
        _ => Err(TranslationError(format!(
            "`{}` must read exactly one instance array",
            module.entry_points[entry_point_index].name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_translate_to_msl() {
        let translation = translate(Target::Metal, &[SHAPES]).unwrap();
        let shapes = &translation.primitives[0];
        assert_eq!(shapes.vertex_entry_point, "vs_shape");
        assert_eq!(shapes.fragment_entry_point, "fs_shape");
        assert!(translation.source.contains("vertex vs_shape"));
        assert!(translation.source.contains("fragment fs_shape"));
        assert_eq!(shapes.instance_stride, 192);
        assert_eq!(translation.globals_size, 16);

        for entry_point in ["vs_shape", "fs_shape"] {
            let signature = entry_point_signature(&translation.source, entry_point);
            assert!(
                signature.contains(&format!("[[buffer({})]]", metal_slots::GLOBALS)),
                "{signature}"
            );
            assert!(
                signature.contains(&format!("[[buffer({})]]", metal_slots::INSTANCES)),
                "{signature}"
            );
            assert!(
                signature.contains(&format!("[[buffer({})]]", metal_slots::BUFFER_SIZES)),
                "{signature}"
            );
        }
        assert!(!translation.source.contains("b_quads"));
    }

    #[test]
    fn shapes_translate_to_hlsl() {
        let translation = translate(Target::Direct3D11, &[SHAPES]).unwrap();
        let shapes = &translation.primitives[0];
        assert_eq!(shapes.vertex_entry_point, "vs_shape");
        assert_eq!(shapes.fragment_entry_point, "fs_shape");
        assert_eq!(shapes.instance_stride, 192);
        assert_eq!(translation.globals_size, 16);

        let source = &translation.source;
        assert!(source.contains(" vs_shape("), "{source}");
        assert!(source.contains(" fs_shape("), "{source}");
        assert!(source.contains(&format!(
            "ByteAddressBuffer b_shapes : register(t{});",
            direct3d_registers::INSTANCES
        )));
        assert!(source.contains(&format!(
            "cbuffer globals : register(b{})",
            direct3d_registers::GLOBALS
        )));
        assert!(source.contains(&format!(
            "cbuffer NagaConstantsBuffer : register(b{}) {{ NagaConstants _NagaConstants; }}",
            direct3d_registers::SPECIAL_CONSTANTS
        )));
        assert!(source.contains("_NagaConstants.first_instance + "));
        assert!(!source.contains("ConstantBuffer<"), "{source}");
        assert!(!source.contains("b_quads"));
    }

    #[test]
    fn rust_constants_carry_the_binding_slots() {
        let metal = translate(Target::Metal, &[SHAPES])
            .unwrap()
            .rust_constants();
        assert!(metal.contains("pub const GLOBALS_BUFFER_INDEX: u64 = 0;\n"));
        assert!(metal.contains("pub const INSTANCES_BUFFER_INDEX: u64 = 1;\n"));
        assert!(metal.contains("pub const BUFFER_SIZES_BUFFER_INDEX: u64 = 2;\n"));
        assert!(metal.contains("pub const GLOBALS_SIZE: usize = 16;\n"));
        assert!(metal.contains("pub const SHAPES_VERTEX_ENTRY_POINT: &str = \"vs_shape\";\n"));
        assert!(metal.contains("pub const SHAPES_FRAGMENT_ENTRY_POINT: &str = \"fs_shape\";\n"));
        assert!(metal.contains("pub const SHAPES_INSTANCE_STRIDE: usize = 192;\n"));

        let direct3d = translate(Target::Direct3D11, &[SHAPES])
            .unwrap()
            .rust_constants();
        assert!(direct3d.contains("pub const INSTANCES_REGISTER: u32 = 1;\n"));
        assert!(direct3d.contains("pub const GLOBALS_REGISTER: u32 = 2;\n"));
        assert!(direct3d.contains("pub const SPECIAL_CONSTANTS_REGISTER: u32 = 3;\n"));
        assert!(direct3d.contains("pub const SHAPES_VERTEX_ENTRY_POINT: &str = \"vs_shape\";\n"));
    }

    #[test]
    fn missing_entry_points_are_reported() {
        let missing = Primitive {
            name: "MISSING",
            vertex_entry_point: "vs_missing",
            fragment_entry_point: "fs_shape",
        };
        let error = translate(Target::Metal, &[missing]).unwrap_err();
        assert!(error.to_string().contains("vs_missing"), "{error}");

        let swapped = Primitive {
            name: "SWAPPED",
            vertex_entry_point: "fs_shape",
            fragment_entry_point: "vs_shape",
        };
        assert!(translate(Target::Metal, &[swapped]).is_err());
    }

    fn entry_point_signature<'a>(source: &'a str, entry_point: &str) -> &'a str {
        let start = source
            .find(&format!(" {entry_point}("))
            .unwrap_or_else(|| panic!("no `{entry_point}` in\n{source}"));
        let end = start
            + source[start..]
                .find(") {")
                .unwrap_or_else(|| panic!("`{entry_point}` has no body"));
        &source[start..end]
    }
}
