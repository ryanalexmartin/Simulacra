//! 2D Lattice Boltzmann Method (D2Q9) on GPU via wgpu compute shaders.
//!
//! Implements the BGK (single relaxation time) collision operator with
//! streaming on a D2Q9 lattice. Supports interactive solid obstacles
//! via a cell_type buffer that can be updated at runtime.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};

/// D2Q9 lattice weights
pub const W: [f32; 9] = [
    4.0 / 9.0,  // center
    1.0 / 9.0,  // east
    1.0 / 9.0,  // north
    1.0 / 9.0,  // west
    1.0 / 9.0,  // south
    1.0 / 36.0, // north-east
    1.0 / 36.0, // north-west
    1.0 / 36.0, // south-west
    1.0 / 36.0, // south-east
];

/// Cell types
pub const CELL_FLUID: u32 = 0;
pub const CELL_SOLID: u32 = 1;
pub const CELL_INLET: u32 = 2;
pub const CELL_OUTLET: u32 = 3;

/// Simulation parameters passed to the GPU as a uniform buffer.
/// Padded to 32 bytes (2x vec4) for uniform buffer alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LbmParams {
    pub width: u32,
    pub height: u32,
    /// Relaxation parameter: omega = 1/tau. Controls viscosity.
    pub omega: f32,
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// The 2D LBM simulation state living on the GPU.
pub struct Lbm2D {
    pub params: LbmParams,
    params_buffer: wgpu::Buffer,
    /// Distribution functions: double-buffered. Layout: f[q * W * H + y * W + x]
    f_buffers: [wgpu::Buffer; 2],
    /// Cell type per cell: 0=fluid, 1=solid, 2=inlet, 3=outlet. CPU-writable.
    pub cell_type_buffer: wgpu::Buffer,
    /// Per-cell properties: [ux, uy] per cell (inlet velocity, etc). CPU-writable.
    pub cell_props_buffer: wgpu::Buffer,
    /// Output: [rho, ux, uy, curl] per cell
    pub output_buffer: wgpu::Buffer,
    current: usize,
    collide_stream_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    output_bind_groups: [wgpu::BindGroup; 2],
}

/// Helper to create a bind group layout entry for a storage or uniform buffer.
fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl Lbm2D {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let omega = 1.85;
        let params = LbmParams {
            width,
            height,
            omega,
            gravity_x: 0.0,
            gravity_y: -0.0001,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        let num_cells = (width * height) as usize;
        let output_size = (num_cells * 4 * std::mem::size_of::<f32>()) as u64;

        // Initialize distributions to equilibrium at rest (rho=1, u=0)
        let mut f_init = vec![0.0f32; num_cells * 9];
        for cell in 0..num_cells {
            for q in 0..9 {
                f_init[q * num_cells + cell] = W[q];
            }
        }

        let f_buffers = [
            gpu.create_buffer_init(
                "f0",
                &f_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            gpu.create_buffer_init(
                "f1",
                &f_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        ];

        let params_buffer = gpu.create_buffer_init(
            "lbm_params",
            &[params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Cell types: all fluid initially
        let cell_types = vec![CELL_FLUID; num_cells];
        let cell_type_buffer = gpu.create_buffer_init(
            "cell_type",
            &cell_types,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // Per-cell properties: 2 floats (ux, uy) per cell
        let cell_props = vec![0.0f32; num_cells * 2];
        let cell_props_buffer = gpu.create_buffer_init(
            "cell_props",
            &cell_props,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer = gpu.create_buffer(
            "lbm_output",
            output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lbm_d2q9"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/lbm_d2q9.wgsl").into(),
                ),
            });

        // Bind group layout: params, f_in, f_out, cell_type
        let uniform = wgpu::BufferBindingType::Uniform;
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };

        let cs_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("cs_bgl"),
                    entries: &[
                        bgl_entry(0, uniform),
                        bgl_entry(1, ro), // f_in
                        bgl_entry(2, rw), // f_out
                        bgl_entry(3, ro), // cell_type
                        bgl_entry(4, ro), // cell_props
                    ],
                });

        // Output layout: params, f_in, output, cell_type, cell_props
        let out_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("out_bgl"),
                    entries: &[
                        bgl_entry(0, uniform),
                        bgl_entry(1, ro), // f_in
                        bgl_entry(2, rw), // output
                        bgl_entry(3, ro), // cell_type
                        bgl_entry(4, ro), // cell_props
                    ],
                });

        let make_pipeline = |layout: &wgpu::BindGroupLayout, entry: &str, label: &str| {
            let pl = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let collide_stream_pipeline = make_pipeline(&cs_bgl, "collide_stream", "cs_pipeline");
        let output_pipeline = make_pipeline(&out_bgl, "compute_output", "out_pipeline");

        let make_bg = |label, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
        };

        let bind_groups = [
            make_bg(
                "cs_0to1",
                &cs_bgl,
                &[
                    bg_entry(0, &params_buffer),
                    bg_entry(1, &f_buffers[0]),
                    bg_entry(2, &f_buffers[1]),
                    bg_entry(3, &cell_type_buffer),
                    bg_entry(4, &cell_props_buffer),
                ],
            ),
            make_bg(
                "cs_1to0",
                &cs_bgl,
                &[
                    bg_entry(0, &params_buffer),
                    bg_entry(1, &f_buffers[1]),
                    bg_entry(2, &f_buffers[0]),
                    bg_entry(3, &cell_type_buffer),
                    bg_entry(4, &cell_props_buffer),
                ],
            ),
        ];

        let output_bind_groups = [
            make_bg(
                "out_0",
                &out_bgl,
                &[
                    bg_entry(0, &params_buffer),
                    bg_entry(1, &f_buffers[0]),
                    bg_entry(2, &output_buffer),
                    bg_entry(3, &cell_type_buffer),
                    bg_entry(4, &cell_props_buffer),
                ],
            ),
            make_bg(
                "out_1",
                &out_bgl,
                &[
                    bg_entry(0, &params_buffer),
                    bg_entry(1, &f_buffers[1]),
                    bg_entry(2, &output_buffer),
                    bg_entry(3, &cell_type_buffer),
                    bg_entry(4, &cell_props_buffer),
                ],
            ),
        ];

        Self {
            params,
            params_buffer,
            f_buffers,
            cell_type_buffer,
            cell_props_buffer,
            output_buffer,
            current: 0,
            collide_stream_pipeline,
            output_pipeline,
            bind_groups,
            output_bind_groups,
        }
    }

    /// Upload cell types from CPU to GPU.
    pub fn upload_cell_types(&self, queue: &wgpu::Queue, data: &[u32]) {
        queue.write_buffer(&self.cell_type_buffer, 0, bytemuck::cast_slice(data));
    }

    /// Upload per-cell properties (2 floats per cell: ux, uy) from CPU to GPU.
    pub fn upload_cell_props(&self, queue: &wgpu::Queue, data: &[f32]) {
        queue.write_buffer(&self.cell_props_buffer, 0, bytemuck::cast_slice(data));
    }

    /// Upload current params to the GPU uniform buffer (call after modifying self.params).
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
    }

    /// Reset distributions to equilibrium at rest. Call after clearing cell types.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        let num_cells = (self.params.width * self.params.height) as usize;
        let mut f_init = vec![0.0f32; num_cells * 9];
        for q in 0..9 {
            for cell in 0..num_cells {
                f_init[q * num_cells + cell] = W[q];
            }
        }
        queue.write_buffer(&self.f_buffers[0], 0, bytemuck::cast_slice(&f_init));
        queue.write_buffer(&self.f_buffers[1], 0, bytemuck::cast_slice(&f_init));
        self.current = 0;
    }

    /// Run N simulation steps, then compute output fields.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, n_steps: u32) {
        for _ in 0..n_steps {
            self.step_one(encoder);
        }
        self.compute_output(encoder);
    }

    /// Run a single collide-stream step (no output computation).
    pub fn step_one(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (self.params.width + 15) / 16;
        let wg_y = (self.params.height + 15) / 16;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lbm_step"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.collide_stream_pipeline);
        pass.set_bind_group(0, &self.bind_groups[self.current], &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        self.current = 1 - self.current;
    }

    /// Compute output fields (rho, ux, uy, curl) from current distribution.
    pub fn compute_output(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (self.params.width + 15) / 16;
        let wg_y = (self.params.height + 15) / 16;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lbm_output"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.output_pipeline);
        pass.set_bind_group(0, &self.output_bind_groups[self.current], &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
