//! 2D Lattice Boltzmann Method (D2Q9) on GPU via wgpu compute shaders.
//!
//! Implements the BGK (single relaxation time) collision operator with
//! streaming on a D2Q9 lattice. Boundary conditions: bounce-back on walls,
//! moving lid on top (lid-driven cavity).

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

/// D2Q9 lattice velocities (ex, ey) for each direction
pub const EX: [i32; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
pub const EY: [i32; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];

/// Opposite direction indices for bounce-back
pub const OPP: [u32; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Simulation parameters passed to the GPU as a uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LbmParams {
    pub width: u32,
    pub height: u32,
    /// Relaxation parameter: omega = 1/tau. Controls viscosity.
    /// viscosity = (1/omega - 0.5) / 3.0 in lattice units.
    pub omega: f32,
    /// Lid velocity (lattice units, typically 0.05-0.1)
    pub lid_velocity: f32,
}

/// The 2D LBM simulation state living on the GPU.
pub struct Lbm2D {
    pub params: LbmParams,
    params_buffer: wgpu::Buffer,
    /// Distribution functions: 9 floats per cell, double-buffered.
    /// Layout: f[direction * width * height + y * width + x]
    f_buffers: [wgpu::Buffer; 2],
    /// Output buffer: density and velocity for visualization.
    /// Layout: [rho, ux, uy, curl] per cell (4 floats)
    pub output_buffer: wgpu::Buffer,
    /// Which f_buffer is current (0 or 1)
    current: usize,
    collide_stream_pipeline: wgpu::ComputePipeline,
    output_pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2], // one per ping-pong direction
    output_bind_groups: [wgpu::BindGroup; 2],
}

impl Lbm2D {
    pub fn new(gpu: &GpuContext, width: u32, height: u32) -> Self {
        let omega = 1.4; // tau = 1/1.4 ≈ 0.714, moderate viscosity
        let lid_velocity = 0.08;
        let params = LbmParams {
            width,
            height,
            omega,
            lid_velocity,
        };

        let num_cells = (width * height) as usize;
        let output_size = (num_cells * 4 * std::mem::size_of::<f32>()) as u64;

        // Initialize distributions to equilibrium at rest (rho=1, u=0)
        let mut f_init = vec![0.0f32; num_cells * 9];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                for q in 0..9 {
                    f_init[q * num_cells + idx] = W[q];
                }
            }
        }

        let f_buffers = [
            gpu.create_buffer_init(
                "f0",
                &f_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            gpu.create_buffer_init(
                "f1",
                &f_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        ];

        let params_buffer = gpu.create_buffer_init(
            "lbm_params",
            &[params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let output_buffer = gpu.create_buffer(
            "lbm_output",
            output_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Shader
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("lbm_d2q9"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/lbm_d2q9.wgsl").into(),
                ),
            });

        // Bind group layout for collide+stream: params (uniform), f_in (read), f_out (write)
        let cs_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("cs_bgl"),
                    entries: &[
                        // params
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // f_in (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // f_out (read-write storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Bind group layout for output: params, f_in, output
        let out_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("out_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let cs_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cs_pl"),
                    bind_group_layouts: &[&cs_bgl],
                    push_constant_ranges: &[],
                });

        let out_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("out_pl"),
                    bind_group_layouts: &[&out_bgl],
                    push_constant_ranges: &[],
                });

        let collide_stream_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("collide_stream"),
                    layout: Some(&cs_pipeline_layout),
                    module: &shader,
                    entry_point: Some("collide_stream"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let output_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute_output"),
                    layout: Some(&out_pipeline_layout),
                    module: &shader,
                    entry_point: Some("compute_output"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Create bind groups for both ping-pong directions
        let bind_groups = [
            // 0→1: read from f0, write to f1
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cs_bg_0to1"),
                layout: &cs_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f_buffers[1].as_entire_binding(),
                    },
                ],
            }),
            // 1→0: read from f1, write to f0
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cs_bg_1to0"),
                layout: &cs_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f_buffers[0].as_entire_binding(),
                    },
                ],
            }),
        ];

        let output_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("out_bg_0"),
                layout: &out_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("out_bg_1"),
                layout: &out_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            }),
        ];

        Self {
            params,
            params_buffer,
            f_buffers,
            output_buffer,
            current: 0,
            collide_stream_pipeline,
            output_pipeline,
            bind_groups,
            output_bind_groups,
        }
    }

    /// Run N simulation steps, then compute output fields.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder, n_steps: u32) {
        let wg_x = (self.params.width + 15) / 16;
        let wg_y = (self.params.height + 15) / 16;

        for _ in 0..n_steps {
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

        // Compute macroscopic output from current distributions
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lbm_output"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.output_pipeline);
        pass.set_bind_group(0, &self.output_bind_groups[self.current], &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
