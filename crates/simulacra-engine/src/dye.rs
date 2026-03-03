//! Passive scalar dye field advected by LBM velocity on the GPU.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};

const MAX_INJECTIONS: usize = 64;

/// GPU-side dye advection parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DyeParams {
    pub width: u32,
    pub height: u32,
    pub decay: f32,
    pub diffusion: f32,
    pub vel_scale: f32,
    pub num_injections: u32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// A single dye injection point (uploaded to GPU each frame).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DyeInjectPoint {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub radius: f32,
    pub strength: f32,
    pub _pad: f32,
}

/// Manages GPU buffers and compute pipeline for dye advection.
pub struct DyeField {
    pub params: DyeParams,
    params_buffer: wgpu::Buffer,
    dye_buffers: [wgpu::Buffer; 2],
    inject_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_groups: [wgpu::BindGroup; 2],
    current: usize,
}

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

impl DyeField {
    /// Create a new dye field. `lbm_output` is the LBM output buffer ([rho,ux,uy,curl] per cell).
    pub fn new(gpu: &GpuContext, width: u32, height: u32, lbm_output: &wgpu::Buffer) -> Self {
        let num_cells = (width * height) as usize;

        let params = DyeParams {
            width,
            height,
            decay: 0.002,
            diffusion: 0.01,
            vel_scale: 20.0, // STEPS_PER_FRAME
            num_injections: 0,
            _pad0: 0.0,
            _pad1: 0.0,
        };

        let params_buffer = gpu.create_buffer_init(
            "dye_params",
            &[params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let dye_init = vec![0.0f32; num_cells * 4];
        let dye_buffers = [
            gpu.create_buffer_init(
                "dye_0",
                &dye_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            gpu.create_buffer_init(
                "dye_1",
                &dye_init,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        ];

        let inject_init = vec![DyeInjectPoint {
            x: 0.0, y: 0.0, r: 0.0, g: 0.0, b: 0.0,
            radius: 0.0, strength: 0.0, _pad: 0.0,
        }; MAX_INJECTIONS];
        let inject_buffer = gpu.create_buffer_init(
            "dye_inject",
            &inject_init,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dye_advect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dye_advect.wgsl").into()),
        });

        let uniform = wgpu::BufferBindingType::Uniform;
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };

        let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dye_bgl"),
            entries: &[
                bgl_entry(0, uniform),  // params
                bgl_entry(1, ro),       // lbm_output
                bgl_entry(2, ro),       // dye_in
                bgl_entry(3, rw),       // dye_out
                bgl_entry(4, ro),       // injections
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dye_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dye_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("advect_dye"),
            compilation_options: Default::default(),
            cache: None,
        });

        let make_bg = |label, dye_in: &wgpu::Buffer, dye_out: &wgpu::Buffer| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[
                    bg_entry(0, &params_buffer),
                    bg_entry(1, lbm_output),
                    bg_entry(2, dye_in),
                    bg_entry(3, dye_out),
                    bg_entry(4, &inject_buffer),
                ],
            })
        };

        let bind_groups = [
            make_bg("dye_bg_0to1", &dye_buffers[0], &dye_buffers[1]),
            make_bg("dye_bg_1to0", &dye_buffers[1], &dye_buffers[0]),
        ];

        Self {
            params,
            params_buffer,
            dye_buffers,
            inject_buffer,
            pipeline,
            bind_groups,
            current: 0,
        }
    }

    /// Run one dye advection step on the GPU.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let wg_x = (self.params.width + 15) / 16;
        let wg_y = (self.params.height + 15) / 16;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dye_advect"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[self.current], &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        self.current = 1 - self.current;
    }

    /// Upload injection points for this frame.
    pub fn upload_injections(&mut self, queue: &wgpu::Queue, points: &[DyeInjectPoint]) {
        let n = points.len().min(MAX_INJECTIONS);
        self.params.num_injections = n as u32;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        if n > 0 {
            queue.write_buffer(&self.inject_buffer, 0, bytemuck::cast_slice(&points[..n]));
        }
    }

    /// Reset dye field to zero.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        let num_cells = (self.params.width * self.params.height) as usize;
        let zeros = vec![0.0f32; num_cells * 4];
        queue.write_buffer(&self.dye_buffers[0], 0, bytemuck::cast_slice(&zeros));
        queue.write_buffer(&self.dye_buffers[1], 0, bytemuck::cast_slice(&zeros));
        self.current = 0;
    }

    /// Get the current dye output buffer (for binding in the render shader).
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        // After step(), current points to the INPUT for the next step,
        // so the OUTPUT we just wrote is 1 - current.
        &self.dye_buffers[1 - self.current]
    }

    /// Index of the current output buffer (0 or 1). Use with pre-created bind groups.
    pub fn output_index(&self) -> usize {
        1 - self.current
    }

    /// Access a dye buffer by index (0 or 1). For creating external bind groups.
    pub fn dye_buffer(&self, index: usize) -> &wgpu::Buffer {
        &self.dye_buffers[index]
    }
}
