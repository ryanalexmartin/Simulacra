use simulacra_engine::gpu::GpuContext;
use simulacra_engine::lbm::{Lbm2D, CELL_FLUID, CELL_INLET, CELL_OUTLET, CELL_SOLID};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const SIM_WIDTH: u32 = 256;
const SIM_HEIGHT: u32 = 256;
const STEPS_PER_FRAME: u32 = 20;
const BRUSH_RADIUS: i32 = 3;
const INLET_VELOCITY: f32 = 0.08;

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Solid,
    Inlet,
    Outlet,
}

struct App {
    state: Option<AppState>,
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    lbm: Lbm2D,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    frame_count: u64,
    last_fps_time: std::time::Instant,
    fps_frame_count: u64,
    // Interaction state
    cell_types: Vec<u32>,
    cell_props: Vec<f32>,
    mouse_pos: (f64, f64),
    left_pressed: bool,
    right_pressed: bool,
    cell_types_dirty: bool,
    cell_props_dirty: bool,
    // Controls
    current_tool: Tool,
    gravity_on: bool,
    paused: bool,
}

impl App {
    fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Simulacra — LBM Fluid")
                        .with_inner_size(winit::dpi::LogicalSize::new(768, 768)),
                )
                .expect("Failed to create window"),
        );

        let state = pollster::block_on(init_state(window));
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    state.surface_config.width = new_size.width;
                    state.surface_config.height = new_size.height;
                    state
                        .surface
                        .configure(&state.gpu.device, &state.surface_config);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_pos = (position.x, position.y);
                if state.left_pressed || state.right_pressed {
                    let drawing = state.left_pressed;
                    paint_cells(state, drawing);
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => match button {
                MouseButton::Left => {
                    state.left_pressed = btn_state == ElementState::Pressed;
                    if state.left_pressed {
                        paint_cells(state, true);
                    }
                }
                MouseButton::Right => {
                    state.right_pressed = btn_state == ElementState::Pressed;
                    if state.right_pressed {
                        paint_cells(state, false);
                    }
                }
                _ => {}
            },
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        handle_key(state, key);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // Upload cell types/props if changed
                if state.cell_types_dirty {
                    state
                        .lbm
                        .upload_cell_types(&state.gpu.queue, &state.cell_types);
                    state.cell_types_dirty = false;
                }
                if state.cell_props_dirty {
                    state
                        .lbm
                        .upload_cell_props(&state.gpu.queue, &state.cell_props);
                    state.cell_props_dirty = false;
                }

                // Upload params every frame (supports runtime changes)
                state.lbm.update_params(&state.gpu.queue);

                let mut encoder =
                    state
                        .gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("frame_encoder"),
                        });

                if !state.paused {
                    state.lbm.step(&mut encoder, STEPS_PER_FRAME);
                }

                let frame = state
                    .surface
                    .get_current_texture()
                    .expect("Failed to get surface texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });

                    rpass.set_pipeline(&state.render_pipeline);
                    rpass.set_bind_group(0, &state.render_bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }

                state.gpu.queue.submit(Some(encoder.finish()));
                frame.present();

                // FPS counter + status
                state.frame_count += 1;
                state.fps_frame_count += 1;
                let elapsed = state.last_fps_time.elapsed();
                if elapsed.as_secs_f64() >= 1.0 {
                    let fps = state.fps_frame_count as f64 / elapsed.as_secs_f64();
                    let sim_steps_per_sec = fps * STEPS_PER_FRAME as f64;
                    let nu = (1.0 / state.lbm.params.omega as f64 - 0.5) / 3.0;
                    let re = INLET_VELOCITY as f64 * SIM_WIDTH as f64 / nu;
                    let tool_name = match state.current_tool {
                        Tool::Solid => "solid",
                        Tool::Inlet => "inlet",
                        Tool::Outlet => "outlet",
                    };
                    let grav = if state.gravity_on { "ON" } else { "OFF" };
                    let pause = if state.paused { " [PAUSED]" } else { "" };
                    state.window.set_title(&format!(
                        "Simulacra — LBM {}x{} | {:.0} fps | {:.0} steps/s | \u{03C9}={:.2} Re~{:.0} | tool:{} grav:{}{} | 1/2/3:tool G:grav \u{2191}\u{2193}:visc R:reset Space:pause",
                        SIM_WIDTH, SIM_HEIGHT, fps, sim_steps_per_sec,
                        state.lbm.params.omega, re, tool_name, grav, pause
                    ));
                    state.last_fps_time = std::time::Instant::now();
                    state.fps_frame_count = 0;
                }

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn handle_key(state: &mut AppState, key: KeyCode) {
    match key {
        KeyCode::KeyG => {
            state.gravity_on = !state.gravity_on;
            if state.gravity_on {
                state.lbm.params.gravity_x = 0.0;
                state.lbm.params.gravity_y = -0.0001;
            } else {
                state.lbm.params.gravity_x = 0.0;
                state.lbm.params.gravity_y = 0.0;
            }
        }
        KeyCode::ArrowUp => {
            let omega = (state.lbm.params.omega + 0.02).min(1.98);
            state.lbm.params.omega = omega;
        }
        KeyCode::ArrowDown => {
            let omega = (state.lbm.params.omega - 0.02).max(1.0);
            state.lbm.params.omega = omega;
        }
        KeyCode::Digit1 => {
            state.current_tool = Tool::Solid;
        }
        KeyCode::Digit2 => {
            state.current_tool = Tool::Inlet;
        }
        KeyCode::Digit3 => {
            state.current_tool = Tool::Outlet;
        }
        KeyCode::KeyR => {
            // Reset simulation
            let num_cells = (SIM_WIDTH * SIM_HEIGHT) as usize;
            state.cell_types = vec![CELL_FLUID; num_cells];
            state.cell_props = vec![0.0f32; num_cells * 2];
            state.cell_types_dirty = true;
            state.cell_props_dirty = true;
            state.lbm.reset(&state.gpu.queue);
        }
        KeyCode::Space => {
            state.paused = !state.paused;
        }
        _ => {}
    }
}

/// Convert mouse position to simulation grid coordinates and paint a brush.
fn paint_cells(state: &mut AppState, draw: bool) {
    let win_size = state.window.inner_size();
    let sx = state.mouse_pos.0 / win_size.width as f64 * SIM_WIDTH as f64;
    let sy = state.mouse_pos.1 / win_size.height as f64 * SIM_HEIGHT as f64;
    let cx = sx as i32;
    let cy = sy as i32;

    for dy in -BRUSH_RADIUS..=BRUSH_RADIUS {
        for dx in -BRUSH_RADIUS..=BRUSH_RADIUS {
            if dx * dx + dy * dy > BRUSH_RADIUS * BRUSH_RADIUS {
                continue; // circular brush
            }
            let px = cx + dx;
            let py = cy + dy;
            if px >= 0 && px < SIM_WIDTH as i32 && py >= 0 && py < SIM_HEIGHT as i32 {
                let idx = py as usize * SIM_WIDTH as usize + px as usize;
                if draw {
                    match state.current_tool {
                        Tool::Solid => {
                            state.cell_types[idx] = CELL_SOLID;
                            state.cell_props[idx * 2] = 0.0;
                            state.cell_props[idx * 2 + 1] = 0.0;
                        }
                        Tool::Inlet => {
                            state.cell_types[idx] = CELL_INLET;
                            state.cell_props[idx * 2] = INLET_VELOCITY;
                            state.cell_props[idx * 2 + 1] = 0.0;
                        }
                        Tool::Outlet => {
                            state.cell_types[idx] = CELL_OUTLET;
                            state.cell_props[idx * 2] = 0.0;
                            state.cell_props[idx * 2 + 1] = 0.0;
                        }
                    }
                } else {
                    // Erase: set to fluid
                    state.cell_types[idx] = CELL_FLUID;
                    state.cell_props[idx * 2] = 0.0;
                    state.cell_props[idx * 2 + 1] = 0.0;
                }
            }
        }
    }
    state.cell_types_dirty = true;
    state.cell_props_dirty = true;
}

async fn init_state(window: Arc<Window>) -> AppState {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let surface = instance
        .create_surface(window.clone())
        .expect("Failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a suitable GPU adapter");

    let adapter_info = adapter.get_info();
    println!("GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("simulacra-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device");

    let size = window.inner_size();
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(surface_caps.formats[0]);

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &surface_config);

    let gpu = GpuContext {
        instance,
        device,
        queue,
        adapter_info,
    };

    let lbm = Lbm2D::new(&gpu, SIM_WIDTH, SIM_HEIGHT);

    // Render pipeline setup
    let render_shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
        });

    let render_params: [u32; 4] = [SIM_WIDTH, SIM_HEIGHT, 0, 0];
    let render_params_buffer = gpu.create_buffer_init(
        "render_params",
        &render_params,
        wgpu::BufferUsages::UNIFORM,
    );

    let render_bgl =
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    let render_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg"),
        layout: &render_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: render_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lbm.output_buffer.as_entire_binding(),
            },
        ],
    });

    let render_pipeline_layout =
        gpu.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pl"),
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            });

    let render_pipeline =
        gpu.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render_pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    let num_cells = (SIM_WIDTH * SIM_HEIGHT) as usize;
    let cell_types = vec![CELL_FLUID; num_cells];
    let cell_props = vec![0.0f32; num_cells * 2];

    AppState {
        window,
        surface,
        surface_config,
        gpu,
        lbm,
        render_pipeline,
        render_bind_group,
        frame_count: 0,
        last_fps_time: std::time::Instant::now(),
        fps_frame_count: 0,
        cell_types,
        cell_props,
        mouse_pos: (0.0, 0.0),
        left_pressed: false,
        right_pressed: false,
        cell_types_dirty: false,
        cell_props_dirty: false,
        current_tool: Tool::Solid,
        gravity_on: true,
        paused: false,
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
