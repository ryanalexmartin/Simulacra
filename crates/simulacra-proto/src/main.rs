mod levels;
mod ui;

use levels::{all_levels, load_level, GoalZone, Level};
use simulacra_engine::dye::{DyeField, DyeInjectPoint};
use simulacra_engine::gpu::GpuContext;
use simulacra_engine::lbm::{Lbm2D, CELL_FLUID, CELL_INLET, CELL_OUTLET, CELL_SOLID};
use simulacra_engine::rigidbody::{BallExplosion, BallGpuData, BallRenderParams, BallWorld, MAX_BALLS};
use std::sync::Arc;
use ui::{UiInputs, draw_goal_zones, draw_ui};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

const SIM_WIDTH: u32 = 1024;
const SIM_HEIGHT: u32 = 576;
const DEFAULT_STEPS_PER_FRAME: u32 = 20;
const BRUSH_RADIUS: i32 = 3;
const INLET_VELOCITY: f32 = 0.25;
const BALL_RADIUS: f32 = 6.0;
const NOZZLE_WIDTH: i32 = 11;
const BODY_DEPTH: i32 = 4;
const MAX_EMITTERS: usize = 32;

const EMITTER_COLORS: [[f32; 3]; 6] = [
    [0.90, 0.25, 0.20], // red
    [0.20, 0.50, 0.90], // blue
    [0.25, 0.80, 0.35], // green
    [0.95, 0.75, 0.15], // yellow
    [0.70, 0.30, 0.85], // purple
    [0.95, 0.55, 0.15], // orange
];

#[derive(Clone)]
struct Emitter {
    pos: [f32; 2],
    angle: f32,
    velocity: f32,
    is_outlet: bool,
    color: [f32; 3],
}

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Solid,
    Inlet,
    Outlet,
    Ball,
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
    render_bind_group_1: wgpu::BindGroup,
    render_dye_bind_groups: [wgpu::BindGroup; 2],
    dye_field: DyeField,
    frame_count: u64,
    last_fps_time: std::time::Instant,
    fps_frame_count: u64,
    // Interaction state
    base_cell_types: Vec<u32>,
    base_cell_props: Vec<f32>,
    cell_types: Vec<u32>,
    cell_props: Vec<f32>,
    mouse_pos: (f64, f64),
    last_paint_pos: Option<(i32, i32)>,
    left_pressed: bool,
    right_pressed: bool,
    // Controls
    current_tool: Tool,
    gravity_on: bool,
    paused: bool,
    inlet_velocity: f32,
    sim_speed: u32,
    // Emitter devices
    emitters: Vec<Emitter>,
    emitter_angle: f32,
    // Rigidbody
    ball_world: BallWorld,
    ball_render_buffer: wgpu::Buffer,
    ball_params_buffer: wgpu::Buffer,
    // Fluid readback for two-way coupling (1-frame delay)
    fluid_readback_buffer: wgpu::Buffer,
    fluid_data: Vec<f32>,
    // egui
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // Level system
    levels: Vec<Level>,
    current_level: Option<usize>,
    show_level_selector: bool,
    level_goals: Vec<GoalZone>,
    level_description: String,
    level_complete: bool,
    locked_emitter_count: usize,
    fps_display: f64,
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
                        .with_title("Simulacra \u{2014} LBM Fluid")
                        .with_fullscreen(Some(Fullscreen::Borderless(None))),
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

        // Let egui consume events first
        let egui_resp = state.egui_state.on_window_event(&state.window, &event);
        let egui_consumed = egui_resp.consumed;

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
                if !egui_consumed {
                    // Only drag-paint with Solid tool
                    if state.current_tool == Tool::Solid
                        && (state.left_pressed || state.right_pressed)
                    {
                        let drawing = state.left_pressed;
                        paint_cells(state, drawing);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: btn_state,
                button,
                ..
            } => {
                if egui_consumed {
                    return;
                }
                let win_size = state.window.inner_size();
                let sx = (state.mouse_pos.0 / win_size.width as f64 * SIM_WIDTH as f64) as f32;
                let sy = (state.mouse_pos.1 / win_size.height as f64 * SIM_HEIGHT as f64) as f32;

                match state.current_tool {
                    Tool::Ball => match button {
                        MouseButton::Left => {
                            state.left_pressed = btn_state == ElementState::Pressed;
                            if state.left_pressed {
                                state.ball_world.spawn(sx, sy, BALL_RADIUS);
                            }
                        }
                        MouseButton::Right => {
                            state.right_pressed = btn_state == ElementState::Pressed;
                            if state.right_pressed {
                                state.ball_world.remove_nearest(sx, sy, BALL_RADIUS * 3.0);
                            }
                        }
                        _ => {}
                    },
                    Tool::Inlet | Tool::Outlet => match button {
                        MouseButton::Left => {
                            state.left_pressed = btn_state == ElementState::Pressed;
                            if state.left_pressed && state.emitters.len() < MAX_EMITTERS {
                                let color = EMITTER_COLORS[state.emitters.len() % EMITTER_COLORS.len()];
                                state.emitters.push(Emitter {
                                    pos: [sx, sy],
                                    angle: state.emitter_angle,
                                    velocity: state.inlet_velocity,
                                    is_outlet: state.current_tool == Tool::Outlet,
                                    color,
                                });
                            }
                        }
                        MouseButton::Right => {
                            state.right_pressed = btn_state == ElementState::Pressed;
                            if state.right_pressed {
                                // Remove nearest emitter within threshold (skip locked)
                                let threshold = (NOZZLE_WIDTH as f32) * 2.0;
                                let mut best_idx = None;
                                let mut best_dist = threshold * threshold;
                                for (i, em) in state.emitters.iter().enumerate() {
                                    if i < state.locked_emitter_count {
                                        continue;
                                    }
                                    let dx = em.pos[0] - sx;
                                    let dy = em.pos[1] - sy;
                                    let d2 = dx * dx + dy * dy;
                                    if d2 < best_dist {
                                        best_dist = d2;
                                        best_idx = Some(i);
                                    }
                                }
                                if let Some(idx) = best_idx {
                                    state.emitters.remove(idx);
                                }
                            }
                        }
                        _ => {}
                    },
                    Tool::Solid => match button {
                        MouseButton::Left => {
                            state.left_pressed = btn_state == ElementState::Pressed;
                            if state.left_pressed {
                                state.last_paint_pos = None;
                                paint_cells(state, true);
                            } else {
                                state.last_paint_pos = None;
                            }
                        }
                        MouseButton::Right => {
                            state.right_pressed = btn_state == ElementState::Pressed;
                            if state.right_pressed {
                                state.last_paint_pos = None;
                                paint_cells(state, false);
                            } else {
                                state.last_paint_pos = None;
                            }
                        }
                        _ => {}
                    },
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if egui_consumed {
                    return;
                }
                // Scroll wheel rotates emitter angle (15 degree steps)
                if state.current_tool == Tool::Inlet || state.current_tool == Tool::Outlet {
                    let scroll_y = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y / 30.0,
                    };
                    if scroll_y > 0.0 {
                        state.emitter_angle += std::f32::consts::PI / 12.0;
                    } else if scroll_y < 0.0 {
                        state.emitter_angle -= std::f32::consts::PI / 12.0;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if egui_consumed {
                    return;
                }
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        handle_key(state, key);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                // --- Physics + LBM sub-stepping ---
                // Ball physics and LBM run in lockstep: each sub-step moves
                // the ball, re-rasterizes it, uploads cell_types, then runs
                // exactly 1 LBM step. This prevents teleportation artifacts
                // where the ball jumps many cells between LBM steps.
                let mut explosions: Vec<BallExplosion> = Vec::new();

                if !state.paused {
                    // Apply fluid forces from LAST frame's readback (once per frame)
                    if !state.ball_world.balls.is_empty() {
                        state.ball_world.apply_fluid_forces(
                            &state.fluid_data,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                            state.sim_speed,
                        );
                    }

                    // Upload LBM params once per frame
                    state.lbm.update_params(&state.gpu.queue);

                    // Pre-rasterize emitters (+ preview) — static within frame
                    let mut emitter_cell_types = state.base_cell_types.clone();
                    let mut emitter_cell_props = state.base_cell_props.clone();
                    rasterize_emitters(
                        &state.emitters,
                        &mut emitter_cell_types,
                        &mut emitter_cell_props,
                        SIM_WIDTH,
                        SIM_HEIGHT,
                    );
                    if state.current_tool == Tool::Inlet || state.current_tool == Tool::Outlet {
                        let win_size = state.window.inner_size();
                        let sx = (state.mouse_pos.0 / win_size.width as f64
                            * SIM_WIDTH as f64) as f32;
                        let sy = (state.mouse_pos.1 / win_size.height as f64
                            * SIM_HEIGHT as f64) as f32;
                        let preview = Emitter {
                            pos: [sx, sy],
                            angle: state.emitter_angle,
                            velocity: state.inlet_velocity,
                            is_outlet: state.current_tool == Tool::Outlet,
                            color: EMITTER_COLORS[state.emitters.len() % EMITTER_COLORS.len()],
                        };
                        rasterize_emitters(
                            &[preview],
                            &mut emitter_cell_types,
                            &mut emitter_cell_props,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                        );
                        rasterize_direction_arrow(
                            [sx, sy],
                            state.emitter_angle,
                            &mut emitter_cell_types,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                        );
                    }
                    // Upload cell_props once (emitter velocities don't change between sub-steps)
                    state
                        .lbm
                        .upload_cell_props(&state.gpu.queue, &emitter_cell_props);

                    // Sub-step loop: ball physics + LBM in lockstep
                    let sub_dt = 1.0 / state.sim_speed as f32;
                    for _ in 0..state.sim_speed {
                        let sub_expl = state.ball_world.step(
                            sub_dt,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                            &state.base_cell_types,
                            state.gravity_on,
                        );
                        explosions.extend(sub_expl);

                        // Rasterize balls on top of emitter base
                        state.cell_types.copy_from_slice(&emitter_cell_types);
                        state.ball_world.rasterize(
                            &mut state.cell_types,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                        );
                        state
                            .lbm
                            .upload_cell_types(&state.gpu.queue, &state.cell_types);

                        // 1 LBM step
                        let mut enc = state.gpu.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("lbm_sub_step"),
                            },
                        );
                        state.lbm.step_one(&mut enc);
                        state.gpu.queue.submit([enc.finish()]);
                    }

                    // Keep cell_props in sync for rendering
                    state.cell_props.copy_from_slice(&emitter_cell_props);

                    // Collect dye injection points from emitters and explosions
                    let mut dye_points: Vec<DyeInjectPoint> = Vec::new();
                    for emitter in &state.emitters {
                        if emitter.is_outlet {
                            continue;
                        }
                        dye_points.push(DyeInjectPoint {
                            x: emitter.pos[0],
                            y: emitter.pos[1],
                            r: emitter.color[0],
                            g: emitter.color[1],
                            b: emitter.color[2],
                            radius: NOZZLE_WIDTH as f32 * 0.8,
                            strength: 0.5,
                            _pad: 0.0,
                        });
                    }
                    for expl in &explosions {
                        let bc = EMITTER_COLORS[(expl.color_id as usize) % EMITTER_COLORS.len()];
                        dye_points.push(DyeInjectPoint {
                            x: expl.x,
                            y: expl.y,
                            r: bc[0],
                            g: bc[1],
                            b: bc[2],
                            radius: expl.radius * 3.0,
                            strength: 2.0,
                            _pad: 0.0,
                        });
                    }
                    state
                        .dye_field
                        .upload_injections(&state.gpu.queue, &dye_points);
                    state.dye_field.params.vel_scale = state.sim_speed as f32;
                } else {
                    // Paused: still rasterize for rendering + preview
                    state.cell_types.copy_from_slice(&state.base_cell_types);
                    state.cell_props.copy_from_slice(&state.base_cell_props);
                    state.ball_world.rasterize(
                        &mut state.cell_types,
                        SIM_WIDTH,
                        SIM_HEIGHT,
                    );
                    rasterize_emitters(
                        &state.emitters,
                        &mut state.cell_types,
                        &mut state.cell_props,
                        SIM_WIDTH,
                        SIM_HEIGHT,
                    );
                    if state.current_tool == Tool::Inlet || state.current_tool == Tool::Outlet {
                        let win_size = state.window.inner_size();
                        let sx = (state.mouse_pos.0 / win_size.width as f64
                            * SIM_WIDTH as f64) as f32;
                        let sy = (state.mouse_pos.1 / win_size.height as f64
                            * SIM_HEIGHT as f64) as f32;
                        let preview = Emitter {
                            pos: [sx, sy],
                            angle: state.emitter_angle,
                            velocity: state.inlet_velocity,
                            is_outlet: state.current_tool == Tool::Outlet,
                            color: EMITTER_COLORS[state.emitters.len() % EMITTER_COLORS.len()],
                        };
                        rasterize_emitters(
                            &[preview],
                            &mut state.cell_types,
                            &mut state.cell_props,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                        );
                        rasterize_direction_arrow(
                            [sx, sy],
                            state.emitter_angle,
                            &mut state.cell_types,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                        );
                    }
                    state
                        .lbm
                        .upload_cell_types(&state.gpu.queue, &state.cell_types);
                    state
                        .lbm
                        .upload_cell_props(&state.gpu.queue, &state.cell_props);
                    state.lbm.update_params(&state.gpu.queue);
                }

                // Check goal completion
                if !state.level_complete && !state.level_goals.is_empty() {
                    check_goals(state);
                }

                // Final encoder: LBM output fields + dye advection + render
                let mut encoder =
                    state
                        .gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("frame_encoder"),
                        });

                if !state.paused {
                    state.lbm.compute_output(&mut encoder);
                    state.dye_field.step(&mut encoder);
                }

                // Upload ball render data
                upload_ball_data(state);

                let frame = state
                    .surface
                    .get_current_texture()
                    .expect("Failed to get surface texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // --- Sim render pass (Clear) ---
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
                    rpass.set_bind_group(1, &state.render_bind_group_1, &[]);
                    rpass.set_bind_group(2, &state.render_dye_bind_groups[state.dye_field.output_index()], &[]);
                    rpass.draw(0..3, 0..1);
                }

                // --- egui pass ---
                let egui_ctx = state.egui_ctx.clone();
                let win_size = state.window.inner_size();
                let level_names: Vec<String> = state.levels.iter().map(|l| l.name.clone()).collect();
                let goals_clone = state.level_goals.clone();
                let ui_inputs = UiInputs {
                    current_tool: state.current_tool,
                    inlet_velocity: state.inlet_velocity,
                    omega: state.lbm.params.omega,
                    emitter_angle: state.emitter_angle,
                    gravity_on: state.gravity_on,
                    paused: state.paused,
                    sim_speed: state.sim_speed,
                    fps: state.fps_display,
                    ball_count: state.ball_world.balls.len(),
                    emitter_count: state.emitters.len(),
                    show_level_selector: state.show_level_selector,
                    current_level: state.current_level,
                    level_names,
                    level_description: state.level_description.clone(),
                    level_complete: state.level_complete,
                    goals: goals_clone.clone(),
                };

                let raw_input = state.egui_state.take_egui_input(&state.window);
                let mut ui_out = None;
                let full_output = egui_ctx.run(raw_input, |ctx| {
                    ui_out = Some(draw_ui(ctx, &ui_inputs));
                    draw_goal_zones(
                        ctx,
                        &goals_clone,
                        |sx, sy| {
                            let px = sx / SIM_WIDTH as f32 * win_size.width as f32;
                            let py = sy / SIM_HEIGHT as f32 * win_size.height as f32;
                            (px, py)
                        },
                        |sr| sr / SIM_WIDTH as f32 * win_size.width as f32,
                    );
                });
                let ui_out = ui_out.unwrap();

                state.egui_state.handle_platform_output(&state.window, full_output.platform_output);

                let clipped_prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                let screen_desc = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [state.surface_config.width, state.surface_config.height],
                    pixels_per_point: full_output.pixels_per_point,
                };

                for (id, delta) in &full_output.textures_delta.set {
                    state.egui_renderer.update_texture(&state.gpu.device, &state.gpu.queue, *id, delta);
                }
                state.egui_renderer.update_buffers(
                    &state.gpu.device,
                    &state.gpu.queue,
                    &mut encoder,
                    &clipped_prims,
                    &screen_desc,
                );

                // egui render pass (Load, composites over sim).
                // Helper fn breaks the lifetime chain between egui_renderer and encoder.
                egui_render_pass(
                    &mut encoder,
                    &state.egui_renderer,
                    &view,
                    &clipped_prims,
                    &screen_desc,
                );

                for id in &full_output.textures_delta.free {
                    state.egui_renderer.free_texture(id);
                }

                state.gpu.queue.submit(Some(encoder.finish()));
                frame.present();

                // Apply UI changes to state
                if let Some(tool) = ui_out.tool_changed {
                    state.current_tool = tool;
                }
                if let Some(vel) = ui_out.velocity_changed {
                    state.inlet_velocity = vel;
                }
                if let Some(omega) = ui_out.omega_changed {
                    state.lbm.params.omega = omega;
                }
                if ui_out.gravity_toggled {
                    toggle_gravity(state);
                }
                if ui_out.pause_toggled {
                    state.paused = !state.paused;
                }
                if let Some(speed) = ui_out.sim_speed_changed {
                    state.sim_speed = speed.max(1);
                }
                state.show_level_selector = ui_out.show_level_selector;
                if let Some(level_idx) = ui_out.load_level {
                    apply_level(state, level_idx);
                }
                if ui_out.reset_requested {
                    reset_sim(state);
                }
                if ui_out.level_complete_dismissed {
                    state.level_complete = false;
                }

                // Readback fluid data for ball coupling (every 4th frame to save bandwidth)
                if !state.paused && !state.ball_world.balls.is_empty() && state.frame_count % 4 == 0 {
                    let output_size = (SIM_WIDTH * SIM_HEIGHT * 4) as u64
                        * std::mem::size_of::<f32>() as u64;
                    let mut enc = state
                        .gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("readback_encoder"),
                        });
                    enc.copy_buffer_to_buffer(
                        &state.lbm.output_buffer,
                        0,
                        &state.fluid_readback_buffer,
                        0,
                        output_size,
                    );
                    state.gpu.queue.submit(Some(enc.finish()));

                    let slice = state.fluid_readback_buffer.slice(..);
                    slice.map_async(wgpu::MapMode::Read, |_| {});
                    state.gpu.device.poll(wgpu::PollType::Wait).unwrap();
                    {
                        let data = slice.get_mapped_range();
                        let floats: &[f32] = bytemuck::cast_slice(&data);
                        state.fluid_data.copy_from_slice(floats);
                    }
                    state.fluid_readback_buffer.unmap();
                }

                // FPS counter
                state.frame_count += 1;
                state.fps_frame_count += 1;
                let elapsed = state.last_fps_time.elapsed();
                if elapsed.as_secs_f64() >= 1.0 {
                    state.fps_display = state.fps_frame_count as f64 / elapsed.as_secs_f64();
                    state.last_fps_time = std::time::Instant::now();
                    state.fps_frame_count = 0;
                }

                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn toggle_gravity(state: &mut AppState) {
    state.gravity_on = !state.gravity_on;
    if state.gravity_on {
        state.lbm.params.gravity_x = 0.0;
        state.lbm.params.gravity_y = -0.0001;
    } else {
        state.lbm.params.gravity_x = 0.0;
        state.lbm.params.gravity_y = 0.0;
    }
}

fn reset_sim(state: &mut AppState) {
    let num_cells = (SIM_WIDTH * SIM_HEIGHT) as usize;
    state.base_cell_types = vec![CELL_FLUID; num_cells];
    state.base_cell_props = vec![0.0f32; num_cells * 2];
    state.cell_types = vec![CELL_FLUID; num_cells];
    state.cell_props = vec![0.0f32; num_cells * 2];
    state.emitters.clear();
    state.ball_world.balls.clear();
    state.lbm.reset(&state.gpu.queue);
    state.dye_field.reset(&state.gpu.queue);
    state.current_level = None;
    state.level_goals.clear();
    state.level_description.clear();
    state.level_complete = false;
    state.locked_emitter_count = 0;
}

fn apply_level(state: &mut AppState, level_idx: usize) {
    let level = &state.levels[level_idx];
    let loaded = load_level(level);

    state.base_cell_types = loaded.cell_types.clone();
    state.base_cell_props = loaded.cell_props.clone();
    state.cell_types = loaded.cell_types;
    state.cell_props = loaded.cell_props;
    state.emitters = loaded.emitters;
    // Assign colors to level emitters
    for (i, em) in state.emitters.iter_mut().enumerate() {
        em.color = EMITTER_COLORS[i % EMITTER_COLORS.len()];
    }
    state.level_goals = loaded.goals;
    state.level_description = loaded.description;
    state.level_complete = false;
    state.locked_emitter_count = loaded.locked_emitter_count;
    state.current_level = Some(level_idx);
    state.inlet_velocity = loaded.inlet_velocity;
    state.lbm.params.omega = loaded.omega;

    // Set gravity
    state.gravity_on = loaded.gravity_on;
    if loaded.gravity_on {
        state.lbm.params.gravity_x = 0.0;
        state.lbm.params.gravity_y = -0.0001;
    } else {
        state.lbm.params.gravity_x = 0.0;
        state.lbm.params.gravity_y = 0.0;
    }

    // Spawn balls
    state.ball_world.balls.clear();
    for b in &loaded.balls {
        state.ball_world.spawn(b.x, b.y, b.radius);
    }

    // Reset LBM distributions and dye
    state.lbm.reset(&state.gpu.queue);
    state.dye_field.reset(&state.gpu.queue);
}

fn check_goals(state: &mut AppState) {
    for goal in &state.level_goals {
        for ball in &state.ball_world.balls {
            let dx = ball.pos[0] - goal.x;
            let dy = ball.pos[1] - goal.y;
            if dx * dx + dy * dy < goal.radius * goal.radius {
                state.level_complete = true;
                return;
            }
        }
    }
}

fn handle_key(state: &mut AppState, key: KeyCode) {
    match key {
        KeyCode::KeyG => {
            toggle_gravity(state);
        }
        KeyCode::ArrowUp => {
            let omega = (state.lbm.params.omega + 0.02).min(1.99);
            state.lbm.params.omega = omega;
        }
        KeyCode::ArrowDown => {
            let omega = (state.lbm.params.omega - 0.02).max(1.0);
            state.lbm.params.omega = omega;
        }
        KeyCode::ArrowRight => {
            state.inlet_velocity = (state.inlet_velocity + 0.01).min(0.40);
        }
        KeyCode::ArrowLeft => {
            state.inlet_velocity = (state.inlet_velocity - 0.01).max(0.01);
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
        KeyCode::Digit4 => {
            state.current_tool = Tool::Ball;
        }
        KeyCode::KeyC => {
            state.ball_world.balls.clear();
        }
        KeyCode::KeyR => {
            reset_sim(state);
        }
        KeyCode::Escape => {
            if state.window.fullscreen().is_some() {
                state.window.set_fullscreen(None);
            } else {
                std::process::exit(0);
            }
        }
        KeyCode::KeyF => {
            if state.window.fullscreen().is_some() {
                state.window.set_fullscreen(None);
            } else {
                state.window.set_fullscreen(Some(Fullscreen::Borderless(None)));
            }
        }
        KeyCode::Space => {
            state.paused = !state.paused;
        }
        _ => {}
    }
}

/// Stamp a single brush circle at grid position (cx, cy).
fn stamp_brush(state: &mut AppState, cx: i32, cy: i32, draw: bool) {
    for dy in -BRUSH_RADIUS..=BRUSH_RADIUS {
        for dx in -BRUSH_RADIUS..=BRUSH_RADIUS {
            if dx * dx + dy * dy > BRUSH_RADIUS * BRUSH_RADIUS {
                continue;
            }
            let px = cx + dx;
            let py = cy + dy;
            if px >= 0 && px < SIM_WIDTH as i32 && py >= 0 && py < SIM_HEIGHT as i32 {
                let idx = py as usize * SIM_WIDTH as usize + px as usize;
                if draw {
                    state.base_cell_types[idx] = CELL_SOLID;
                    state.base_cell_props[idx * 2] = 0.0;
                    state.base_cell_props[idx * 2 + 1] = 0.0;
                } else {
                    state.base_cell_types[idx] = CELL_FLUID;
                    state.base_cell_props[idx * 2] = 0.0;
                    state.base_cell_props[idx * 2 + 1] = 0.0;
                }
            }
        }
    }
}

/// Paint cells with interpolation for continuous strokes.
fn paint_cells(state: &mut AppState, draw: bool) {
    let win_size = state.window.inner_size();
    let sx = state.mouse_pos.0 / win_size.width as f64 * SIM_WIDTH as f64;
    let sy = state.mouse_pos.1 / win_size.height as f64 * SIM_HEIGHT as f64;
    let cx = sx as i32;
    let cy = sy as i32;

    if let Some((lx, ly)) = state.last_paint_pos {
        let dx = (cx - lx).abs();
        let dy = -(cy - ly).abs();
        let sx_step = if lx < cx { 1 } else { -1 };
        let sy_step = if ly < cy { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x = lx;
        let mut y = ly;
        loop {
            stamp_brush(state, x, y, draw);
            if x == cx && y == cy {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx_step;
            }
            if e2 <= dx {
                err += dx;
                y += sy_step;
            }
        }
    } else {
        stamp_brush(state, cx, cy, draw);
    }

    state.last_paint_pos = Some((cx, cy));
}

/// Rasterize emitter devices into cell_types and cell_props.
fn rasterize_emitters(
    emitters: &[Emitter],
    cell_types: &mut [u32],
    cell_props: &mut [f32],
    width: u32,
    height: u32,
) {
    for emitter in emitters {
        let dir_x = emitter.angle.cos();
        let dir_y = emitter.angle.sin();
        let tan_x = -emitter.angle.sin();
        let tan_y = emitter.angle.cos();

        // Nozzle face (inlet/outlet cells)
        for t in -(NOZZLE_WIDTH / 2)..=(NOZZLE_WIDTH / 2) {
            let x = (emitter.pos[0] + tan_x * t as f32).round() as i32;
            let y = (emitter.pos[1] + tan_y * t as f32).round() as i32;
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = y as usize * width as usize + x as usize;
                if emitter.is_outlet {
                    cell_types[idx] = CELL_OUTLET;
                } else {
                    cell_types[idx] = CELL_INLET;
                    cell_props[idx * 2] = dir_x * emitter.velocity;
                    cell_props[idx * 2 + 1] = dir_y * emitter.velocity;
                }
            }
        }

        // Pipe walls
        for d in 0..=(BODY_DEPTH + 1) {
            for &side in &[-(NOZZLE_WIDTH / 2 + 1), NOZZLE_WIDTH / 2 + 1] {
                let x = (emitter.pos[0] - dir_x * d as f32 + tan_x * side as f32).round() as i32;
                let y = (emitter.pos[1] - dir_y * d as f32 + tan_y * side as f32).round() as i32;
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    let idx = y as usize * width as usize + x as usize;
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }

        // Back face
        let back_d = (BODY_DEPTH + 1) as f32;
        for t in -(NOZZLE_WIDTH / 2)..=(NOZZLE_WIDTH / 2) {
            let x = (emitter.pos[0] - dir_x * back_d + tan_x * t as f32).round() as i32;
            let y = (emitter.pos[1] - dir_y * back_d + tan_y * t as f32).round() as i32;
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = y as usize * width as usize + x as usize;
                if emitter.is_outlet {
                    cell_types[idx] = CELL_INLET;
                    cell_props[idx * 2] = -dir_x * emitter.velocity;
                    cell_props[idx * 2 + 1] = -dir_y * emitter.velocity;
                } else {
                    cell_types[idx] = CELL_OUTLET;
                }
            }
        }
    }
}

/// Draw a solid-cell arrow from the nozzle center in the emission direction.
fn rasterize_direction_arrow(
    pos: [f32; 2],
    angle: f32,
    cell_types: &mut [u32],
    width: u32,
    height: u32,
) {
    let dir_x = angle.cos();
    let dir_y = angle.sin();
    let tan_x = -angle.sin();
    let tan_y = angle.cos();

    let start = (NOZZLE_WIDTH / 2 + 2) as f32;
    let shaft_len: i32 = 12;

    for i in 0..=shaft_len {
        let d = start + i as f32;
        let x = (pos[0] + dir_x * d).round() as i32;
        let y = (pos[1] + dir_y * d).round() as i32;
        if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
            let idx = y as usize * width as usize + x as usize;
            cell_types[idx] = CELL_SOLID;
        }
    }

    let tip = start + shaft_len as f32;
    for h in 1..=4_i32 {
        let back = tip - h as f32;
        for &side in &[-1.0_f32, 1.0] {
            let x = (pos[0] + dir_x * back + tan_x * side * h as f32).round() as i32;
            let y = (pos[1] + dir_y * back + tan_y * side * h as f32).round() as i32;
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = y as usize * width as usize + x as usize;
                cell_types[idx] = CELL_SOLID;
            }
        }
    }
}

/// Run egui render pass in a separate function to break the lifetime chain
/// between egui_renderer and encoder (avoids borrow-checker conflict).
fn egui_render_pass(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &egui_wgpu::Renderer,
    view: &wgpu::TextureView,
    clipped_prims: &[egui::ClippedPrimitive],
    screen_desc: &egui_wgpu::ScreenDescriptor,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("egui_render_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    })
    .forget_lifetime();
    renderer.render(&mut rpass, clipped_prims, screen_desc);
}

/// Upload ball render data to GPU buffers.
fn upload_ball_data(state: &mut AppState) {
    let params = BallRenderParams {
        num_balls: state.ball_world.balls.len() as u32,
        sim_width: SIM_WIDTH,
        sim_height: SIM_HEIGHT,
        _pad: 0,
    };
    state
        .gpu
        .queue
        .write_buffer(&state.ball_params_buffer, 0, bytemuck::bytes_of(&params));

    let gpu_data = state.ball_world.gpu_data();
    if !gpu_data.is_empty() {
        state
            .gpu
            .queue
            .write_buffer(&state.ball_render_buffer, 0, bytemuck::cast_slice(&gpu_data));
    }
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

    let ball_params_buffer = gpu.create_buffer_init(
        "ball_params",
        &[BallRenderParams {
            num_balls: 0,
            sim_width: SIM_WIDTH,
            sim_height: SIM_HEIGHT,
            _pad: 0,
        }],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    let ball_render_data = vec![BallGpuData { x: 0.0, y: 0.0, radius: 0.0, color_id: 0.0 }; MAX_BALLS];
    let ball_render_buffer = gpu.create_buffer_init(
        "ball_render_data",
        &ball_render_data,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let render_bgl_1 =
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bgl_1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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

    let render_bind_group_1 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bg_1"),
        layout: &render_bgl_1,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ball_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ball_render_buffer.as_entire_binding(),
            },
        ],
    });

    // Dye field
    let dye_field = DyeField::new(&gpu, SIM_WIDTH, SIM_HEIGHT, &lbm.output_buffer);

    let render_bgl_2 =
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("render_bgl_2"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
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

    // Two bind groups for ping-pong dye buffers (indexed by output_index)
    let render_dye_bind_groups = [
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_dye_bg_0"),
            layout: &render_bgl_2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dye_field.dye_buffer(0).as_entire_binding(),
            }],
        }),
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_dye_bg_1"),
            layout: &render_bgl_2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dye_field.dye_buffer(1).as_entire_binding(),
            }],
        }),
    ];

    let render_pipeline_layout =
        gpu.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render_pl"),
                bind_group_layouts: &[&render_bgl, &render_bgl_1, &render_bgl_2],
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
    let base_cell_types = vec![CELL_FLUID; num_cells];
    let base_cell_props = vec![0.0f32; num_cells * 2];
    let cell_types = vec![CELL_FLUID; num_cells];
    let cell_props = vec![0.0f32; num_cells * 2];

    let output_size = (num_cells * 4 * std::mem::size_of::<f32>()) as u64;
    let fluid_readback_buffer = gpu.create_buffer(
        "fluid_readback",
        output_size,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );

    // --- egui init ---
    let egui_ctx = egui::Context::default();
    let egui_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
        None,
    );
    let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, surface_format, None, 1, false);

    let levels = all_levels();

    AppState {
        window,
        surface,
        surface_config,
        gpu,
        lbm,
        render_pipeline,
        render_bind_group,
        render_bind_group_1,
        render_dye_bind_groups,
        dye_field,
        frame_count: 0,
        last_fps_time: std::time::Instant::now(),
        fps_frame_count: 0,
        base_cell_types,
        base_cell_props,
        cell_types,
        cell_props,
        mouse_pos: (0.0, 0.0),
        last_paint_pos: None,
        left_pressed: false,
        right_pressed: false,
        current_tool: Tool::Solid,
        gravity_on: true,
        paused: false,
        inlet_velocity: INLET_VELOCITY,
        sim_speed: DEFAULT_STEPS_PER_FRAME,
        emitters: Vec::new(),
        emitter_angle: 0.0,
        ball_world: {
            let mut bw = BallWorld::new();
            bw.gravity = 0.12 * (SIM_HEIGHT as f32 / 256.0);
            bw
        },
        ball_render_buffer,
        ball_params_buffer,
        fluid_readback_buffer,
        fluid_data: vec![0.0f32; num_cells * 4],
        egui_ctx,
        egui_state,
        egui_renderer,
        levels,
        current_level: None,
        show_level_selector: false,
        level_goals: Vec::new(),
        level_description: String::new(),
        level_complete: false,
        locked_emitter_count: 0,
        fps_display: 0.0,
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
