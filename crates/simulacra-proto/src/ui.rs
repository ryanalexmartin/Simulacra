use crate::levels::GoalZone;
use crate::Tool;

/// Mutable inputs extracted from AppState so we can pass them into egui_ctx.run()
/// without borrow-checker conflicts (egui_ctx lives inside AppState).
pub struct UiInputs {
    pub current_tool: Tool,
    pub inlet_velocity: f32,
    pub omega: f32,
    pub emitter_angle: f32,
    pub gravity_on: bool,
    pub paused: bool,
    pub sim_speed: u32,
    pub fps: f64,
    pub ball_count: usize,
    pub emitter_count: usize,
    pub show_level_selector: bool,
    pub current_level: Option<usize>,
    pub level_names: Vec<String>,
    pub level_description: String,
    pub level_complete: bool,
    pub goals: Vec<GoalZone>,
}

/// Changes requested by the UI, applied back to AppState after egui_ctx.run().
#[derive(Default)]
pub struct UiOutputs {
    pub tool_changed: Option<Tool>,
    pub velocity_changed: Option<f32>,
    pub omega_changed: Option<f32>,
    pub gravity_toggled: bool,
    pub pause_toggled: bool,
    pub sim_speed_changed: Option<u32>,
    pub show_level_selector: bool,
    pub load_level: Option<usize>,
    pub reset_requested: bool,
    pub level_complete_dismissed: bool,
}

pub fn draw_ui(ctx: &egui::Context, inputs: &UiInputs) -> UiOutputs {
    let mut outputs = UiOutputs::default();
    outputs.show_level_selector = inputs.show_level_selector;

    // Bottom toolbar panel
    egui::TopBottomPanel::bottom("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Tool buttons
            ui.label("Tool:");
            let mut tool = inputs.current_tool;
            if ui.selectable_label(tool == Tool::Solid, "Solid [1]").clicked() {
                tool = Tool::Solid;
            }
            if ui.selectable_label(tool == Tool::Inlet, "Inlet [2]").clicked() {
                tool = Tool::Inlet;
            }
            if ui.selectable_label(tool == Tool::Outlet, "Outlet [3]").clicked() {
                tool = Tool::Outlet;
            }
            if ui.selectable_label(tool == Tool::Ball, "Ball [4]").clicked() {
                tool = Tool::Ball;
            }
            if tool != inputs.current_tool {
                outputs.tool_changed = Some(tool);
            }

            ui.separator();

            // Velocity slider
            let mut vel = inputs.inlet_velocity;
            ui.label("Vel:");
            if ui.add(egui::Slider::new(&mut vel, 0.01..=0.40).max_decimals(2)).changed() {
                outputs.velocity_changed = Some(vel);
            }

            ui.separator();

            // Sim speed slider
            let mut speed = inputs.sim_speed as f32;
            ui.label("Speed:");
            if ui.add(egui::Slider::new(&mut speed, 1.0..=200.0).logarithmic(true).max_decimals(0)).changed() {
                outputs.sim_speed_changed = Some(speed as u32);
            }

            ui.separator();

            // Omega slider
            let mut omega = inputs.omega;
            ui.label("\u{03C9}:");
            if ui.add(egui::Slider::new(&mut omega, 1.0..=1.99).max_decimals(2)).changed() {
                outputs.omega_changed = Some(omega);
            }

            ui.separator();

            // Emitter angle readout
            ui.label(format!("Angle: {:.0}\u{00B0}", inputs.emitter_angle.to_degrees()));

            ui.separator();

            // Gravity toggle
            let grav_label = if inputs.gravity_on { "Gravity: ON" } else { "Gravity: OFF" };
            if ui.button(grav_label).clicked() {
                outputs.gravity_toggled = true;
            }

            // Pause/Play
            let pause_label = if inputs.paused { "\u{25B6} Play" } else { "\u{23F8} Pause" };
            if ui.button(pause_label).clicked() {
                outputs.pause_toggled = true;
            }

            ui.separator();

            // Status
            ui.label(format!("{:.0}fps", inputs.fps));
            let nu = (1.0 / inputs.omega as f64 - 0.5) / 3.0;
            let re = inputs.inlet_velocity as f64 * 1024.0 / nu;
            ui.label(format!("Re~{:.0}", re));
            ui.label(format!("balls:{}", inputs.ball_count));
            ui.label(format!("emitters:{}", inputs.emitter_count));

            ui.separator();

            // Levels button
            if ui.button("Levels...").clicked() {
                outputs.show_level_selector = !outputs.show_level_selector;
            }

            // Reset button
            if ui.button("Reset [R]").clicked() {
                outputs.reset_requested = true;
            }
        });
    });

    // Level selector window
    if outputs.show_level_selector {
        let mut open = true;
        egui::Window::new("Level Selector")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                for (i, name) in inputs.level_names.iter().enumerate() {
                    let selected = inputs.current_level == Some(i);
                    let label = if selected {
                        format!("\u{25B6} {}", name)
                    } else {
                        format!("  {}", name)
                    };
                    if ui.selectable_label(selected, &label).clicked() {
                        outputs.load_level = Some(i);
                        outputs.show_level_selector = false;
                    }
                }
                ui.separator();
                if ui.button("Free Play (clear level)").clicked() {
                    outputs.reset_requested = true;
                    outputs.load_level = None;
                    outputs.show_level_selector = false;
                }
            });
        if !open {
            outputs.show_level_selector = false;
        }
    }

    // Level description overlay (top-left)
    if inputs.current_level.is_some() && !inputs.level_description.is_empty() {
        egui::Window::new("Level")
            .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
            .resizable(false)
            .collapsible(true)
            .title_bar(true)
            .show(ctx, |ui| {
                ui.label(&inputs.level_description);
            });
    }

    // Goal zone rendering
    if !inputs.goals.is_empty() {
        // We draw goal zones using egui painter in screen space.
        // This requires converting sim coords to screen coords.
        // We'll do this in the caller since we need window size info.
    }

    // Level complete popup
    if inputs.level_complete {
        egui::Window::new("Level Complete!")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.heading("Congratulations!");
                ui.label("The ball reached the goal zone.");
                if ui.button("Continue").clicked() {
                    outputs.level_complete_dismissed = true;
                }
            });
    }

    outputs
}

/// Draw goal zones as translucent green circles on the egui painter.
/// sim_to_screen converts (sim_x, sim_y) to screen pixel coordinates.
pub fn draw_goal_zones(
    ctx: &egui::Context,
    goals: &[GoalZone],
    sim_to_screen: impl Fn(f32, f32) -> (f32, f32),
    sim_radius_to_screen: impl Fn(f32) -> f32,
) {
    let painter = ctx.layer_painter(egui::LayerId::background());
    for goal in goals {
        let (sx, sy) = sim_to_screen(goal.x, goal.y);
        let sr = sim_radius_to_screen(goal.radius);
        let center = egui::pos2(sx, sy);

        // Filled circle
        painter.circle_filled(
            center,
            sr,
            egui::Color32::from_rgba_unmultiplied(50, 220, 80, 60),
        );
        // Outline
        painter.circle_stroke(
            center,
            sr,
            egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(50, 220, 80, 180)),
        );
        // Label
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            &goal.label,
            egui::FontId::proportional(14.0),
            egui::Color32::from_rgba_unmultiplied(200, 255, 200, 200),
        );
    }
}
