use simulacra_engine::lbm::{CELL_FLUID, CELL_INLET, CELL_OUTLET, CELL_SOLID};

use crate::{Emitter, SIM_HEIGHT, SIM_WIDTH};

pub struct WallRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

pub struct WallCircle {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}

#[derive(Clone)]
pub struct GoalZone {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub label: String,
}

#[derive(Clone)]
pub struct LevelBall {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

pub struct LevelInletRow {
    pub y: u32,
    pub x_start: u32,
    pub x_end: u32,
    pub vx: f32,
    pub vy: f32,
    pub is_outlet: bool,
}

#[derive(Clone)]
pub struct LevelShipSpawn {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

pub struct Level {
    pub name: String,
    pub description: String,
    pub wall_rects: Vec<WallRect>,
    pub wall_circles: Vec<WallCircle>,
    pub emitters: Vec<Emitter>,
    pub balls: Vec<LevelBall>,
    pub goals: Vec<GoalZone>,
    pub inlet_rows: Vec<LevelInletRow>,
    pub gravity_on: bool,
    pub omega: f32,
    pub inlet_velocity: f32,
    pub lock_presets: bool,
    pub ship_spawn: Option<LevelShipSpawn>,
}

pub struct LoadedLevel {
    pub cell_types: Vec<u32>,
    pub cell_props: Vec<f32>,
    pub emitters: Vec<Emitter>,
    pub balls: Vec<LevelBall>,
    pub goals: Vec<GoalZone>,
    pub gravity_on: bool,
    pub omega: f32,
    pub inlet_velocity: f32,
    pub locked_emitter_count: usize,
    pub description: String,
    pub ship_spawn: Option<LevelShipSpawn>,
}

pub fn load_level(level: &Level) -> LoadedLevel {
    let num_cells = (SIM_WIDTH * SIM_HEIGHT) as usize;
    let mut cell_types = vec![CELL_FLUID; num_cells];
    let mut cell_props = vec![0.0f32; num_cells * 2];

    // Rasterize wall rects
    for rect in &level.wall_rects {
        for dy in 0..rect.h {
            for dx in 0..rect.w {
                let px = rect.x + dx;
                let py = rect.y + dy;
                if px < SIM_WIDTH && py < SIM_HEIGHT {
                    let idx = py as usize * SIM_WIDTH as usize + px as usize;
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    // Rasterize wall circles
    for circ in &level.wall_circles {
        let r2 = circ.radius * circ.radius;
        let min_x = (circ.cx - circ.radius - 1.0).max(0.0) as u32;
        let max_x = (circ.cx + circ.radius + 1.0).min(SIM_WIDTH as f32 - 1.0) as u32;
        let min_y = (circ.cy - circ.radius - 1.0).max(0.0) as u32;
        let max_y = (circ.cy + circ.radius + 1.0).min(SIM_HEIGHT as f32 - 1.0) as u32;
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 - circ.cx;
                let dy = py as f32 - circ.cy;
                if dx * dx + dy * dy <= r2 {
                    let idx = py as usize * SIM_WIDTH as usize + px as usize;
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    // Set inlet/outlet rows directly (for lid-driven cavity, channels, etc.)
    for row in &level.inlet_rows {
        for px in row.x_start..row.x_end {
            if px < SIM_WIDTH && row.y < SIM_HEIGHT {
                let idx = row.y as usize * SIM_WIDTH as usize + px as usize;
                if row.is_outlet {
                    cell_types[idx] = CELL_OUTLET;
                } else {
                    cell_types[idx] = CELL_INLET;
                    cell_props[idx * 2] = row.vx;
                    cell_props[idx * 2 + 1] = row.vy;
                }
            }
        }
    }

    let locked_emitter_count = if level.lock_presets { level.emitters.len() } else { 0 };
    LoadedLevel {
        cell_types,
        cell_props,
        emitters: level.emitters.clone(),
        balls: level.balls.clone(),
        goals: level.goals.clone(),
        gravity_on: level.gravity_on,
        omega: level.omega,
        inlet_velocity: level.inlet_velocity,
        locked_emitter_count,
        description: level.description.clone(),
        ship_spawn: level.ship_spawn.clone(),
    }
}

pub fn all_levels() -> Vec<Level> {
    vec![
        venturi_tube(),
        von_karman_vortex_street(),
        lid_driven_cavity(),
        ball_fluid_coupling(),
        puzzle_guide_the_ball(),
        open_arena(),
        current_run(),
        target_practice(),
    ]
}

/// Level 1: Horizontal channel with constriction. Verifies fluid acceleration.
fn venturi_tube() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;
    let channel_top = h / 2 - 80;
    let channel_bot = h / 2 + 80;
    let constriction_top = h / 2 - 30;
    let constriction_bot = h / 2 + 30;
    let constriction_start = w / 2 - 60;
    let constriction_end = w / 2 + 60;

    // Build walls: top wall, bottom wall, with tapered constriction
    let mut wall_rects = vec![
        // Top wall before constriction
        WallRect { x: 0, y: channel_top, w: constriction_start, h: 3 },
        // Top wall after constriction
        WallRect { x: constriction_end, y: channel_top, w: w - constriction_end, h: 3 },
        // Bottom wall before constriction
        WallRect { x: 0, y: channel_bot, w: constriction_start, h: 3 },
        // Bottom wall after constriction
        WallRect { x: constriction_end, y: channel_bot, w: w - constriction_end, h: 3 },
        // Constriction top wall
        WallRect { x: constriction_start, y: constriction_top, w: constriction_end - constriction_start, h: 3 },
        // Constriction bottom wall
        WallRect { x: constriction_start, y: constriction_bot, w: constriction_end - constriction_start, h: 3 },
    ];

    // Taper: connect channel walls to constriction walls (use f32 to avoid u32 underflow)
    let taper_len = 40u32;
    let ct = channel_top as f32;
    let cb = channel_bot as f32;
    let cst = constriction_top as f32;
    let csb = constriction_bot as f32;

    // Top taper (left side): slopes from channel_top down to constriction_top
    for i in 0..taper_len {
        let frac = i as f32 / taper_len as f32;
        let y = ct + (cst - ct) * frac;
        let x = constriction_start - taper_len + i;
        wall_rects.push(WallRect { x, y: y as u32, w: 2, h: 2 });
    }
    // Top taper (right side)
    for i in 0..taper_len {
        let frac = 1.0 - i as f32 / taper_len as f32;
        let y = ct + (cst - ct) * frac;
        let x = constriction_end + i;
        wall_rects.push(WallRect { x, y: y as u32, w: 2, h: 2 });
    }
    // Bottom taper (left side): slopes from channel_bot up to constriction_bot
    for i in 0..taper_len {
        let frac = i as f32 / taper_len as f32;
        let y = cb + (csb - cb) * frac;
        let x = constriction_start - taper_len + i;
        wall_rects.push(WallRect { x, y: y as u32, w: 2, h: 2 });
    }
    // Bottom taper (right side)
    for i in 0..taper_len {
        let frac = 1.0 - i as f32 / taper_len as f32;
        let y = cb + (csb - cb) * frac;
        let x = constriction_end + i;
        wall_rects.push(WallRect { x, y: y as u32, w: 2, h: 2 });
    }

    // Inlet on left, outlet on right
    let mut inlet_rows = Vec::new();
    for y in (channel_top + 3)..(channel_bot) {
        inlet_rows.push(LevelInletRow { y, x_start: 0, x_end: 3, vx: 0.08, vy: 0.0, is_outlet: false });
    }
    for y in (channel_top + 3)..(channel_bot) {
        inlet_rows.push(LevelInletRow { y, x_start: w - 3, x_end: w, vx: 0.0, vy: 0.0, is_outlet: true });
    }

    Level {
        name: "Venturi Tube".into(),
        description: "Flow through a constriction. Watch fluid accelerate in the narrow section.\nInlet velocity preset on the left.".into(),
        wall_rects,
        wall_circles: vec![],
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows,
        gravity_on: false,
        omega: 1.85,
        inlet_velocity: 0.08,
        lock_presets: false,
        ship_spawn: None,
    }
}

/// Level 2: Flow past a solid circle. Verifies alternating vortex shedding.
fn von_karman_vortex_street() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;
    let cy = h as f32 / 2.0;
    let cx = w as f32 * 0.25;

    // Channel walls top and bottom
    let wall_rects = vec![
        WallRect { x: 0, y: 0, w, h: 3 },
        WallRect { x: 0, y: h - 3, w, h: 3 },
    ];

    // Circular obstacle
    let wall_circles = vec![
        WallCircle { cx, cy, radius: 20.0 },
    ];

    // Inlet on left, outlet on right
    let mut inlet_rows = Vec::new();
    for y in 3..(h - 3) {
        inlet_rows.push(LevelInletRow { y, x_start: 0, x_end: 3, vx: 0.20, vy: 0.0, is_outlet: false });
    }
    for y in 3..(h - 3) {
        inlet_rows.push(LevelInletRow { y, x_start: w - 3, x_end: w, vx: 0.0, vy: 0.0, is_outlet: true });
    }

    Level {
        name: "Von Karman Vortex Street".into(),
        description: "Flow past a cylinder. At this Reynolds number, alternating vortices\nshould shed behind the obstacle.".into(),
        wall_rects,
        wall_circles,
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows,
        gravity_on: false,
        omega: 1.7,
        inlet_velocity: 0.20,
        lock_presets: false,
        ship_spawn: None,
    }
}

/// Level 3: Sealed box with moving lid. Classic CFD benchmark.
fn lid_driven_cavity() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;
    let margin = 100u32;
    let box_x = margin;
    let box_y = margin;
    let box_w = w - 2 * margin;
    let box_h = h - 2 * margin;

    // Box walls: left, right, bottom. Top is the moving lid.
    let wall_rects = vec![
        // Left wall
        WallRect { x: box_x, y: box_y, w: 3, h: box_h },
        // Right wall
        WallRect { x: box_x + box_w - 3, y: box_y, w: 3, h: box_h },
        // Bottom wall
        WallRect { x: box_x, y: box_y + box_h - 3, w: box_w, h: 3 },
        // Top wall (thin, lid is just above the interior)
        WallRect { x: box_x, y: box_y, w: box_w, h: 3 },
    ];

    // Moving lid: inlet row just below the top wall, moving right
    let mut inlet_rows = Vec::new();
    for y in (box_y + 3)..(box_y + 5) {
        inlet_rows.push(LevelInletRow {
            y,
            x_start: box_x + 3,
            x_end: box_x + box_w - 3,
            vx: 0.10,
            vy: 0.0,
            is_outlet: false,
        });
    }

    Level {
        name: "Lid-Driven Cavity".into(),
        description: "Sealed box with a moving lid (top). Classic CFD benchmark.\nA primary vortex should form in the center.".into(),
        wall_rects,
        wall_circles: vec![],
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows,
        gravity_on: false,
        omega: 1.85,
        inlet_velocity: 0.10,
        lock_presets: false,
        ship_spawn: None,
    }
}

/// Level 4: Horizontal channel with balls above. Verifies drag + buoyancy.
fn ball_fluid_coupling() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;
    let channel_top = h / 2 - 100;
    let channel_bot = h / 2 + 100;

    let wall_rects = vec![
        WallRect { x: 0, y: channel_top, w, h: 3 },
        WallRect { x: 0, y: channel_bot, w, h: 3 },
    ];

    // Inlet on left, outlet on right
    let mut inlet_rows = Vec::new();
    for y in (channel_top + 3)..(channel_bot) {
        inlet_rows.push(LevelInletRow { y, x_start: 0, x_end: 3, vx: 0.10, vy: 0.0, is_outlet: false });
    }
    for y in (channel_top + 3)..(channel_bot) {
        inlet_rows.push(LevelInletRow { y, x_start: w - 3, x_end: w, vx: 0.0, vy: 0.0, is_outlet: true });
    }

    // 3 balls positioned in the channel
    let mid_y = (channel_top + channel_bot) as f32 / 2.0;
    let balls = vec![
        LevelBall { x: w as f32 * 0.3, y: mid_y - 30.0, radius: 8.0 },
        LevelBall { x: w as f32 * 0.5, y: mid_y, radius: 10.0 },
        LevelBall { x: w as f32 * 0.7, y: mid_y + 20.0, radius: 6.0 },
    ];

    Level {
        name: "Ball-Fluid Coupling".into(),
        description: "Horizontal channel with 3 balls. Watch the flow push them.\nGravity is on -- balls should settle and drift.".into(),
        wall_rects,
        wall_circles: vec![],
        emitters: vec![],
        balls,
        goals: vec![],
        inlet_rows,
        gravity_on: true,
        omega: 1.85,
        inlet_velocity: 0.10,
        lock_presets: false,
        ship_spawn: None,
    }
}

/// Level 5: Ball at start, maze walls, green goal. Player creates flow.
fn puzzle_guide_the_ball() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;

    // Maze-like walls
    let wall_rects = vec![
        // Outer boundary
        WallRect { x: 50, y: 50, w: w - 100, h: 3 },         // top
        WallRect { x: 50, y: h - 53, w: w - 100, h: 3 },     // bottom
        WallRect { x: 50, y: 50, w: 3, h: h - 100 },         // left
        WallRect { x: w - 53, y: 50, w: 3, h: h - 100 },     // right

        // Internal maze walls
        // Horizontal barrier from left, gap on right
        WallRect { x: 53, y: h / 3, w: w / 2, h: 3 },
        // Horizontal barrier from right, gap on left
        WallRect { x: w / 2 - 50, y: 2 * h / 3, w: w / 2 - 3, h: 3 },
        // Vertical barrier in middle, gap top/bottom
        WallRect { x: w / 2, y: h / 3 + 3, w: 3, h: h / 3 - 30 },
    ];

    // Ball starts top-left
    let balls = vec![
        LevelBall { x: 120.0, y: 120.0, radius: 8.0 },
    ];

    // Goal bottom-right
    let goals = vec![
        GoalZone {
            x: (w - 120) as f32,
            y: (h - 120) as f32,
            radius: 30.0,
            label: "Goal".into(),
        },
    ];

    Level {
        name: "Puzzle: Guide the Ball".into(),
        description: "Guide the ball to the green goal zone!\nPlace inlet/outlet emitters to create flow that pushes the ball through the maze.".into(),
        wall_rects,
        wall_circles: vec![],
        emitters: vec![],
        balls,
        goals,
        inlet_rows: vec![],
        gravity_on: true,
        omega: 1.85,
        inlet_velocity: 0.15,
        lock_presets: true,
        ship_spawn: None,
    }
}

// --- Shooter levels ---

/// Level 6: Open arena with scattered obstacles. Fly and shoot freely.
fn open_arena() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;

    // Edge walls
    let wall_rects = vec![
        WallRect { x: 0, y: 0, w, h: 3 },
        WallRect { x: 0, y: h - 3, w, h: 3 },
        WallRect { x: 0, y: 0, w: 3, h },
        WallRect { x: w - 3, y: 0, w: 3, h },
    ];

    // Scattered circular obstacles
    let wall_circles = vec![
        WallCircle { cx: w as f32 * 0.25, cy: h as f32 * 0.35, radius: 18.0 },
        WallCircle { cx: w as f32 * 0.7, cy: h as f32 * 0.6, radius: 22.0 },
        WallCircle { cx: w as f32 * 0.5, cy: h as f32 * 0.2, radius: 14.0 },
        WallCircle { cx: w as f32 * 0.85, cy: h as f32 * 0.25, radius: 16.0 },
        WallCircle { cx: w as f32 * 0.15, cy: h as f32 * 0.75, radius: 20.0 },
    ];

    Level {
        name: "Open Arena".into(),
        description: "Fly through the arena and shoot!\nW=thrust, A/D=rotate, Space=fire, P=pause\nWatch your exhaust create fluid wakes.".into(),
        wall_rects,
        wall_circles,
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows: vec![],
        gravity_on: false,
        omega: 1.7,
        inlet_velocity: 0.10,
        lock_presets: false,
        ship_spawn: Some(LevelShipSpawn {
            x: w as f32 / 2.0,
            y: h as f32 / 2.0,
            angle: 0.0,
        }),
    }
}

/// Level 7: Strong horizontal flow. Ship must navigate in a current.
fn current_run() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;

    // Channel walls
    let wall_rects = vec![
        WallRect { x: 0, y: 0, w, h: 3 },
        WallRect { x: 0, y: h - 3, w, h: 3 },
    ];

    // Some obstacles in the flow
    let wall_circles = vec![
        WallCircle { cx: w as f32 * 0.4, cy: h as f32 * 0.4, radius: 20.0 },
        WallCircle { cx: w as f32 * 0.6, cy: h as f32 * 0.65, radius: 18.0 },
        WallCircle { cx: w as f32 * 0.8, cy: h as f32 * 0.35, radius: 15.0 },
    ];

    // Strong inlet on left, outlet on right
    let mut inlet_rows = Vec::new();
    for y in 3..(h - 3) {
        inlet_rows.push(LevelInletRow {
            y,
            x_start: 0,
            x_end: 3,
            vx: 0.12,
            vy: 0.0,
            is_outlet: false,
        });
    }
    for y in 3..(h - 3) {
        inlet_rows.push(LevelInletRow {
            y,
            x_start: w - 3,
            x_end: w,
            vx: 0.0,
            vy: 0.0,
            is_outlet: true,
        });
    }

    Level {
        name: "Current Run".into(),
        description: "Strong horizontal flow pushes you right.\nFly upstream! Shoot to create turbulence.\nW=thrust, A/D=rotate, Space=fire".into(),
        wall_rects,
        wall_circles,
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows,
        gravity_on: false,
        omega: 1.7,
        inlet_velocity: 0.12,
        lock_presets: true,
        ship_spawn: Some(LevelShipSpawn {
            x: w as f32 * 0.15,
            y: h as f32 / 2.0,
            angle: std::f32::consts::PI, // face left (upstream)
        }),
    }
}

/// Level 8: Walled arena with solid blocks to shoot at.
fn target_practice() -> Level {
    let w = SIM_WIDTH;
    let h = SIM_HEIGHT;

    // Edge walls
    let mut wall_rects = vec![
        WallRect { x: 0, y: 0, w, h: 3 },
        WallRect { x: 0, y: h - 3, w, h: 3 },
        WallRect { x: 0, y: 0, w: 3, h },
        WallRect { x: w - 3, y: 0, w: 3, h },
    ];

    // Target blocks scattered around right side
    wall_rects.push(WallRect { x: w * 3 / 4, y: h / 4, w: 30, h: 8 });
    wall_rects.push(WallRect { x: w * 3 / 4 - 40, y: h / 2, w: 8, h: 30 });
    wall_rects.push(WallRect { x: w * 3 / 4 + 20, y: h * 3 / 4 - 15, w: 25, h: 8 });
    wall_rects.push(WallRect { x: w / 2, y: h / 3, w: 8, h: 8 });
    wall_rects.push(WallRect { x: w / 2 + 40, y: h * 2 / 3, w: 8, h: 8 });

    Level {
        name: "Target Practice".into(),
        description: "Shoot the blocks! Watch pressure waves\nripple through the fluid on impact.\nW=thrust, A/D=rotate, Space=fire".into(),
        wall_rects,
        wall_circles: vec![],
        emitters: vec![],
        balls: vec![],
        goals: vec![],
        inlet_rows: vec![],
        gravity_on: false,
        omega: 1.7,
        inlet_velocity: 0.10,
        lock_presets: false,
        ship_spawn: Some(LevelShipSpawn {
            x: w as f32 * 0.15,
            y: h as f32 / 2.0,
            angle: 0.0,
        }),
    }
}
