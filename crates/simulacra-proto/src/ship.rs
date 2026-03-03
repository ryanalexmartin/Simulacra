use bytemuck::{Pod, Zeroable};
use simulacra_engine::lbm::{CELL_INLET, CELL_SOLID};

pub const MAX_BULLETS: usize = 16;

// Ship constants
pub const SHIP_HALF_LEN: f32 = 12.0;
pub const SHIP_HALF_WIDTH: f32 = 7.0;
pub const SHIP_MASS: f32 = 1.0;
pub const THRUST_FORCE: f32 = 0.25;
pub const SHIP_DRAG: f32 = 0.995;
pub const THRUSTER_VELOCITY: f32 = 0.55;

// Bullet constants
pub const BULLET_SPEED: f32 = 4.5;
pub const BULLET_RADIUS: f32 = 2.5;
pub const BULLET_LIFETIME: u32 = 240;
pub const FIRE_COOLDOWN: u32 = 8;

// --- GPU data types ---

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ShipGpuData {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub alive: f32,
    pub half_len: f32,
    pub half_width: f32,
    pub thrusting: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BulletRenderParams {
    pub num_bullets: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BulletGpuData {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub _pad: f32,
}

// --- Ship ---

pub struct Ship {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub angle: f32,
    pub alive: bool,
    pub fire_cooldown: u32,
    pub thrusting: bool,
}

impl Ship {
    pub fn new(x: f32, y: f32, angle: f32) -> Self {
        Self {
            pos: [x, y],
            vel: [0.0, 0.0],
            angle,
            alive: true,
            fire_cooldown: 0,
            thrusting: false,
        }
    }

    pub fn step(
        &mut self,
        sub_dt: f32,
        forward: bool,
        backward: bool,
        strafe_left: bool,
        strafe_right: bool,
        target_angle: f32,
        width: u32,
        height: u32,
        base_cell_types: &[u32],
        skip_edge_clamp: bool,
    ) {
        if !self.alive {
            return;
        }

        // Instant cursor aim
        self.angle = target_angle;

        // Movement directions
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let fwd = [cos_a, sin_a];
        let left = [-sin_a, cos_a];

        self.thrusting = forward;
        if forward {
            self.vel[0] += fwd[0] * THRUST_FORCE * sub_dt / SHIP_MASS;
            self.vel[1] += fwd[1] * THRUST_FORCE * sub_dt / SHIP_MASS;
        }
        if backward {
            self.vel[0] -= fwd[0] * THRUST_FORCE * 0.6 * sub_dt / SHIP_MASS;
            self.vel[1] -= fwd[1] * THRUST_FORCE * 0.6 * sub_dt / SHIP_MASS;
        }
        if strafe_left {
            self.vel[0] += left[0] * THRUST_FORCE * 0.7 * sub_dt / SHIP_MASS;
            self.vel[1] += left[1] * THRUST_FORCE * 0.7 * sub_dt / SHIP_MASS;
        }
        if strafe_right {
            self.vel[0] -= left[0] * THRUST_FORCE * 0.7 * sub_dt / SHIP_MASS;
            self.vel[1] -= left[1] * THRUST_FORCE * 0.7 * sub_dt / SHIP_MASS;
        }

        // Drag (timestep-independent: same total drag per frame regardless of sim speed)
        let drag = SHIP_DRAG.powf(sub_dt);
        self.vel[0] *= drag;
        self.vel[1] *= drag;

        // Integrate position
        self.pos[0] += self.vel[0] * sub_dt;
        self.pos[1] += self.vel[1] * sub_dt;

        // Clamp to bounds (disabled in zone mode so ship can exit through openings)
        if !skip_edge_clamp {
            let margin = SHIP_HALF_LEN + 2.0;
            let w = width as f32;
            let h = height as f32;
            if self.pos[0] < margin {
                self.pos[0] = margin;
                self.vel[0] = self.vel[0].abs() * 0.5;
            }
            if self.pos[0] > w - margin {
                self.pos[0] = w - margin;
                self.vel[0] = -self.vel[0].abs() * 0.5;
            }
            if self.pos[1] < margin {
                self.pos[1] = margin;
                self.vel[1] = self.vel[1].abs() * 0.5;
            }
            if self.pos[1] > h - margin {
                self.pos[1] = h - margin;
                self.vel[1] = -self.vel[1].abs() * 0.5;
            }
        }

        // Ship-solid collision: sample cells around hull and push away
        let sample_r = SHIP_HALF_LEN;
        let num_samples = 12;
        let mut push_x = 0.0f32;
        let mut push_y = 0.0f32;
        let mut contact_count = 0u32;

        for i in 0..num_samples {
            let a = (i as f32) / (num_samples as f32) * std::f32::consts::TAU;
            let sx = self.pos[0] + sample_r * a.cos();
            let sy = self.pos[1] + sample_r * a.sin();
            let ix = sx as i32;
            let iy = sy as i32;
            if ix >= 0 && ix < width as i32 && iy >= 0 && iy < height as i32 {
                let idx = iy as usize * width as usize + ix as usize;
                if base_cell_types[idx] == CELL_SOLID {
                    let dx = self.pos[0] - sx;
                    let dy = self.pos[1] - sy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.001 {
                        push_x += dx / dist;
                        push_y += dy / dist;
                    }
                    contact_count += 1;
                }
            }
        }

        if contact_count > 0 {
            let len = (push_x * push_x + push_y * push_y).sqrt();
            if len > 0.001 {
                let nx = push_x / len;
                let ny = push_y / len;
                self.pos[0] += nx * 1.5;
                self.pos[1] += ny * 1.5;
                let vn = self.vel[0] * nx + self.vel[1] * ny;
                if vn < 0.0 {
                    self.vel[0] -= 1.5 * vn * nx;
                    self.vel[1] -= 1.5 * vn * ny;
                }
            }
        }
    }

    /// Compute the three vertices of the ship triangle.
    pub fn vertices(&self) -> [[f32; 2]; 3] {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let fwd = [cos_a, sin_a];
        let left = [-sin_a, cos_a];

        let nose = [
            self.pos[0] + fwd[0] * SHIP_HALF_LEN,
            self.pos[1] + fwd[1] * SHIP_HALF_LEN,
        ];
        let rear_left = [
            self.pos[0] - fwd[0] * SHIP_HALF_LEN + left[0] * SHIP_HALF_WIDTH,
            self.pos[1] - fwd[1] * SHIP_HALF_LEN + left[1] * SHIP_HALF_WIDTH,
        ];
        let rear_right = [
            self.pos[0] - fwd[0] * SHIP_HALF_LEN - left[0] * SHIP_HALF_WIDTH,
            self.pos[1] - fwd[1] * SHIP_HALF_LEN - left[1] * SHIP_HALF_WIDTH,
        ];

        [nose, rear_left, rear_right]
    }

    /// Rasterize the ship hull as CELL_SOLID cells.
    pub fn rasterize(&self, cell_types: &mut [u32], width: u32, height: u32) {
        if !self.alive {
            return;
        }
        let [v0, v1, v2] = self.vertices();
        rasterize_triangle(v0, v1, v2, cell_types, width, height, CELL_SOLID);
    }

    /// When thrusting, place inlet cells behind the ship for exhaust jet.
    pub fn rasterize_thruster(
        &self,
        cell_types: &mut [u32],
        cell_props: &mut [f32],
        width: u32,
        height: u32,
    ) {
        if !self.alive || !self.thrusting {
            return;
        }

        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let fwd = [cos_a, sin_a];
        let left = [-sin_a, cos_a];

        // Inlet cells behind the ship's rear
        let exhaust_dist = SHIP_HALF_LEN + 3.0;
        let exhaust_width = 5;
        let exhaust_vx = -fwd[0] * THRUSTER_VELOCITY;
        let exhaust_vy = -fwd[1] * THRUSTER_VELOCITY;

        for t in -(exhaust_width / 2)..=(exhaust_width / 2) {
            let x = (self.pos[0] - fwd[0] * exhaust_dist + left[0] * t as f32).round() as i32;
            let y = (self.pos[1] - fwd[1] * exhaust_dist + left[1] * t as f32).round() as i32;
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let idx = y as usize * width as usize + x as usize;
                cell_types[idx] = CELL_INLET;
                cell_props[idx * 2] = exhaust_vx;
                cell_props[idx * 2 + 1] = exhaust_vy;
            }
        }
    }

    /// Sample fluid around the hull and apply drag force.
    pub fn apply_fluid_forces(
        &mut self,
        fluid_data: &[f32],
        width: u32,
        height: u32,
        steps_per_frame: u32,
    ) {
        if !self.alive {
            return;
        }

        let w = width as f32;
        let h = height as f32;
        let drag_coeff = 0.003;
        let spf = steps_per_frame as f32;

        let sample_r = SHIP_HALF_LEN + 3.0;
        let num_samples = 12;
        let mut avg_ux = 0.0f32;
        let mut avg_uy = 0.0f32;
        let mut count = 0u32;

        for i in 0..num_samples {
            let angle = (i as f32) / (num_samples as f32) * std::f32::consts::TAU;
            let sx = self.pos[0] + sample_r * angle.cos();
            let sy = self.pos[1] + sample_r * angle.sin();
            let ix = sx as i32;
            let iy = sy as i32;
            if ix >= 0 && ix < w as i32 && iy >= 0 && iy < h as i32 {
                let idx = (iy as usize * width as usize + ix as usize) * 4;
                if idx + 3 < fluid_data.len() {
                    let rho = fluid_data[idx];
                    if rho > 0.0 {
                        avg_ux += fluid_data[idx + 1];
                        avg_uy += fluid_data[idx + 2];
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            avg_ux /= count as f32;
            avg_uy /= count as f32;

            let fluid_vx = avg_ux * spf;
            let fluid_vy = avg_uy * spf;
            let rel_vx = fluid_vx - self.vel[0];
            let rel_vy = fluid_vy - self.vel[1];
            let cross_section = 2.0 * SHIP_HALF_WIDTH;
            let force_x = drag_coeff * rel_vx * cross_section;
            let force_y = drag_coeff * rel_vy * cross_section;

            self.vel[0] += force_x / SHIP_MASS;
            self.vel[1] += force_y / SHIP_MASS;
        }
    }

    /// Nose position (bullet spawn point).
    pub fn nose(&self) -> [f32; 2] {
        [
            self.pos[0] + self.angle.cos() * SHIP_HALF_LEN,
            self.pos[1] + self.angle.sin() * SHIP_HALF_LEN,
        ]
    }

    /// Thruster position (exhaust dye injection point).
    pub fn thruster_pos(&self) -> [f32; 2] {
        [
            self.pos[0] - self.angle.cos() * (SHIP_HALF_LEN + 1.0),
            self.pos[1] - self.angle.sin() * (SHIP_HALF_LEN + 1.0),
        ]
    }

    pub fn gpu_data(&self) -> ShipGpuData {
        ShipGpuData {
            x: self.pos[0],
            y: self.pos[1],
            angle: self.angle,
            alive: if self.alive { 1.0 } else { 0.0 },
            half_len: SHIP_HALF_LEN,
            half_width: SHIP_HALF_WIDTH,
            thrusting: if self.thrusting { 1.0 } else { 0.0 },
            _pad: 0.0,
        }
    }
}

// --- Bullet ---

pub struct Bullet {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub age: u32,
    pub alive: bool,
}

impl Bullet {
    pub fn spawn(ship: &Ship) -> Self {
        let nose = ship.nose();
        Self {
            pos: nose,
            vel: [
                ship.vel[0] + ship.angle.cos() * BULLET_SPEED,
                ship.vel[1] + ship.angle.sin() * BULLET_SPEED,
            ],
            age: 0,
            alive: true,
        }
    }

    /// Step bullet position. Returns true if it hit a solid wall.
    pub fn step(
        &mut self,
        sub_dt: f32,
        width: u32,
        height: u32,
        base_cell_types: &[u32],
    ) -> bool {
        if !self.alive {
            return false;
        }

        self.pos[0] += self.vel[0] * sub_dt;
        self.pos[1] += self.vel[1] * sub_dt;

        // Out of bounds
        let w = width as f32;
        let h = height as f32;
        if self.pos[0] < 0.0 || self.pos[0] >= w || self.pos[1] < 0.0 || self.pos[1] >= h {
            self.alive = false;
            return false;
        }

        // Check collision with solid cells
        let ix = self.pos[0] as i32;
        let iy = self.pos[1] as i32;
        let r = BULLET_RADIUS.ceil() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let px = ix + dx;
                let py = iy + dy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = py as usize * width as usize + px as usize;
                    if base_cell_types[idx] == CELL_SOLID {
                        let fx = px as f32 + 0.5 - self.pos[0];
                        let fy = py as f32 + 0.5 - self.pos[1];
                        if fx * fx + fy * fy <= (BULLET_RADIUS + 1.0) * (BULLET_RADIUS + 1.0) {
                            self.alive = false;
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Rasterize bullet as small solid circle.
    pub fn rasterize(&self, cell_types: &mut [u32], width: u32, height: u32) {
        if !self.alive {
            return;
        }
        let cx = self.pos[0] as i32;
        let cy = self.pos[1] as i32;
        let ri = BULLET_RADIUS.ceil() as i32;
        let r2 = BULLET_RADIUS * BULLET_RADIUS;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let fx = px as f32 + 0.5 - self.pos[0];
                    let fy = py as f32 + 0.5 - self.pos[1];
                    if fx * fx + fy * fy <= r2 {
                        let idx = py as usize * width as usize + px as usize;
                        cell_types[idx] = CELL_SOLID;
                    }
                }
            }
        }
    }

    pub fn gpu_data(&self) -> BulletGpuData {
        BulletGpuData {
            x: self.pos[0],
            y: self.pos[1],
            radius: BULLET_RADIUS,
            _pad: 0.0,
        }
    }
}

// --- ExplosionWave ---

pub struct ExplosionWave {
    pub pos: [f32; 2],
    pub radius: f32,
    pub strength: f32,
    pub remaining_steps: u32,
}

impl ExplosionWave {
    pub fn new(pos: [f32; 2]) -> Self {
        Self {
            pos,
            radius: 16.0,
            strength: 0.30,
            remaining_steps: 5,
        }
    }

    /// Rasterize as a ring of inlet cells pointing radially outward.
    pub fn rasterize(
        &self,
        cell_types: &mut [u32],
        cell_props: &mut [f32],
        width: u32,
        height: u32,
    ) {
        if self.remaining_steps == 0 {
            return;
        }

        let cx = self.pos[0] as i32;
        let cy = self.pos[1] as i32;
        let r = self.radius.ceil() as i32;
        let inner_r2 = (self.radius - 2.0) * (self.radius - 2.0);
        let outer_r2 = (self.radius + 1.0) * (self.radius + 1.0);

        for dy in -r - 1..=r + 1 {
            for dx in -r - 1..=r + 1 {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let fx = px as f32 + 0.5 - self.pos[0];
                    let fy = py as f32 + 0.5 - self.pos[1];
                    let d2 = fx * fx + fy * fy;
                    if d2 >= inner_r2 && d2 <= outer_r2 {
                        let dist = d2.sqrt();
                        if dist > 0.001 {
                            let nx = fx / dist;
                            let ny = fy / dist;
                            let idx = py as usize * width as usize + px as usize;
                            cell_types[idx] = CELL_INLET;
                            cell_props[idx * 2] = nx * self.strength;
                            cell_props[idx * 2 + 1] = ny * self.strength;
                        }
                    }
                }
            }
        }
    }
}

// --- Triangle rasterization helper ---

fn rasterize_triangle(
    v0: [f32; 2],
    v1: [f32; 2],
    v2: [f32; 2],
    cell_types: &mut [u32],
    width: u32,
    height: u32,
    cell_type: u32,
) {
    let min_x = v0[0].min(v1[0]).min(v2[0]).max(0.0) as i32;
    let max_x = v0[0].max(v1[0]).max(v2[0]).min(width as f32 - 1.0) as i32;
    let min_y = v0[1].min(v1[1]).min(v2[1]).max(0.0) as i32;
    let max_y = v0[1].max(v1[1]).max(v2[1]).min(height as f32 - 1.0) as i32;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = [px as f32 + 0.5, py as f32 + 0.5];
            let e0 = edge_fn(v0, v1, p);
            let e1 = edge_fn(v1, v2, p);
            let e2 = edge_fn(v2, v0, p);

            if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
            {
                let idx = py as usize * width as usize + px as usize;
                cell_types[idx] = cell_type;
            }
        }
    }
}

fn edge_fn(v0: [f32; 2], v1: [f32; 2], p: [f32; 2]) -> f32 {
    (v1[0] - v0[0]) * (p[1] - v0[1]) - (v1[1] - v0[1]) * (p[0] - v0[0])
}
