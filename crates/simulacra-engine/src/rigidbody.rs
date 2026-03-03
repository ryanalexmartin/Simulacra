use bytemuck::{Pod, Zeroable};

pub const MAX_BALLS: usize = 64;

/// GPU-side ball data for the render shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BallGpuData {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color_id: f32,
}

/// Uniform params for ball rendering.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BallRenderParams {
    pub num_balls: u32,
    pub sim_width: u32,
    pub sim_height: u32,
    pub _pad: u32,
}

/// A single rigid ball.
pub struct Ball {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub radius: f32,
    pub mass: f32,
    pub restitution: f32,
    pub color_id: u32,
}

/// Event generated when a ball shatters on hard impact.
pub struct BallExplosion {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color_id: u32,
    pub speed: f32,
}

/// Manages a collection of rigid balls with simple physics.
pub struct BallWorld {
    pub balls: Vec<Ball>,
    pub gravity: f32,
    next_color: u32,
}

impl BallWorld {
    pub fn new() -> Self {
        Self {
            balls: Vec::new(),
            gravity: 0.03,
            next_color: 0,
        }
    }

    /// Spawn a ball at (x, y) with the given radius. Returns index if successful.
    pub fn spawn(&mut self, x: f32, y: f32, radius: f32) -> Option<usize> {
        if self.balls.len() >= MAX_BALLS {
            return None;
        }
        let color_id = self.next_color;
        self.next_color += 1;
        let mass = radius * radius * 0.06; // area-proportional mass
        self.balls.push(Ball {
            pos: [x, y],
            vel: [0.0, 0.0],
            radius,
            mass,
            restitution: 0.75,
            color_id,
        });
        Some(self.balls.len() - 1)
    }

    /// Remove the nearest ball within threshold distance. Returns true if removed.
    pub fn remove_nearest(&mut self, x: f32, y: f32, threshold: f32) -> bool {
        let mut best_idx = None;
        let mut best_dist = threshold * threshold;
        for (i, b) in self.balls.iter().enumerate() {
            let dx = b.pos[0] - x;
            let dy = b.pos[1] - y;
            let d2 = dx * dx + dy * dy;
            if d2 < best_dist {
                best_dist = d2;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            self.balls.swap_remove(idx);
            true
        } else {
            false
        }
    }

    /// Step the physics simulation.
    /// y=0 is TOP, y increases downward. Gravity adds to vel[1].
    /// Returns a list of explosions for balls that shattered on hard impact.
    pub fn step(
        &mut self,
        dt: f32,
        width: u32,
        height: u32,
        base_cell_types: &[u32],
        gravity_on: bool,
    ) -> Vec<BallExplosion> {
        let w = width as f32;
        let h = height as f32;

        // 1. Apply gravity
        if gravity_on {
            for b in &mut self.balls {
                b.vel[1] += self.gravity * dt;
            }
        }

        // Snapshot velocities after gravity, before collisions
        let vel_before: Vec<[f32; 2]> = self.balls.iter().map(|b| b.vel).collect();

        // 2. Ball-ball elastic collisions (O(n^2))
        let n = self.balls.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.balls[j].pos[0] - self.balls[i].pos[0];
                let dy = self.balls[j].pos[1] - self.balls[i].pos[1];
                let dist2 = dx * dx + dy * dy;
                let min_dist = self.balls[i].radius + self.balls[j].radius;

                if dist2 < min_dist * min_dist && dist2 > 0.0001 {
                    let dist = dist2.sqrt();
                    let nx = dx / dist;
                    let ny = dy / dist;

                    // Separate balls
                    let overlap = min_dist - dist;
                    let total_mass = self.balls[i].mass + self.balls[j].mass;
                    let sep_i = overlap * self.balls[j].mass / total_mass;
                    let sep_j = overlap * self.balls[i].mass / total_mass;
                    self.balls[i].pos[0] -= nx * sep_i;
                    self.balls[i].pos[1] -= ny * sep_i;
                    self.balls[j].pos[0] += nx * sep_j;
                    self.balls[j].pos[1] += ny * sep_j;

                    // Elastic collision impulse
                    let dvx = self.balls[i].vel[0] - self.balls[j].vel[0];
                    let dvy = self.balls[i].vel[1] - self.balls[j].vel[1];
                    let dvn = dvx * nx + dvy * ny;
                    if dvn > 0.0 {
                        let restitution =
                            (self.balls[i].restitution + self.balls[j].restitution) * 0.5;
                        let impulse = (1.0 + restitution) * dvn / total_mass;
                        self.balls[i].vel[0] -= impulse * self.balls[j].mass * nx;
                        self.balls[i].vel[1] -= impulse * self.balls[j].mass * ny;
                        self.balls[j].vel[0] += impulse * self.balls[i].mass * nx;
                        self.balls[j].vel[1] += impulse * self.balls[i].mass * ny;
                    }
                }
            }
        }

        // 3. Integrate position
        for b in &mut self.balls {
            b.pos[0] += b.vel[0] * dt;
            b.pos[1] += b.vel[1] * dt;
        }

        // 4. Ball-wall bouncing
        for b in &mut self.balls {
            let r = b.radius;
            let rest = b.restitution;

            if b.pos[0] - r < 0.0 {
                b.pos[0] = r;
                b.vel[0] = b.vel[0].abs() * rest;
            }
            if b.pos[0] + r > w {
                b.pos[0] = w - r;
                b.vel[0] = -b.vel[0].abs() * rest;
            }
            if b.pos[1] - r < 0.0 {
                b.pos[1] = r;
                b.vel[1] = b.vel[1].abs() * rest;
            }
            if b.pos[1] + r > h {
                b.pos[1] = h - r;
                b.vel[1] = -b.vel[1].abs() * rest;
            }
        }

        // 5. Ball-static-solid collision
        let cell_solid = 1u32; // CELL_SOLID
        for b in &mut self.balls {
            let r = b.radius;
            // Sample cells around the ball to find solid contacts
            let min_x = ((b.pos[0] - r - 1.0).max(0.0)) as i32;
            let max_x = ((b.pos[0] + r + 1.0).min(w - 1.0)) as i32;
            let min_y = ((b.pos[1] - r - 1.0).max(0.0)) as i32;
            let max_y = ((b.pos[1] + r + 1.0).min(h - 1.0)) as i32;

            let mut push_x = 0.0f32;
            let mut push_y = 0.0f32;
            let mut contact_count = 0u32;

            for cy in min_y..=max_y {
                for cx in min_x..=max_x {
                    let idx = cy as usize * width as usize + cx as usize;
                    if base_cell_types[idx] == cell_solid {
                        let dx = b.pos[0] - (cx as f32 + 0.5);
                        let dy = b.pos[1] - (cy as f32 + 0.5);
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist < r + 0.7 {
                            // Cell center within ball radius + half cell
                            if dist > 0.001 {
                                push_x += dx / dist;
                                push_y += dy / dist;
                            }
                            contact_count += 1;
                        }
                    }
                }
            }

            if contact_count > 0 {
                let len = (push_x * push_x + push_y * push_y).sqrt();
                if len > 0.001 {
                    let nx = push_x / len;
                    let ny = push_y / len;
                    // Push ball out
                    b.pos[0] += nx * 1.0;
                    b.pos[1] += ny * 1.0;
                    // Reflect velocity
                    let vn = b.vel[0] * nx + b.vel[1] * ny;
                    if vn < 0.0 {
                        b.vel[0] -= (1.0 + b.restitution) * vn * nx;
                        b.vel[1] -= (1.0 + b.restitution) * vn * ny;
                    }
                }
            }
        }

        // 6. Light velocity damping
        for b in &mut self.balls {
            b.vel[0] *= 0.9999;
            b.vel[1] *= 0.9999;
        }

        // 7. Detect explosions: balls with large delta_v shatter
        let mut shattered = vec![false; self.balls.len()];
        let mut explosions = Vec::new();

        for (i, b) in self.balls.iter().enumerate() {
            let dvx = b.vel[0] - vel_before[i][0];
            let dvy = b.vel[1] - vel_before[i][1];
            let delta_v = (dvx * dvx + dvy * dvy).sqrt();
            if delta_v > 1.5 {
                explosions.push(BallExplosion {
                    x: b.pos[0],
                    y: b.pos[1],
                    radius: b.radius,
                    color_id: b.color_id,
                    speed: delta_v,
                });
                shattered[i] = true;
            }
        }

        // Remove shattered balls
        let mut idx = 0;
        self.balls.retain(|_| {
            let keep = !shattered[idx];
            idx += 1;
            keep
        });

        explosions
    }

    /// Apply fluid forces to balls (two-way coupling: fluid → ball).
    /// `fluid_data` layout: [rho, ux, uy, curl] per cell, row-major.
    /// `steps_per_frame`: how many LBM sub-steps per frame (needed to convert
    /// LBM lattice velocity to ball-frame velocity).
    pub fn apply_fluid_forces(
        &mut self,
        fluid_data: &[f32],
        width: u32,
        height: u32,
        steps_per_frame: u32,
    ) {
        let w = width as f32;
        let h = height as f32;
        let drag_coeff = 0.015;
        let spf = steps_per_frame as f32;

        for b in &mut self.balls {
            let r = b.radius;
            // Sample just outside the solid rasterization region
            let sample_r = r + 2.0;
            let num_samples = 16;
            let mut avg_ux = 0.0f32;
            let mut avg_uy = 0.0f32;
            let mut count = 0u32;

            for i in 0..num_samples {
                let angle = (i as f32) / (num_samples as f32) * std::f32::consts::TAU;
                let sx = b.pos[0] + sample_r * angle.cos();
                let sy = b.pos[1] + sample_r * angle.sin();

                let ix = sx as i32;
                let iy = sy as i32;
                if ix >= 0 && ix < w as i32 && iy >= 0 && iy < h as i32 {
                    let idx = (iy as usize * width as usize + ix as usize) * 4;
                    if idx + 3 < fluid_data.len() {
                        let rho = fluid_data[idx];
                        // Skip solid/inlet/outlet sentinel values (negative rho)
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

                // Convert LBM velocity (cells/LBM-step) to cells/frame
                let fluid_vx = avg_ux * spf;
                let fluid_vy = avg_uy * spf;

                // Drag: proportional to relative velocity and cross-section
                let rel_vx = fluid_vx - b.vel[0];
                let rel_vy = fluid_vy - b.vel[1];
                let cross_section = 2.0 * r;
                let force_x = drag_coeff * rel_vx * cross_section;
                let force_y = drag_coeff * rel_vy * cross_section;

                b.vel[0] += force_x / b.mass;
                b.vel[1] += force_y / b.mass;
            }
        }
    }

    /// Rasterize ball footprints into cell_types as CELL_SOLID.
    pub fn rasterize(&self, cell_types: &mut [u32], width: u32, height: u32) {
        let cell_solid = 1u32;
        let w = width as i32;
        let h = height as i32;

        for b in &self.balls {
            let r = b.radius;
            let cx = b.pos[0] as i32;
            let cy = b.pos[1] as i32;
            let ri = r.ceil() as i32;
            let solid_r = r - 1.0; // slightly smaller than visual radius
            let solid_r2 = solid_r * solid_r;

            for dy in -ri..=ri {
                for dx in -ri..=ri {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px >= 0 && px < w && py >= 0 && py < h {
                        let fx = px as f32 + 0.5 - b.pos[0];
                        let fy = py as f32 + 0.5 - b.pos[1];
                        if fx * fx + fy * fy <= solid_r2 {
                            let idx = py as usize * width as usize + px as usize;
                            cell_types[idx] = cell_solid;
                        }
                    }
                }
            }
        }
    }

    /// Pack ball data for the GPU render shader.
    pub fn gpu_data(&self) -> Vec<BallGpuData> {
        self.balls
            .iter()
            .map(|b| BallGpuData {
                x: b.pos[0],
                y: b.pos[1],
                radius: b.radius,
                color_id: b.color_id as f32,
            })
            .collect()
    }
}
