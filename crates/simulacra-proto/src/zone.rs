use simulacra_engine::dye::DyeField;
use simulacra_engine::gpu::GpuContext;
use simulacra_engine::lbm::{Lbm2D, CELL_FLUID, CELL_INLET, CELL_SOLID};

/// A simulation region: an LBM grid with dye field, cell state, and render resources.
/// Each zone and tunnel is a SimRegion.
pub struct SimRegion {
    pub lbm: Lbm2D,
    pub dye: DyeField,
    pub base_cell_types: Vec<u32>,
    pub base_cell_props: Vec<f32>,
    pub cell_types: Vec<u32>,
    pub cell_props: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub world_offset: [f32; 2],
    // CPU-side copy of LBM output for coupling readback
    pub fluid_data: Vec<f32>,
    pub fluid_readback_buffer: wgpu::Buffer,
    // Render resources for this region
    pub render_params_buffer: wgpu::Buffer,
    pub render_bind_group_0: wgpu::BindGroup,
    pub render_dye_bind_groups: [wgpu::BindGroup; 2],
}

impl SimRegion {
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        world_offset: [f32; 2],
        render_bgl_0: &wgpu::BindGroupLayout,
        render_bgl_2: &wgpu::BindGroupLayout,
    ) -> Self {
        let lbm = Lbm2D::new(gpu, width, height);
        let dye = DyeField::new(gpu, width, height, &lbm.output_buffer);
        let num_cells = (width * height) as usize;

        // Per-region render params buffer (48 bytes = 3 x vec4)
        let render_params_buffer = gpu.create_buffer(
            "region_render_params",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let render_bind_group_0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("region_bg_0"),
            layout: render_bgl_0,
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

        let render_dye_bind_groups = [
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("region_dye_bg_0"),
                layout: render_bgl_2,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dye.dye_buffer(0).as_entire_binding(),
                }],
            }),
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("region_dye_bg_1"),
                layout: render_bgl_2,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dye.dye_buffer(1).as_entire_binding(),
                }],
            }),
        ];

        let output_size = (num_cells * 4 * std::mem::size_of::<f32>()) as u64;
        let fluid_readback_buffer = gpu.create_buffer(
            "region_readback",
            output_size,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        Self {
            lbm,
            dye,
            base_cell_types: vec![CELL_FLUID; num_cells],
            base_cell_props: vec![0.0f32; num_cells * 2],
            cell_types: vec![CELL_FLUID; num_cells],
            cell_props: vec![0.0f32; num_cells * 2],
            width,
            height,
            world_offset,
            fluid_data: vec![0.0f32; num_cells * 4],
            fluid_readback_buffer,
            render_params_buffer,
            render_bind_group_0,
            render_dye_bind_groups,
        }
    }

    pub fn num_cells(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Set a rectangular region as solid walls.
    pub fn set_wall_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        for dy in 0..h {
            for dx in 0..w {
                let px = x + dx;
                let py = y + dy;
                if px < self.width && py < self.height {
                    let idx = (py * self.width + px) as usize;
                    self.base_cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    /// Set a circular region as solid walls.
    pub fn set_wall_circle(&mut self, cx: f32, cy: f32, radius: f32) {
        let r2 = radius * radius;
        let min_x = (cx - radius - 1.0).max(0.0) as u32;
        let max_x = (cx + radius + 1.0).min(self.width as f32 - 1.0) as u32;
        let min_y = (cy - radius - 1.0).max(0.0) as u32;
        let max_y = (cy + radius + 1.0).min(self.height as f32 - 1.0) as u32;
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                if dx * dx + dy * dy <= r2 {
                    let idx = (py * self.width + px) as usize;
                    self.base_cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    /// Set a row of inlet cells in base state.
    pub fn set_inlet_column(&mut self, x_start: u32, x_end: u32, y_start: u32, y_end: u32, vx: f32, vy: f32) {
        for y in y_start..y_end {
            for x in x_start..x_end {
                if x < self.width && y < self.height {
                    let idx = (y * self.width + x) as usize;
                    self.base_cell_types[idx] = CELL_INLET;
                    self.base_cell_props[idx * 2] = vx;
                    self.base_cell_props[idx * 2 + 1] = vy;
                }
            }
        }
    }

    /// Set a row of outlet cells in base state.
    pub fn set_outlet_column(&mut self, x_start: u32, x_end: u32, y_start: u32, y_end: u32) {
        for y in y_start..y_end {
            for x in x_start..x_end {
                if x < self.width && y < self.height {
                    let idx = (y * self.width + x) as usize;
                    self.base_cell_types[idx] = simulacra_engine::lbm::CELL_OUTLET;
                }
            }
        }
    }

    /// Sync cell_types/cell_props from base arrays.
    pub fn reset_cell_state(&mut self) {
        self.cell_types.copy_from_slice(&self.base_cell_types);
        self.cell_props.copy_from_slice(&self.base_cell_props);
    }
}

/// Which end of a tunnel connects to which zone.
pub struct TunnelEnd {
    pub zone_id: usize,
    /// Flat cell indices in the ZONE grid where we read outflow values
    pub zone_outlet_cells: Vec<usize>,
    /// Flat cell indices in the ZONE grid where we inject inflow
    pub zone_inlet_cells: Vec<usize>,
    /// Flat cell indices in the TUNNEL grid at this end (inlet from zone)
    pub tunnel_inlet_cells: Vec<usize>,
    /// Flat cell indices in the TUNNEL grid at this end (outlet to zone)
    pub tunnel_outlet_cells: Vec<usize>,
    /// Ship entry point in zone-local coords
    pub zone_entry_pos: [f32; 2],
    /// Ship entry point in tunnel-local coords
    pub tunnel_entry_pos: [f32; 2],
    /// Direction ship faces when entering
    pub entry_angle: f32,
}

pub struct Tunnel {
    pub region: SimRegion,
    pub end_a: TunnelEnd,
    pub end_b: TunnelEnd,
}

pub struct Zone {
    pub region: SimRegion,
    /// (tunnel_id, which_end: 0=A, 1=B)
    pub connections: Vec<(usize, u8)>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ActiveRegion {
    Zone(usize),
    Tunnel(usize),
}

pub struct ZoneManager {
    pub zones: Vec<Zone>,
    pub tunnels: Vec<Tunnel>,
    pub active: ActiveRegion,
}

/// Internal: coupling transfer to apply
struct CouplingTransfer {
    target_is_tunnel: bool,
    target_id: usize,
    cell_idx: usize,
    ux: f32,
    uy: f32,
}

fn sample_fluid(fluid_data: &[f32], cell_idx: usize) -> (f32, f32, f32) {
    let base = cell_idx * 4;
    if base + 2 < fluid_data.len() {
        (fluid_data[base], fluid_data[base + 1], fluid_data[base + 2])
    } else {
        (1.0, 0.0, 0.0)
    }
}

pub struct TransitionInfo {
    pub new_active: ActiveRegion,
    pub new_pos: [f32; 2],
}

impl ZoneManager {
    /// Check if ship has exited the active region. Returns transition info if so.
    pub fn check_ship_transition(&self, ship_pos: [f32; 2]) -> Option<TransitionInfo> {
        let (width, height) = match self.active {
            ActiveRegion::Zone(id) => (self.zones[id].region.width, self.zones[id].region.height),
            ActiveRegion::Tunnel(id) => (self.tunnels[id].region.width, self.tunnels[id].region.height),
        };
        let w = width as f32;
        let h = height as f32;

        // Ship hasn't exited — no transition
        if ship_pos[0] >= 0.0 && ship_pos[0] < w && ship_pos[1] >= 0.0 && ship_pos[1] < h {
            return None;
        }

        match self.active {
            ActiveRegion::Zone(zone_id) => {
                // Check each tunnel connected to this zone
                for &(tunnel_id, end) in &self.zones[zone_id].connections {
                    let tunnel = &self.tunnels[tunnel_id];
                    let tunnel_end = if end == 0 { &tunnel.end_a } else { &tunnel.end_b };

                    // Position mapping: offset from zone entry → offset in tunnel entry
                    let new_x = tunnel_end.tunnel_entry_pos[0] + (ship_pos[0] - tunnel_end.zone_entry_pos[0]);
                    let new_y = tunnel_end.tunnel_entry_pos[1] + (ship_pos[1] - tunnel_end.zone_entry_pos[1]);

                    // Check if mapped position is within tunnel bounds (with margin)
                    let tw = tunnel.region.width as f32;
                    let th = tunnel.region.height as f32;
                    if new_x >= -20.0 && new_x < tw + 20.0 && new_y >= -20.0 && new_y < th + 20.0 {
                        return Some(TransitionInfo {
                            new_active: ActiveRegion::Tunnel(tunnel_id),
                            new_pos: [new_x, new_y],
                        });
                    }
                }
                None
            }
            ActiveRegion::Tunnel(tunnel_id) => {
                let tunnel = &self.tunnels[tunnel_id];

                // Check end A (ship exits left side of tunnel → enter zone A)
                if ship_pos[0] < 0.0 {
                    let end = &tunnel.end_a;
                    let new_x = end.zone_entry_pos[0] + (ship_pos[0] - end.tunnel_entry_pos[0]);
                    let new_y = end.zone_entry_pos[1] + (ship_pos[1] - end.tunnel_entry_pos[1]);
                    return Some(TransitionInfo {
                        new_active: ActiveRegion::Zone(end.zone_id),
                        new_pos: [new_x, new_y],
                    });
                }

                // Check end B (ship exits right side of tunnel → enter zone B)
                let tw = tunnel.region.width as f32;
                if ship_pos[0] >= tw {
                    let end = &tunnel.end_b;
                    let new_x = end.zone_entry_pos[0] + (ship_pos[0] - end.tunnel_entry_pos[0]);
                    let new_y = end.zone_entry_pos[1] + (ship_pos[1] - end.tunnel_entry_pos[1]);
                    return Some(TransitionInfo {
                        new_active: ActiveRegion::Zone(end.zone_id),
                        new_pos: [new_x, new_y],
                    });
                }

                None
            }
        }
    }

    /// Get the active SimRegion.
    pub fn active_region(&self) -> &SimRegion {
        match self.active {
            ActiveRegion::Zone(id) => &self.zones[id].region,
            ActiveRegion::Tunnel(id) => &self.tunnels[id].region,
        }
    }

    pub fn active_region_mut(&mut self) -> &mut SimRegion {
        match self.active {
            ActiveRegion::Zone(id) => &mut self.zones[id].region,
            ActiveRegion::Tunnel(id) => &mut self.tunnels[id].region,
        }
    }

    /// Compute world bounding box from all regions' offset + size.
    pub fn world_bounds(&self) -> (f32, f32) {
        let mut max_x: f32 = 0.0;
        let mut max_y: f32 = 0.0;
        for zone in &self.zones {
            let r = &zone.region;
            max_x = max_x.max(r.world_offset[0] + r.width as f32);
            max_y = max_y.max(r.world_offset[1] + r.height as f32);
        }
        for tunnel in &self.tunnels {
            let r = &tunnel.region;
            max_x = max_x.max(r.world_offset[0] + r.width as f32);
            max_y = max_y.max(r.world_offset[1] + r.height as f32);
        }
        (max_x, max_y)
    }

    /// Reset all regions' cell state to base, then apply coupling.
    /// Call once per frame before LBM steps.
    pub fn coupling_step(&mut self) {
        // Reset all cell state to base
        for zone in &mut self.zones {
            zone.region.reset_cell_state();
        }
        for tunnel in &mut self.tunnels {
            tunnel.region.reset_cell_state();
        }

        // Gather coupling transfers (two-phase to avoid borrow issues)
        let mut transfers: Vec<CouplingTransfer> = Vec::new();

        for (tunnel_idx, tunnel) in self.tunnels.iter().enumerate() {
            // --- End A: Zone A <-> Tunnel ---
            let zone_a_data = &self.zones[tunnel.end_a.zone_id].region.fluid_data;
            let tunnel_data = &tunnel.region.fluid_data;

            // Zone A outlet → Tunnel inlet (at end A)
            for i in 0..tunnel.end_a.zone_outlet_cells.len().min(tunnel.end_a.tunnel_inlet_cells.len()) {
                let (_, ux, uy) = sample_fluid(zone_a_data, tunnel.end_a.zone_outlet_cells[i]);
                transfers.push(CouplingTransfer {
                    target_is_tunnel: true,
                    target_id: tunnel_idx,
                    cell_idx: tunnel.end_a.tunnel_inlet_cells[i],
                    ux,
                    uy,
                });
            }
            // Tunnel outlet (at end A) → Zone A inlet
            for i in 0..tunnel.end_a.tunnel_outlet_cells.len().min(tunnel.end_a.zone_inlet_cells.len()) {
                let (_, ux, uy) = sample_fluid(tunnel_data, tunnel.end_a.tunnel_outlet_cells[i]);
                transfers.push(CouplingTransfer {
                    target_is_tunnel: false,
                    target_id: tunnel.end_a.zone_id,
                    cell_idx: tunnel.end_a.zone_inlet_cells[i],
                    ux,
                    uy,
                });
            }

            // --- End B: Zone B <-> Tunnel ---
            let zone_b_data = &self.zones[tunnel.end_b.zone_id].region.fluid_data;
            let tunnel_data = &tunnel.region.fluid_data;

            // Zone B outlet → Tunnel inlet (at end B)
            for i in 0..tunnel.end_b.zone_outlet_cells.len().min(tunnel.end_b.tunnel_inlet_cells.len()) {
                let (_, ux, uy) = sample_fluid(zone_b_data, tunnel.end_b.zone_outlet_cells[i]);
                transfers.push(CouplingTransfer {
                    target_is_tunnel: true,
                    target_id: tunnel_idx,
                    cell_idx: tunnel.end_b.tunnel_inlet_cells[i],
                    ux,
                    uy,
                });
            }
            // Tunnel outlet (at end B) → Zone B inlet
            for i in 0..tunnel.end_b.tunnel_outlet_cells.len().min(tunnel.end_b.zone_inlet_cells.len()) {
                let (_, ux, uy) = sample_fluid(tunnel_data, tunnel.end_b.tunnel_outlet_cells[i]);
                transfers.push(CouplingTransfer {
                    target_is_tunnel: false,
                    target_id: tunnel.end_b.zone_id,
                    cell_idx: tunnel.end_b.zone_inlet_cells[i],
                    ux,
                    uy,
                });
            }
        }

        // Apply all transfers
        for t in &transfers {
            let region = if t.target_is_tunnel {
                &mut self.tunnels[t.target_id].region
            } else {
                &mut self.zones[t.target_id].region
            };
            region.cell_types[t.cell_idx] = CELL_INLET;
            region.cell_props[t.cell_idx * 2] = t.ux;
            region.cell_props[t.cell_idx * 2 + 1] = t.uy;
        }
    }

    /// Determine how many LBM steps each region should run this frame.
    /// Returns (zone_steps, tunnel_steps) indexed by id.
    /// All regions run at full speed since they're all visible simultaneously.
    pub fn schedule_steps(&self, base_speed: u32) -> (Vec<u32>, Vec<u32>) {
        let zone_steps = vec![base_speed; self.zones.len()];
        let tunnel_steps = vec![base_speed; self.tunnels.len()];
        (zone_steps, tunnel_steps)
    }

    /// Run LBM steps for all regions. Active region gets full speed,
    /// others get their scheduled step count.
    pub fn step_all(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        zone_steps: &[u32],
        tunnel_steps: &[u32],
    ) {
        // Upload and step each zone
        for (i, zone) in self.zones.iter_mut().enumerate() {
            if zone_steps[i] == 0 {
                continue;
            }
            zone.region.lbm.upload_cell_types(queue, &zone.region.cell_types);
            zone.region.lbm.upload_cell_props(queue, &zone.region.cell_props);
            zone.region.lbm.update_params(queue);
            for _ in 0..zone_steps[i] {
                zone.region.lbm.step_one(encoder);
            }
        }

        // Upload and step each tunnel
        for (i, tunnel) in self.tunnels.iter_mut().enumerate() {
            if tunnel_steps[i] == 0 {
                continue;
            }
            tunnel.region.lbm.upload_cell_types(queue, &tunnel.region.cell_types);
            tunnel.region.lbm.upload_cell_props(queue, &tunnel.region.cell_props);
            tunnel.region.lbm.update_params(queue);
            for _ in 0..tunnel_steps[i] {
                tunnel.region.lbm.step_one(encoder);
            }
        }
    }

    /// Compute LBM output fields for all regions that ran this frame.
    pub fn compute_outputs(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        zone_steps: &[u32],
        tunnel_steps: &[u32],
    ) {
        for (i, zone) in self.zones.iter().enumerate() {
            if zone_steps[i] > 0 {
                zone.region.lbm.compute_output(encoder);
            }
        }
        for (i, tunnel) in self.tunnels.iter().enumerate() {
            if tunnel_steps[i] > 0 {
                tunnel.region.lbm.compute_output(encoder);
            }
        }
    }

    /// Run dye advection for all regions that ran this frame.
    pub fn step_dye(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        zone_steps: &[u32],
        tunnel_steps: &[u32],
        sim_speed: u32,
    ) {
        for (i, zone) in self.zones.iter_mut().enumerate() {
            if zone_steps[i] > 0 {
                zone.region.dye.params.vel_scale = sim_speed as f32;
                zone.region.dye.upload_injections(queue, &[]);
                zone.region.dye.step(encoder);
            }
        }
        for (i, tunnel) in self.tunnels.iter_mut().enumerate() {
            if tunnel_steps[i] > 0 {
                tunnel.region.dye.params.vel_scale = sim_speed as f32;
                tunnel.region.dye.upload_injections(queue, &[]);
                tunnel.region.dye.step(encoder);
            }
        }
    }

    /// Read back fluid data from GPU for all regions that ran.
    /// Batches all copies into one submit + one sync.
    pub fn readback_fluid_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        zone_steps: &[u32],
        tunnel_steps: &[u32],
    ) {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zone_readback_copy"),
        });
        let mut regions_to_read: Vec<(bool, usize)> = Vec::new();

        for (i, zone) in self.zones.iter().enumerate() {
            if zone_steps[i] > 0 {
                let size = (zone.region.num_cells() * 4 * std::mem::size_of::<f32>()) as u64;
                enc.copy_buffer_to_buffer(
                    &zone.region.lbm.output_buffer,
                    0,
                    &zone.region.fluid_readback_buffer,
                    0,
                    size,
                );
                regions_to_read.push((false, i));
            }
        }
        for (i, tunnel) in self.tunnels.iter().enumerate() {
            if tunnel_steps[i] > 0 {
                let size = (tunnel.region.num_cells() * 4 * std::mem::size_of::<f32>()) as u64;
                enc.copy_buffer_to_buffer(
                    &tunnel.region.lbm.output_buffer,
                    0,
                    &tunnel.region.fluid_readback_buffer,
                    0,
                    size,
                );
                regions_to_read.push((true, i));
            }
        }

        if regions_to_read.is_empty() {
            return;
        }

        queue.submit([enc.finish()]);

        // Map all readback buffers
        for &(is_tunnel, id) in &regions_to_read {
            let buf = if is_tunnel {
                &self.tunnels[id].region.fluid_readback_buffer
            } else {
                &self.zones[id].region.fluid_readback_buffer
            };
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }

        device.poll(wgpu::PollType::Wait).unwrap();

        // Read data
        for &(is_tunnel, id) in &regions_to_read {
            let region = if is_tunnel {
                &mut self.tunnels[id].region
            } else {
                &mut self.zones[id].region
            };
            let slice = region.fluid_readback_buffer.slice(..);
            {
                let data = slice.get_mapped_range();
                let floats: &[f32] = bytemuck::cast_slice(&data);
                region.fluid_data.copy_from_slice(floats);
            }
            region.fluid_readback_buffer.unmap();
        }
    }
}

/// Create a 2-zone + 1 tunnel test level demonstrating coupling.
/// Zone 0: inlet on left, tunnel opening on right.
/// Zone 1: tunnel opening on left, outlet on right.
/// Tunnel: 256x64 connecting the two zones.
pub fn create_zone_test(
    gpu: &GpuContext,
    render_bgl_0: &wgpu::BindGroupLayout,
    render_bgl_2: &wgpu::BindGroupLayout,
) -> ZoneManager {
    let zw: u32 = 1024;
    let zh: u32 = 576;
    let tw: u32 = 256;
    let th: u32 = 64;

    // Tunnel opening position in zone coords (centered vertically)
    let opening_y_start = zh / 2 - th / 2; // 256
    let opening_y_end = zh / 2 + th / 2; // 320

    // --- Zone 0 ---
    let mut zone_0 = SimRegion::new(gpu, zw, zh, [0.0, 0.0], render_bgl_0, render_bgl_2);
    // Walls on all edges
    zone_0.set_wall_rect(0, 0, zw, 3); // top
    zone_0.set_wall_rect(0, zh - 3, zw, 3); // bottom
    zone_0.set_wall_rect(0, 0, 3, zh); // left
    // Right wall with opening
    zone_0.set_wall_rect(zw - 3, 0, 3, opening_y_start); // right above opening
    zone_0.set_wall_rect(zw - 3, opening_y_end, 3, zh - opening_y_end); // right below opening
    // Inlet on left edge
    zone_0.set_inlet_column(0, 3, 3, zh - 3, 0.15, 0.0);
    // Some obstacles
    zone_0.set_wall_circle(zw as f32 * 0.3, zh as f32 * 0.4, 18.0);
    zone_0.set_wall_circle(zw as f32 * 0.5, zh as f32 * 0.65, 14.0);
    zone_0.lbm.params.omega = 1.65;

    // --- Zone 1 ---
    let mut zone_1 = SimRegion::new(gpu, zw, zh, [1280.0, 0.0], render_bgl_0, render_bgl_2);
    // Walls on all edges
    zone_1.set_wall_rect(0, 0, zw, 3); // top
    zone_1.set_wall_rect(0, zh - 3, zw, 3); // bottom
    zone_1.set_wall_rect(zw - 3, 0, 3, zh); // right
    // Left wall with opening
    zone_1.set_wall_rect(0, 0, 3, opening_y_start); // left above opening
    zone_1.set_wall_rect(0, opening_y_end, 3, zh - opening_y_end); // left below opening
    // Outlet on right edge
    zone_1.set_outlet_column(zw - 3, zw, 3, zh - 3);
    // Some obstacles
    zone_1.set_wall_circle(zw as f32 * 0.6, zh as f32 * 0.35, 20.0);
    zone_1.set_wall_circle(zw as f32 * 0.4, zh as f32 * 0.7, 16.0);
    zone_1.lbm.params.omega = 1.65;

    // --- Tunnel ---
    let mut tunnel_region = SimRegion::new(gpu, tw, th, [1024.0, 256.0], render_bgl_0, render_bgl_2);
    // Walls on top and bottom only
    tunnel_region.set_wall_rect(0, 0, tw, 3); // top
    tunnel_region.set_wall_rect(0, th - 3, tw, 3); // bottom
    tunnel_region.lbm.params.omega = 1.65;

    // --- Coupling cell definitions ---
    // Zone 0 right edge ↔ Tunnel left end
    // Zone 0 outlet: 1 column at x=zw-6, y=opening range
    // Zone 0 inlet: 1 column at x=zw-4, y=opening range
    // Tunnel inlet (left): 1 column at x=3, y=3..th-3
    // Tunnel outlet (left): 1 column at x=5, y=3..th-3
    let mut end_a_zone_outlet = Vec::new();
    let mut end_a_zone_inlet = Vec::new();
    let mut end_a_tunnel_inlet = Vec::new();
    let mut end_a_tunnel_outlet = Vec::new();

    let coupling_height = (opening_y_end - opening_y_start).min(th - 6);
    for i in 0..coupling_height {
        let zy = opening_y_start + i;
        let ty = 3 + i;
        if zy < zh && ty < th - 3 {
            end_a_zone_outlet.push((zy * zw + (zw - 6)) as usize);
            end_a_zone_inlet.push((zy * zw + (zw - 4)) as usize);
            end_a_tunnel_inlet.push((ty * tw + 3) as usize);
            end_a_tunnel_outlet.push((ty * tw + 5) as usize);
        }
    }

    // Zone 1 left edge ↔ Tunnel right end
    let mut end_b_zone_outlet = Vec::new();
    let mut end_b_zone_inlet = Vec::new();
    let mut end_b_tunnel_inlet = Vec::new();
    let mut end_b_tunnel_outlet = Vec::new();

    for i in 0..coupling_height {
        let zy = opening_y_start + i;
        let ty = 3 + i;
        if zy < zh && ty < th - 3 {
            end_b_zone_outlet.push((zy * zw + 5) as usize);
            end_b_zone_inlet.push((zy * zw + 3) as usize);
            end_b_tunnel_inlet.push((ty * tw + (tw - 4)) as usize);
            end_b_tunnel_outlet.push((ty * tw + (tw - 6)) as usize);
        }
    }

    let end_a = TunnelEnd {
        zone_id: 0,
        zone_outlet_cells: end_a_zone_outlet,
        zone_inlet_cells: end_a_zone_inlet,
        tunnel_inlet_cells: end_a_tunnel_inlet,
        tunnel_outlet_cells: end_a_tunnel_outlet,
        // Entry anchors must satisfy: zone_entry + zone_offset == tunnel_entry + tunnel_offset
        // so world position is continuous across transitions.
        // Boundary at world x=1024: zone0 x=1024(=zw), tunnel x=0
        zone_entry_pos: [zw as f32, (opening_y_start + opening_y_end) as f32 / 2.0],
        tunnel_entry_pos: [0.0, th as f32 / 2.0],
        entry_angle: 0.0,
    };

    let end_b = TunnelEnd {
        zone_id: 1,
        zone_outlet_cells: end_b_zone_outlet,
        zone_inlet_cells: end_b_zone_inlet,
        tunnel_inlet_cells: end_b_tunnel_inlet,
        tunnel_outlet_cells: end_b_tunnel_outlet,
        // Boundary at world x=1280: tunnel x=256(=tw), zone1 x=0
        zone_entry_pos: [0.0, (opening_y_start + opening_y_end) as f32 / 2.0],
        tunnel_entry_pos: [tw as f32, th as f32 / 2.0],
        entry_angle: std::f32::consts::PI,
    };

    ZoneManager {
        zones: vec![
            Zone {
                region: zone_0,
                connections: vec![(0, 0)],
            },
            Zone {
                region: zone_1,
                connections: vec![(0, 1)],
            },
        ],
        tunnels: vec![Tunnel {
            region: tunnel_region,
            end_a,
            end_b,
        }],
        active: ActiveRegion::Zone(0),
    }
}
