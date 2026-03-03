// Fullscreen quad rendering of LBM output data.
// Reads from a storage buffer containing [rho, ux, uy, curl] per cell.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
    cam_offset_x: f32,
    cam_offset_y: f32,
    cam_view_w: f32,
    cam_view_h: f32,
    world_offset_x: f32,
    world_offset_y: f32,
    _pad2: f32,
    _pad3: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct BallParams {
    num_balls: u32,
    sim_width: u32,
    sim_height: u32,
    _pad: u32,
};

struct BallData {
    x: f32,
    y: f32,
    radius: f32,
    color_id: f32,
};

struct ShipData {
    x: f32,
    y: f32,
    angle: f32,
    alive: f32,
    half_len: f32,
    half_width: f32,
    thrusting: f32,
    _pad: f32,
};

struct BulletParams {
    num_bullets: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct BulletData {
    x: f32,
    y: f32,
    radius: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;

@group(1) @binding(0) var<uniform> ball_params: BallParams;
@group(1) @binding(1) var<storage, read> balls: array<BallData>;

@group(2) @binding(0) var<storage, read> dye: array<f32>;

@group(3) @binding(0) var<uniform> ship: ShipData;
@group(3) @binding(1) var<uniform> bullet_params: BulletParams;
@group(3) @binding(2) var<storage, read> bullets: array<BulletData>;

// Fullscreen triangle trick: 3 vertices, no vertex buffer needed
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a triangle that covers the full screen
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

// Color map: cyan-black-magenta for signed values (curl/vorticity)
fn colormap_curl(t: f32) -> vec3<f32> {
    // t in [-1, 1]: cyan (negative spin) -> dark (zero) -> magenta (positive spin)
    let s = clamp(t, -1.0, 1.0);
    if s < 0.0 {
        return mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.8, 0.9), -s);
    } else {
        return mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.9, 0.1, 0.7), s);
    }
}

// Color map: velocity magnitude (dark -> bright, inferno-like)
fn colormap_speed(t: f32) -> vec3<f32> {
    let s = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.001, 0.000, 0.014); // near-black
    let c1 = vec3<f32>(0.42, 0.05, 0.52);    // deep purple
    let c2 = vec3<f32>(0.93, 0.25, 0.23);    // hot red-orange
    let c3 = vec3<f32>(0.99, 0.91, 0.15);    // bright yellow

    if s < 0.333 {
        return mix(c0, c1, s * 3.0);
    } else if s < 0.666 {
        return mix(c1, c2, (s - 0.333) * 3.0);
    } else {
        return mix(c2, c3, (s - 0.666) * 3.0);
    }
}

// Ball color palette: 6 distinct colors
fn ball_color(id: f32) -> vec3<f32> {
    let idx = u32(id) % 6u;
    switch idx {
        case 0u: { return vec3<f32>(0.90, 0.25, 0.20); } // red
        case 1u: { return vec3<f32>(0.20, 0.50, 0.90); } // blue
        case 2u: { return vec3<f32>(0.25, 0.80, 0.35); } // green
        case 3u: { return vec3<f32>(0.95, 0.75, 0.15); } // yellow
        case 4u: { return vec3<f32>(0.70, 0.30, 0.85); } // purple
        case 5u: { return vec3<f32>(0.95, 0.55, 0.15); } // orange
        default: { return vec3<f32>(0.8, 0.8, 0.8); }
    }
}

// Edge function for triangle test
fn edge_fn(v0: vec2<f32>, v1: vec2<f32>, p: vec2<f32>) -> f32 {
    return (v1.x - v0.x) * (p.y - v0.y) - (v1.y - v0.y) * (p.x - v0.x);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // World-space coordinate from camera
    let world_x = params.cam_offset_x + in.uv.x * params.cam_view_w;
    let world_y = params.cam_offset_y + in.uv.y * params.cam_view_h;

    // Local cell coordinate for this region
    let local_x = world_x - params.world_offset_x;
    let local_y = world_y - params.world_offset_y;

    // Out of bounds for this region: discard (let other draws fill it)
    if local_x < 0.0 || local_x >= f32(params.width) || local_y < 0.0 || local_y >= f32(params.height) {
        discard;
    }

    let x = clamp(u32(local_x), 0u, params.width - 1u);
    let y = clamp(u32(local_y), 0u, params.height - 1u);

    let field_base = (y * params.width + x) * 4u;
    let rho = field[field_base + 0u];
    let ux = field[field_base + 1u];
    let uy = field[field_base + 2u];
    let curl = field[field_base + 3u];

    // Solid cells: rho == -1.0 sentinel
    var fluid_color: vec3<f32>;
    if rho > -1.5 && rho < -0.5 {
        fluid_color = vec3<f32>(0.08, 0.08, 0.10);
    } else if rho > -2.5 && rho < -1.5 {
        // Inlet cells: rho == -2.0 sentinel
        let speed = sqrt(ux * ux + uy * uy);
        let raw_color = colormap_speed(speed * 5.0);
        let base_color = mix(raw_color, vec3<f32>(dot(raw_color, vec3<f32>(0.3, 0.6, 0.1))), 0.4);
        fluid_color = mix(base_color, vec3<f32>(0.15, 0.3, 0.85), 0.35);
    } else if rho > -3.5 && rho < -2.5 {
        // Outlet cells: rho == -3.0 sentinel
        let speed = sqrt(ux * ux + uy * uy);
        let raw_color = colormap_speed(speed * 5.0);
        let base_color = mix(raw_color, vec3<f32>(dot(raw_color, vec3<f32>(0.3, 0.6, 0.1))), 0.4);
        fluid_color = mix(base_color, vec3<f32>(0.15, 0.6, 0.25), 0.25);
    } else {
        // Velocity magnitude — very subtle dark base, letting dye dominate
        let speed = sqrt(ux * ux + uy * uy);
        let raw_color = colormap_speed(speed * 1.5);
        let luma = dot(raw_color, vec3<f32>(0.3, 0.6, 0.1));
        let color = mix(raw_color, vec3<f32>(luma), 0.6);
        // Vorticity: subtle additive tint where curl is strong
        let curl_strength = clamp(abs(curl) * 15.0, 0.0, 1.0);
        let curl_color = colormap_curl(curl * 15.0);
        fluid_color = color + curl_color * curl_strength * 0.10;
    }

    // Dye overlay: blend colored dye on top of base fluid
    let dye_base = (y * params.width + x) * 4u;
    let dr = dye[dye_base];
    let dg = dye[dye_base + 1u];
    let db = dye[dye_base + 2u];
    let dye_strength = max(dr, max(dg, db));
    let dye_alpha = clamp(dye_strength * 3.0, 0.0, 0.9);
    let norm_dye = vec3<f32>(dr, dg, db) / max(dye_strength, 0.001);
    fluid_color = mix(fluid_color, norm_dye, dye_alpha);

    // Use world-space coordinates for object rendering (ship, bullets, balls)
    let sim_x = world_x;
    let sim_y = world_y;
    var result = fluid_color;

    for (var i = 0u; i < ball_params.num_balls; i++) {
        let ball = balls[i];
        let dx = sim_x - ball.x;
        let dy = sim_y - ball.y;
        let dist = sqrt(dx * dx + dy * dy);

        // Early out: skip if far away
        if dist > ball.radius + 1.5 {
            continue;
        }

        let bc = ball_color(ball.color_id);

        // Antialiased edge
        let edge_alpha = 1.0 - smoothstep(ball.radius - 1.0, ball.radius + 0.5, dist);

        if edge_alpha > 0.001 {
            // 3D shading: highlight top-left, shadow bottom-right
            let nx = dx / ball.radius;
            let ny = dy / ball.radius;
            let light = clamp(-nx * 0.5 - ny * 0.7, -1.0, 1.0);
            let shaded = bc * (0.6 + 0.4 * light);

            // Specular highlight
            let spec = pow(clamp(light, 0.0, 1.0), 8.0) * 0.3;
            let ball_col = shaded + vec3<f32>(spec, spec, spec);

            // Dark outline at edge
            let outline = smoothstep(ball.radius - 1.5, ball.radius - 0.5, dist);
            let final_ball = mix(ball_col, vec3<f32>(0.05, 0.05, 0.08), outline * 0.6);

            result = mix(result, final_ball, edge_alpha);
        }
    }

    // Ship rendering
    if ship.alive > 0.5 {
        let cos_a = cos(ship.angle);
        let sin_a = sin(ship.angle);
        let fwd = vec2<f32>(cos_a, sin_a);
        let left = vec2<f32>(-sin_a, cos_a);
        let ship_pos = vec2<f32>(ship.x, ship.y);

        // Triangle vertices
        let nose = ship_pos + fwd * ship.half_len;
        let rear_left = ship_pos - fwd * ship.half_len + left * ship.half_width;
        let rear_right = ship_pos - fwd * ship.half_len - left * ship.half_width;

        let p = vec2<f32>(sim_x, sim_y);

        // Edge function test
        let e0 = edge_fn(nose, rear_left, p);
        let e1 = edge_fn(rear_left, rear_right, p);
        let e2 = edge_fn(rear_right, nose, p);

        let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);

        if inside {
            // Local coordinates for shading
            let local_fwd = dot(p - ship_pos, fwd);
            let local_side = dot(p - ship_pos, left);

            // Silver-blue metallic base
            let base_col = vec3<f32>(0.55, 0.62, 0.78);

            // Directional lighting: brighter toward nose, darker at rear
            let fwd_frac = local_fwd / ship.half_len; // -1 at rear, +1 at nose
            let light = clamp(fwd_frac * 0.3 + 0.7, 0.4, 1.0);

            // Side shading for 3D feel
            let side_frac = abs(local_side) / ship.half_width;
            let side_shade = 1.0 - side_frac * 0.2;

            var ship_color = base_col * light * side_shade;

            // Nose highlight
            if fwd_frac > 0.7 {
                let nose_glow = (fwd_frac - 0.7) / 0.3;
                ship_color = mix(ship_color, vec3<f32>(0.85, 0.9, 1.0), nose_glow * 0.4);
            }

            // Cockpit stripe
            if abs(local_side) < 1.5 && fwd_frac > 0.0 {
                ship_color = mix(ship_color, vec3<f32>(0.2, 0.35, 0.6), 0.3);
            }

            result = ship_color;
        }

        // Thruster glow (behind ship when thrusting)
        if ship.thrusting > 0.5 {
            let thruster_pos = ship_pos - fwd * (ship.half_len + 2.0);
            let dist_to_thruster = length(p - thruster_pos);
            let glow_radius = 8.0;
            let glow = 1.0 - smoothstep(0.0, glow_radius, dist_to_thruster);
            if glow > 0.001 {
                let inner = 1.0 - smoothstep(0.0, glow_radius * 0.3, dist_to_thruster);
                let glow_color = mix(vec3<f32>(1.0, 0.5, 0.1), vec3<f32>(1.0, 1.0, 0.8), inner);
                result = mix(result, glow_color, glow * 0.85);
            }
        }
    }

    // Bullet rendering
    for (var i = 0u; i < bullet_params.num_bullets; i++) {
        let b = bullets[i];
        let bx = sim_x - b.x;
        let by = sim_y - b.y;
        let bdist = sqrt(bx * bx + by * by);

        if bdist < b.radius + 2.0 {
            let bullet_alpha = 1.0 - smoothstep(b.radius - 0.3, b.radius + 0.5, bdist);
            if bullet_alpha > 0.001 {
                // Bright white-yellow core
                let core = 1.0 - smoothstep(0.0, b.radius * 0.5, bdist);
                let bullet_color = mix(vec3<f32>(1.0, 0.9, 0.5), vec3<f32>(1.0, 1.0, 1.0), core * 0.6);
                result = mix(result, bullet_color, bullet_alpha);
            }
        }
    }

    return vec4<f32>(result, 1.0);
}
