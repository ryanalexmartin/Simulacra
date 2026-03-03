// Fullscreen quad rendering of LBM output data.
// Reads from a storage buffer containing [rho, ux, uy, curl] per cell.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;

@group(1) @binding(0) var<uniform> ball_params: BallParams;
@group(1) @binding(1) var<storage, read> balls: array<BallData>;

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

// Color map: blue-white-red for signed values (curl/vorticity)
fn colormap_bwr(t: f32) -> vec3<f32> {
    // t in [-1, 1], mapped to blue (negative) -> white (zero) -> red (positive)
    let s = clamp(t, -1.0, 1.0);
    if s < 0.0 {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.1, 0.2, 0.9), -s);
    } else {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.9, 0.1, 0.1), s);
    }
}

// Color map: velocity magnitude (dark -> bright, viridis-like)
fn colormap_speed(t: f32) -> vec3<f32> {
    let s = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.267, 0.004, 0.329);
    let c1 = vec3<f32>(0.282, 0.140, 0.458);
    let c2 = vec3<f32>(0.127, 0.566, 0.551);
    let c3 = vec3<f32>(0.741, 0.873, 0.150);

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = u32(in.uv.x * f32(params.width));
    let py = u32(in.uv.y * f32(params.height));
    let x = clamp(px, 0u, params.width - 1u);
    let y = clamp(py, 0u, params.height - 1u);

    let base = (y * params.width + x) * 4u;
    let rho = field[base + 0u];
    let ux = field[base + 1u];
    let uy = field[base + 2u];
    let curl = field[base + 3u];

    // Solid cells: rho == -1.0 sentinel
    var fluid_color: vec3<f32>;
    if rho > -1.5 && rho < -0.5 {
        fluid_color = vec3<f32>(0.15, 0.15, 0.18);
    } else if rho > -2.5 && rho < -1.5 {
        // Inlet cells: rho == -2.0 sentinel (blue tint)
        let speed = sqrt(ux * ux + uy * uy);
        let base_color = colormap_speed(speed * 15.0);
        fluid_color = mix(base_color, vec3<f32>(0.2, 0.4, 0.9), 0.5);
    } else if rho > -3.5 && rho < -2.5 {
        // Outlet cells: rho == -3.0 sentinel (green tint)
        let speed = sqrt(ux * ux + uy * uy);
        let base_color = colormap_speed(speed * 15.0);
        fluid_color = mix(base_color, vec3<f32>(0.2, 0.8, 0.3), 0.5);
    } else {
        // Velocity magnitude visualization
        let speed = sqrt(ux * ux + uy * uy);
        let color = colormap_speed(speed * 15.0);
        // Mix in vorticity as a subtle overlay
        let curl_color = colormap_bwr(curl * 50.0);
        fluid_color = mix(color, curl_color, 0.4);
    }

    // Ball rendering: overlay smooth circles
    let sim_x = in.uv.x * f32(ball_params.sim_width);
    let sim_y = in.uv.y * f32(ball_params.sim_height);
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

    return vec4<f32>(result, 1.0);
}
