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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;

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
    // t in [-1, 1], mapped to blue (negative) → white (zero) → red (positive)
    let s = clamp(t, -1.0, 1.0);
    if s < 0.0 {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.1, 0.2, 0.9), -s);
    } else {
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.9, 0.1, 0.1), s);
    }
}

// Color map: velocity magnitude (dark → bright, viridis-like)
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

    // Velocity magnitude visualization
    let speed = sqrt(ux * ux + uy * uy);
    let color = colormap_speed(speed * 15.0);

    // Mix in vorticity as a subtle overlay
    let curl_color = colormap_bwr(curl * 50.0);
    let final_color = mix(color, curl_color, 0.4);

    return vec4<f32>(final_color, 1.0);
}
