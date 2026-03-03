// Dye advection compute shader.
// Semi-Lagrangian advection of a passive RGBA scalar field using LBM velocity output.

struct DyeParams {
    width: u32,
    height: u32,
    decay: f32,
    diffusion: f32,
    vel_scale: f32,
    num_injections: u32,
    _pad0: f32,
    _pad1: f32,
};

struct DyeInjectPoint {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    radius: f32,
    strength: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: DyeParams;
@group(0) @binding(1) var<storage, read> lbm_output: array<f32>;
@group(0) @binding(2) var<storage, read> dye_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> dye_out: array<f32>;
@group(0) @binding(4) var<storage, read> injections: array<DyeInjectPoint>;

fn dye_idx(x: u32, y: u32) -> u32 {
    return (y * params.width + x) * 4u;
}

fn sample_dye(fx: f32, fy: f32) -> vec4<f32> {
    // Bilinear sample with clamping
    let x0 = clamp(i32(floor(fx)), 0, i32(params.width) - 1);
    let y0 = clamp(i32(floor(fy)), 0, i32(params.height) - 1);
    let x1 = clamp(x0 + 1, 0, i32(params.width) - 1);
    let y1 = clamp(y0 + 1, 0, i32(params.height) - 1);

    let sx = fx - floor(fx);
    let sy = fy - floor(fy);

    let i00 = dye_idx(u32(x0), u32(y0));
    let i10 = dye_idx(u32(x1), u32(y0));
    let i01 = dye_idx(u32(x0), u32(y1));
    let i11 = dye_idx(u32(x1), u32(y1));

    let c00 = vec4<f32>(dye_in[i00], dye_in[i00+1u], dye_in[i00+2u], dye_in[i00+3u]);
    let c10 = vec4<f32>(dye_in[i10], dye_in[i10+1u], dye_in[i10+2u], dye_in[i10+3u]);
    let c01 = vec4<f32>(dye_in[i01], dye_in[i01+1u], dye_in[i01+2u], dye_in[i01+3u]);
    let c11 = vec4<f32>(dye_in[i11], dye_in[i11+1u], dye_in[i11+2u], dye_in[i11+3u]);

    return mix(mix(c00, c10, sx), mix(c01, c11, sx), sy);
}

@compute @workgroup_size(16, 16)
fn advect_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let ci = y * params.width + x;
    let lbm_base = ci * 4u;

    let rho = lbm_output[lbm_base];
    let ux = lbm_output[lbm_base + 1u];
    let uy = lbm_output[lbm_base + 2u];

    let out_base = dye_idx(x, y);

    // Solid/inlet/outlet sentinel: zero dye on walls
    if rho < 0.0 {
        dye_out[out_base] = 0.0;
        dye_out[out_base + 1u] = 0.0;
        dye_out[out_base + 2u] = 0.0;
        dye_out[out_base + 3u] = 0.0;
        return;
    }

    // Semi-Lagrangian advection: trace back
    let src_x = f32(x) - ux * params.vel_scale;
    let src_y = f32(y) - uy * params.vel_scale;
    var advected = sample_dye(src_x, src_y);

    // Diffusion: blend with cardinal neighbors
    if params.diffusion > 0.0 {
        let xm = max(i32(x) - 1, 0);
        let xp = min(i32(x) + 1, i32(params.width) - 1);
        let ym = max(i32(y) - 1, 0);
        let yp = min(i32(y) + 1, i32(params.height) - 1);

        let il = dye_idx(u32(xm), y);
        let ir = dye_idx(u32(xp), y);
        let iu = dye_idx(x, u32(ym));
        let id = dye_idx(x, u32(yp));

        let left  = vec4<f32>(dye_in[il], dye_in[il+1u], dye_in[il+2u], dye_in[il+3u]);
        let right = vec4<f32>(dye_in[ir], dye_in[ir+1u], dye_in[ir+2u], dye_in[ir+3u]);
        let up    = vec4<f32>(dye_in[iu], dye_in[iu+1u], dye_in[iu+2u], dye_in[iu+3u]);
        let down  = vec4<f32>(dye_in[id], dye_in[id+1u], dye_in[id+2u], dye_in[id+3u]);

        let avg = (left + right + up + down) * 0.25;
        advected = mix(advected, avg, params.diffusion);
    }

    // Decay
    advected *= (1.0 - params.decay);

    // Injection
    for (var i = 0u; i < params.num_injections; i++) {
        let inj = injections[i];
        let dx = f32(x) - inj.x;
        let dy = f32(y) - inj.y;
        let dist = sqrt(dx * dx + dy * dy);
        if dist < inj.radius {
            let falloff = 1.0 - dist / inj.radius;
            let amount = inj.strength * falloff * falloff;
            advected += vec4<f32>(inj.r * amount, inj.g * amount, inj.b * amount, amount);
        }
    }

    // Clamp
    advected = clamp(advected, vec4<f32>(0.0), vec4<f32>(3.0));

    dye_out[out_base] = advected.x;
    dye_out[out_base + 1u] = advected.y;
    dye_out[out_base + 2u] = advected.z;
    dye_out[out_base + 3u] = advected.w;
}
