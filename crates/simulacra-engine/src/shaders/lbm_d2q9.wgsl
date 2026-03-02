// D2Q9 Lattice Boltzmann - BGK collision + streaming
//
// Lattice directions:
//   6  2  5
//    \ | /
//   3--0--1
//    / | \
//   7  4  8
//
// Distribution layout (Structure of Arrays):
//   f[q * width * height + y * width + x]

struct Params {
    width: u32,
    height: u32,
    omega: f32,
    lid_velocity: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;

// D2Q9 weights
const W = array<f32, 9>(
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);

// Lattice velocities
const EX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const EY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);

// Opposite directions for bounce-back
const OPP = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);

fn idx(q: u32, x: u32, y: u32) -> u32 {
    return q * params.width * params.height + y * params.width + x;
}

// Compute equilibrium distribution for direction q given macroscopic rho, ux, uy
fn feq(q: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let ex = f32(EX[q]);
    let ey = f32(EY[q]);
    let eu = ex * ux + ey * uy;        // e . u
    let usq = ux * ux + uy * uy;       // |u|^2
    return W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
}

@compute @workgroup_size(16, 16)
fn collide_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let n = params.width * params.height;
    let is_wall_left = x == 0u;
    let is_wall_right = x == params.width - 1u;
    let is_wall_bottom = y == 0u;
    let is_lid = y == params.height - 1u;

    let is_boundary = is_wall_left || is_wall_right || is_wall_bottom || is_lid;

    // --- Read distributions for this cell ---
    var f: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        f[q] = f_in[idx(q, x, y)];
    }

    // --- Compute macroscopic quantities ---
    var rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    var ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho;
    var uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho;

    // --- Lid boundary: impose velocity ---
    if is_lid {
        ux = params.lid_velocity;
        uy = 0.0;
        rho = (f[0] + f[1] + f[3] + 2.0 * (f[2] + f[5] + f[6])) / (1.0 + uy);
    }

    // --- BGK collision ---
    var f_post: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        let eq = feq(q, rho, ux, uy);
        f_post[q] = f[q] - params.omega * (f[q] - eq);
    }

    // --- Streaming with boundary handling ---
    for (var q = 0u; q < 9u; q++) {
        let nx = i32(x) + EX[q];
        let ny = i32(y) + EY[q];

        // Bounce-back: if the neighbor is outside the domain, reflect
        if nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height) {
            // Bounce back to current cell, opposite direction
            let oq = OPP[q];
            f_out[idx(oq, x, y)] = f_post[q];
        } else {
            // Stream to neighbor
            f_out[idx(q, u32(nx), u32(ny))] = f_post[q];
        }
    }
}

// Compute macroscopic output: density, velocity, vorticity (curl)
@compute @workgroup_size(16, 16)
fn compute_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    // Read distributions
    var f: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        f[q] = f_in[idx(q, x, y)];
    }

    let rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    let ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho;
    let uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho;

    // Approximate vorticity (curl of velocity) using finite differences
    var curl = 0.0;
    if x > 0u && x < params.width - 1u && y > 0u && y < params.height - 1u {
        // Read neighbor velocities for curl computation
        // duy/dx - dux/dy
        var f_r: array<f32, 9>;
        var f_l: array<f32, 9>;
        var f_u: array<f32, 9>;
        var f_d: array<f32, 9>;
        for (var q = 0u; q < 9u; q++) {
            f_r[q] = f_in[idx(q, x + 1u, y)];
            f_l[q] = f_in[idx(q, x - 1u, y)];
            f_u[q] = f_in[idx(q, x, y + 1u)];
            f_d[q] = f_in[idx(q, x, y - 1u)];
        }
        let rho_r = f_r[0]+f_r[1]+f_r[2]+f_r[3]+f_r[4]+f_r[5]+f_r[6]+f_r[7]+f_r[8];
        let rho_l = f_l[0]+f_l[1]+f_l[2]+f_l[3]+f_l[4]+f_l[5]+f_l[6]+f_l[7]+f_l[8];
        let rho_u = f_u[0]+f_u[1]+f_u[2]+f_u[3]+f_u[4]+f_u[5]+f_u[6]+f_u[7]+f_u[8];
        let rho_d = f_d[0]+f_d[1]+f_d[2]+f_d[3]+f_d[4]+f_d[5]+f_d[6]+f_d[7]+f_d[8];

        let uy_r = (f_r[2]+f_r[5]+f_r[6]-f_r[4]-f_r[7]-f_r[8]) / rho_r;
        let uy_l = (f_l[2]+f_l[5]+f_l[6]-f_l[4]-f_l[7]-f_l[8]) / rho_l;
        let ux_u = (f_u[1]+f_u[5]+f_u[8]-f_u[3]-f_u[6]-f_u[7]) / rho_u;
        let ux_d = (f_d[1]+f_d[5]+f_d[8]-f_d[3]-f_d[6]-f_d[7]) / rho_d;

        curl = (uy_r - uy_l) * 0.5 - (ux_u - ux_d) * 0.5;
    }

    // Write output: [rho, ux, uy, curl]
    let out_idx = (y * params.width + x) * 4u;
    f_out[out_idx + 0u] = rho;
    f_out[out_idx + 1u] = ux;
    f_out[out_idx + 2u] = uy;
    f_out[out_idx + 3u] = curl;
}
