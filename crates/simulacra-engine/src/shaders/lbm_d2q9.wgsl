// D2Q9 Lattice Boltzmann - TRT collision + Smagorinsky turbulence model
//
// TRT (Two Relaxation Time) with magic parameter Lambda=1/4.
// Smagorinsky subgrid model: locally increases effective viscosity where
// strain rates are high, preventing checkerboard instability at low
// viscosity while preserving detail elsewhere. Enables stable simulation
// at very high Reynolds numbers.
//
// Supports: solid bounce-back, inlet (equilibrium BC), outlet (open BC),
//           gravity via Guo forcing scheme, velocity clamping.
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
    gravity_x: f32,
    gravity_y: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(3) var<storage, read> cell_type: array<u32>;
@group(0) @binding(4) var<storage, read> cell_props: array<f32>;

const CELL_FLUID: u32 = 0u;
const CELL_SOLID: u32 = 1u;
const CELL_INLET: u32 = 2u;
const CELL_OUTLET: u32 = 3u;

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

fn cell_idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn f_idx(q: u32, x: u32, y: u32) -> u32 {
    return q * params.width * params.height + y * params.width + x;
}

fn feq(q: u32, rho: f32, ux: f32, uy: f32) -> f32 {
    let ex = f32(EX[q]);
    let ey = f32(EY[q]);
    let eu = ex * ux + ey * uy;
    let usq = ux * ux + uy * uy;
    return W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
}

@compute @workgroup_size(16, 16)
fn collide_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let ci = cell_idx(x, y);
    let ct = cell_type[ci];

    // --- Solid cells: full bounce-back ---
    if ct == CELL_SOLID {
        for (var q = 0u; q < 9u; q++) {
            let oq = OPP[q];
            f_out[f_idx(oq, x, y)] = f_in[f_idx(q, x, y)];
        }
        return;
    }

    // --- Inlet cells: equilibrium at prescribed velocity, rho=1.0 ---
    if ct == CELL_INLET {
        let inlet_ux = cell_props[ci * 2u + 0u];
        let inlet_uy = cell_props[ci * 2u + 1u];
        var f_post: array<f32, 9>;
        for (var q = 0u; q < 9u; q++) {
            f_post[q] = feq(q, 1.0, inlet_ux, inlet_uy);
        }
        // Stream to neighbors
        for (var q = 0u; q < 9u; q++) {
            let nx = i32(x) + EX[q];
            let ny = i32(y) + EY[q];
            if nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height) {
                f_out[f_idx(OPP[q], x, y)] = f_post[q];
            } else if cell_type[cell_idx(u32(nx), u32(ny))] == CELL_SOLID {
                f_out[f_idx(OPP[q], x, y)] = f_post[q];
            } else {
                f_out[f_idx(q, u32(nx), u32(ny))] = f_post[q];
            }
        }
        return;
    }

    // --- Outlet cells: pressure BC at rho=1.0 with free velocity ---
    // Holds reference density to absorb/release mass, balancing inlet injection.
    if ct == CELL_OUTLET {
        // Compute local velocity from incoming distributions
        var rho_local = 0.0;
        var ux_local = 0.0;
        var uy_local = 0.0;
        for (var q = 0u; q < 9u; q++) {
            let fq = f_in[f_idx(q, x, y)];
            rho_local += fq;
            ux_local += f32(EX[q]) * fq;
            uy_local += f32(EY[q]) * fq;
        }

        // Stability guard: if rho is bad, fall back to zero velocity.
        // !(a && b) reliably catches NaN since NaN comparisons return false.
        if !(rho_local > 0.3 && rho_local < 3.0) {
            ux_local = 0.0;
            uy_local = 0.0;
        } else {
            ux_local /= rho_local;
            uy_local /= rho_local;
            // Clamp velocity
            let spd2 = ux_local * ux_local + uy_local * uy_local;
            if spd2 > 0.3 * 0.3 {
                let s = 0.3 / sqrt(spd2);
                ux_local *= s;
                uy_local *= s;
            }
        }

        // Set equilibrium at reference density, free velocity
        var f_post: array<f32, 9>;
        for (var q = 0u; q < 9u; q++) {
            f_post[q] = feq(q, 1.0, ux_local, uy_local);
        }
        // Stream to neighbors
        for (var q = 0u; q < 9u; q++) {
            let nx = i32(x) + EX[q];
            let ny = i32(y) + EY[q];
            if nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height) {
                continue; // let distributions leave domain
            } else if cell_type[cell_idx(u32(nx), u32(ny))] == CELL_SOLID {
                f_out[f_idx(OPP[q], x, y)] = f_post[q];
            } else {
                f_out[f_idx(q, u32(nx), u32(ny))] = f_post[q];
            }
        }
        return;
    }

    // --- Read distributions ---
    var f: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        f[q] = f_in[f_idx(q, x, y)];
    }

    // --- Macroscopic quantities ---
    var rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    var ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]);
    var uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]);

    // --- Stability guard: reset to equilibrium if density is out of range or NaN ---
    // !(a && b) reliably catches NaN since NaN comparisons return false on GPU.
    if !(rho > 0.3 && rho < 3.0) {
        rho = 1.0;
        ux = 0.0;
        uy = 0.0;
        for (var q = 0u; q < 9u; q++) {
            f[q] = feq(q, 1.0, 0.0, 0.0);
        }
    } else {
        ux /= rho;
        uy /= rho;
    }

    // --- Guo velocity correction: u_phys = u_lattice + F/(2*rho) ---
    let gx = params.gravity_x;
    let gy = params.gravity_y;
    ux += gx * 0.5 / rho;
    uy += gy * 0.5 / rho;

    // --- Velocity clamping (stability guard catches blowups, so we can push higher) ---
    let speed_sq = ux * ux + uy * uy;
    let max_vel = 0.3;
    if speed_sq > max_vel * max_vel {
        let s = max_vel / sqrt(speed_sq);
        ux *= s;
        uy *= s;
    }

    // --- Pre-compute equilibrium ---
    var eq: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        eq[q] = feq(q, rho, ux, uy);
    }

    // --- Smagorinsky subgrid turbulence model ---
    // Compute non-equilibrium stress tensor to estimate local strain rate.
    // Where strain is high (constrictions, wakes), increase effective viscosity
    // to prevent checkerboard instability while preserving detail elsewhere.
    var pi_neq_xx = 0.0;
    var pi_neq_xy = 0.0;
    var pi_neq_yy = 0.0;
    for (var q = 0u; q < 9u; q++) {
        let f_neq = f[q] - eq[q];
        let ex = f32(EX[q]);
        let ey = f32(EY[q]);
        pi_neq_xx += f_neq * ex * ex;
        pi_neq_xy += f_neq * ex * ey;
        pi_neq_yy += f_neq * ey * ey;
    }
    let pi_mag = sqrt(pi_neq_xx * pi_neq_xx + 2.0 * pi_neq_xy * pi_neq_xy + pi_neq_yy * pi_neq_yy);

    // Smagorinsky constant (0.1 = moderate, higher = more dissipation)
    let C_s = 0.1;
    let tau_0 = 1.0 / params.omega;
    // Effective relaxation time: tau_eff = 0.5 * (tau_0 + sqrt(tau_0^2 + 18 * C_s^2 * |Pi| / rho))
    let tau_eff = 0.5 * (tau_0 + sqrt(tau_0 * tau_0 + 18.0 * C_s * C_s * pi_mag / rho));
    let omega_s = 1.0 / tau_eff;

    // --- TRT collision with Smagorinsky-adapted omega ---
    // omega_a via magic parameter Lambda = 1/4: (1/omega_s - 0.5)(1/omega_a - 0.5) = 1/4
    let omega_a = 2.0 - omega_s;

    var f_post: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        let oq = OPP[q];

        // Symmetric (even) and antisymmetric (odd) parts
        let f_plus  = 0.5 * (f[q] + f[oq]);
        let f_minus = 0.5 * (f[q] - f[oq]);
        let eq_plus  = 0.5 * (eq[q] + eq[oq]);
        let eq_minus = 0.5 * (eq[q] - eq[oq]);

        // TRT: relax symmetric part at omega_s, antisymmetric at omega_a
        f_post[q] = f[q]
            - omega_s * (f_plus - eq_plus)
            - omega_a * (f_minus - eq_minus);

        // Guo forcing term (uses omega_s for viscosity correction)
        let ex = f32(EX[q]);
        let ey = f32(EY[q]);
        let eu = ex * ux + ey * uy;
        let Fi = (1.0 - omega_s * 0.5) * W[q] * (
            (ex - ux + 3.0 * eu * ex) * gx +
            (ey - uy + 3.0 * eu * ey) * gy
        ) * 3.0;

        f_post[q] += Fi;
    }

    // --- Streaming (fluid cells only; solid/inlet/outlet already returned) ---
    for (var q = 0u; q < 9u; q++) {
        let nx = i32(x) + EX[q];
        let ny = i32(y) + EY[q];

        if nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height) {
            // Domain boundary: bounce-back
            f_out[f_idx(OPP[q], x, y)] = f_post[q];
        } else if cell_type[cell_idx(u32(nx), u32(ny))] == CELL_SOLID {
            // Bounce back from solid neighbor
            f_out[f_idx(OPP[q], x, y)] = f_post[q];
        } else {
            f_out[f_idx(q, u32(nx), u32(ny))] = f_post[q];
        }
    }
}

// Compute macroscopic output: density, velocity, vorticity
@compute @workgroup_size(16, 16)
fn compute_output(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let ci = cell_idx(x, y);
    let out_base = ci * 4u;
    let ct = cell_type[ci];

    // Solid cells: output zero velocity, mark with sentinel rho
    if ct == CELL_SOLID {
        f_out[out_base + 0u] = -1.0;
        f_out[out_base + 1u] = 0.0;
        f_out[out_base + 2u] = 0.0;
        f_out[out_base + 3u] = 0.0;
        return;
    }

    // Inlet cells: sentinel rho = -2.0
    if ct == CELL_INLET {
        f_out[out_base + 0u] = -2.0;
        f_out[out_base + 1u] = cell_props[ci * 2u + 0u];
        f_out[out_base + 2u] = cell_props[ci * 2u + 1u];
        f_out[out_base + 3u] = 0.0;
        return;
    }

    // Outlet cells: sentinel rho = -3.0
    if ct == CELL_OUTLET {
        f_out[out_base + 0u] = -3.0;
        f_out[out_base + 1u] = 0.0;
        f_out[out_base + 2u] = 0.0;
        f_out[out_base + 3u] = 0.0;
        return;
    }

    var f: array<f32, 9>;
    for (var q = 0u; q < 9u; q++) {
        f[q] = f_in[f_idx(q, x, y)];
    }

    let rho = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8];
    let ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho;
    let uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho;

    // Vorticity via finite differences
    var curl = 0.0;
    if x > 0u && x < params.width - 1u && y > 0u && y < params.height - 1u {
        let r_fluid = cell_type[cell_idx(x + 1u, y)] == CELL_FLUID;
        let l_fluid = cell_type[cell_idx(x - 1u, y)] == CELL_FLUID;
        let u_fluid = cell_type[cell_idx(x, y + 1u)] == CELL_FLUID;
        let d_fluid = cell_type[cell_idx(x, y - 1u)] == CELL_FLUID;

        if r_fluid && l_fluid && u_fluid && d_fluid {
            var f_r: array<f32, 9>;
            var f_l: array<f32, 9>;
            var f_u: array<f32, 9>;
            var f_d: array<f32, 9>;
            for (var q = 0u; q < 9u; q++) {
                f_r[q] = f_in[f_idx(q, x + 1u, y)];
                f_l[q] = f_in[f_idx(q, x - 1u, y)];
                f_u[q] = f_in[f_idx(q, x, y + 1u)];
                f_d[q] = f_in[f_idx(q, x, y - 1u)];
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
    }

    f_out[out_base + 0u] = rho;
    f_out[out_base + 1u] = ux;
    f_out[out_base + 2u] = uy;
    f_out[out_base + 3u] = curl;
}
