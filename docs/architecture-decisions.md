# Simulacra Engine: Architecture Decision Analysis

This document explores the foundational technical decisions for the Simulacra engine.
Each section presents the options, their trade-offs, and an initial recommendation
to be debated and revised.

---

## 1. Language: Rust vs C++ vs Alternatives

### Rust

**Pros:**
- Memory safety without garbage collection. In a complex engine with three coupled
  physics domains sharing state, use-after-free and data races are not theoretical
  risks—they're near-certainties in C++. Rust eliminates these classes of bugs at
  compile time.
- Ownership model maps naturally to GPU buffer management, simulation state
  lifecycle, and resource cleanup.
- `wgpu` provides Rust-native cross-platform GPU compute (Vulkan/Metal/DX12/WebGPU).
- Cargo is a genuine strength: dependency management, testing, benchmarking,
  documentation generation, all built in. No CMake hell.
- Fearless concurrency. Multi-threaded physics (e.g., rigidbody solver on CPU while
  GPU runs fluid) is safe by construction.
- Algebraic types and pattern matching are excellent for state machines, variant
  component types, and error handling.
- Can target WebAssembly + WebGPU for browser demos.
- Growing game dev ecosystem: Bevy (ECS framework), wgpu, winit (windowing),
  glam/nalgebra (math), egui (debug UI).

**Cons:**
- The borrow checker fights you on graph-like structures. Physics constraint graphs,
  scene graphs, and particle neighbor lists all have shared/cyclic references that
  Rust makes awkward. Workarounds exist (arena allocation, index-based references,
  ECS) but add conceptual overhead.
- Smaller ecosystem of physics/graphics prior art. Most physics engine reference
  code and research implementations are in C/C++. Porting or rewriting is required.
- Compile times can be painful, especially with heavy generics. Incremental
  compilation helps but full rebuilds are slow.
- Fewer experienced Rust game developers to hire (if the project grows beyond solo).
- Some low-level GPU and SIMD intrinsics have less ergonomic access than C++,
  though `std::simd` (nightly) and `core::arch` are improving.

### C++

**Pros:**
- Decades of physics engine prior art. Bullet, PhysX, Havok, Box2D—all C++.
  Patterns, pitfalls, and solutions are well-documented.
- Most published physics research provides C++ reference implementations.
  SPH papers, LBM implementations, EM solvers—direct translation.
- Mature GPU compute: CUDA (NVIDIA), extensive Vulkan/DX12 compute shader
  tooling, profiling (Nsight, RenderDoc).
- Fine-grained control over memory layout, SIMD intrinsics, cache-line alignment.
  When you need to squeeze every cycle out of an inner loop, C++ lets you.
- Template metaprogramming enables compile-time dispatch for physics variants
  (e.g., different collision types, boundary conditions).
- Massive existing library ecosystem: Eigen (linear algebra), ImGui (debug UI),
  stb (image loading), etc.

**Cons:**
- Memory safety is entirely on you. In an engine with three coupled domains, shared
  buffers, and multi-threaded access, this is a significant and ongoing risk.
  Subtle memory bugs can manifest as non-deterministic simulation divergence
  that's nearly impossible to diagnose.
- Build system complexity (CMake, vcpkg/conan, platform-specific flags) is a
  constant tax on development velocity.
- No standard package manager. Dependency management is manual and fragile.
- Undefined behavior lurks everywhere. Signed overflow, strict aliasing, object
  lifetime—the compiler assumes you don't violate rules it doesn't enforce.
- Thread safety is manual. Every shared mutable access needs explicit locking
  or atomic operations, with no compiler verification.
- Technical debt accumulates faster due to the language's permissiveness.

### Zig

**Pros:**
- C-level performance with better ergonomics. No hidden control flow, comptime
  evaluation, clean syntax.
- Seamless C interop (can include C headers directly). Easy to leverage existing
  C physics libraries.
- Fast compilation, good error messages.
- No GC, manual memory management with allocator-awareness built into the language.

**Cons:**
- Very young ecosystem. Few game dev libraries, limited GPU compute story.
- Language still pre-1.0, breaking changes possible.
- Tiny community—you're on your own for most problems.
- No Rust-style safety guarantees. Memory bugs are still possible.

### Analysis

Rust is the right choice for a 2026 greenfield engine. The core argument: when you
have three coupled physics domains with shared mutable state, GPU buffer management,
and multi-threaded CPU work, the cost of memory/concurrency bugs is enormous—not
crashes, but *silent simulation corruption* that's nearly impossible to debug. Rust
prevents this by construction.

The borrow checker pain for graph structures is real but solvable. The standard
patterns are:
- Arena allocation (typed-arena, bumpalo) for simulation particles/cells
- Index-based references (generational indices) instead of pointers for graphs
- ECS architecture for entity management (specs, bevy_ecs)

The ecosystem gap is narrowing fast. wgpu is production-quality. glam is as fast as
GLM. Bevy's ECS is battle-tested in shipped games.

**Recommendation: Rust**

---

## 2. Fluid Simulation Method

### SPH (Smoothed Particle Hydrodynamics)

**How it works:** Fluid is represented as particles. Each particle carries mass,
velocity, and density. Physical quantities are computed by smoothing over nearby
particles using a kernel function. Forces (pressure, viscosity) are computed from
these smoothed quantities.

**Pros:**
- Naturally handles free surfaces, splashing, fragmentation, and merging.
  Particles just go where the fluid goes.
- Lagrangian (particles move with the fluid)—intuitive and easy to visualize.
- Couples naturally with rigidbody (both are discrete bodies/particles).
- Handles variable-density flows and multi-phase (oil/water) interactions.
- Adaptive resolution by varying particle density where detail is needed.

**Cons:**
- Incompressibility is hard to enforce. Standard SPH is weakly compressible,
  leading to density fluctuations and visual "sloshing" artifacts. IISPH and
  DFSPH fix this but add cost.
- Tensile instability: particles can clump together, creating voids in the fluid.
- Neighbor search is expensive: O(N log N) with spatial hashing, and it must
  happen every timestep.
- Boundary handling (fluid-solid interfaces) is tricky. Requires ghost/boundary
  particles that must be carefully managed.
- Noisy pressure field makes coupling with EM (where you need smooth gradients)
  problematic.
- Needs many particles (100k+) for visually smooth results in 3D.

### PBF (Position Based Fluids)

**How it works:** Extension of Position Based Dynamics to fluids. Particles are
moved to satisfy density constraints iteratively—like springs pulling particles
to maintain target density.

**Pros:**
- Very stable, even with large timesteps. Won't explode like SPH can.
- Built on the PBD framework, which also handles cloth, soft bodies, and ropes.
  Unified solver for multiple deformable types.
- Good incompressibility via iterative constraint projection.
- Fewer particles needed than SPH for similar visual quality.
- Natural integration with PBD-based rigidbody solver.
- Simpler to implement than SPH variants (IISPH, DFSPH).

**Cons:**
- NOT physically accurate. PBF is a visual approximation, not a simulation of
  real fluid dynamics. Constraint-based, not force-based.
- Energy is artificially dissipated. Fluid "dies down" faster than it should.
  Vorticity confinement is needed to inject energy back.
- No physically meaningful pressure. This is critical for our use case—coupling
  with EM requires pressure gradients that have physical units and magnitudes.
- Less suited for thermal coupling (no natural concept of fluid temperature
  affecting dynamics in a physically consistent way).
- The "fake physics" issue directly conflicts with our north star: "if the
  simulation feels fake, no amount of game design saves it."

### Eulerian Grid-Based (MAC Grid / Marker-and-Cell)

**How it works:** Fluid velocities are stored on a fixed grid. Navier-Stokes
equations are solved directly: advection moves quantities through the grid,
pressure projection enforces incompressibility.

**Pros:**
- Natural for incompressible flow. Pressure Poisson solve enforces divergence-free
  velocity fields cleanly.
- Easy to compute derivatives (gradients, curl, divergence) on a regular grid—
  just finite differences. Critical for EM coupling (curl of B, gradient of phi).
- **Natural coupling with EM**: both fluid and EM use grid-based PDEs. They can
  share the same spatial grid. Pressure solve and electric potential solve use
  the same type of Poisson solver.
- Well-understood numerical methods. Decades of CFD literature.
- GPU-friendly: regular memory access patterns, no dynamic neighbor search.
- Vorticity is preserved well with good advection schemes.

**Cons:**
- Fixed resolution. Can't add detail where it's needed without refining the
  entire grid (or implementing complex adaptive mesh refinement).
- Numerical dissipation: the advection step smears sharp features. Semi-Lagrangian
  advection is stable but diffusive. Higher-order schemes (BFECC, MacCormack)
  help but add cost.
- Free surfaces and splashing are hard. Level-set or volume-of-fluid methods
  track the surface but can't naturally represent thin sheets, droplets, or spray.
- Memory intensive: storing velocity + pressure + auxiliary fields on a 256^3
  grid is already ~200MB+.
- Fluid only exists where the grid exists. Large open domains are wasteful.

### FLIP/PIC (Fluid Implicit Particle)

**How it works:** Hybrid approach. Particles carry velocity and move through a
background grid. Each timestep: transfer particle velocities to grid, solve
pressure on grid, transfer velocity corrections back to particles, advect
particles.

**Pros:**
- Best of both worlds: particles give Lagrangian detail (splashing, thin features,
  no dissipation), grid gives Eulerian stability (clean pressure solve,
  incompressibility).
- Low numerical dissipation. Particles preserve velocity detail that the grid
  would smear. This matters for visual quality and physical accuracy.
- Industry standard in VFX for a reason (Houdini FLIP, Mantaflow, SPlisHSPlasH).
- Natural free-surface handling: where there are particles, there's fluid.
  Spray, foam, and bubbles emerge naturally.
- The grid component couples naturally with an EM grid solver.
- APIC (Affine Particle-In-Cell) variant preserves angular momentum and reduces
  noise significantly.

**Cons:**
- More complex implementation. Particle-to-grid (P2G) and grid-to-particle (G2P)
  transfers must be carefully implemented to conserve momentum.
- Pure FLIP is noisy (particles get jittery). Pure PIC is too dissipative. Need
  to blend, typically 95-99% FLIP + 1-5% PIC. The blend ratio affects visual
  quality and requires tuning.
- Still limited by grid resolution for the pressure solve. Fine fluid features
  beyond grid resolution aren't captured in the pressure.
- Particle-grid transfers have overhead, especially on GPU (scatter operations,
  atomics for P2G).
- More memory than pure grid (need both grid storage AND per-particle storage).
- APIC/FLIP GPU implementations are less well-documented than CPU versions.

### LBM (Lattice Boltzmann Method)

**How it works:** Instead of solving Navier-Stokes directly, LBM simulates fluid
at a mesoscopic level using probability distribution functions on a lattice. Each
cell stores a set of distributions (D3Q19: 19 values per cell in 3D) representing
the probability of finding a fluid "packet" moving in each lattice direction.
Each timestep: collision (local relaxation toward equilibrium) then streaming
(shift distributions to neighbor cells).

**Pros:**
- **Embarrassingly parallel.** The stream+collide loop has perfect data locality.
  Each cell's collision is independent. Streaming is a simple shift. This maps
  perfectly to GPU compute shaders.
- Simple algorithm. The core loop is ~50 lines of code. Collision is a local
  operation (no global solve), streaming is a copy.
- Complex geometries are natural. Solid boundaries are just cells with bounce-back
  rules. No meshing required.
- **Extensibility is the killer feature.** Adding new physics means modifying the
  collision operator or adding force terms:
  - Body forces (gravity, EM Lorentz force): add force term to collision
  - Thermal convection: add a second distribution for temperature (double DDF)
  - Multi-phase/multi-component: Shan-Chen or free-energy models
  - Each extension is modular and localized
- **EM coupling is natural.** Lorentz force on conductive fluid is just a body
  force term. Joule heating feeds into thermal LBM. MHD-LBM is a published,
  validated approach.
- No pressure Poisson solve needed. Pressure is a local function of density.
  This avoids the most expensive step in grid-based Navier-Stokes.
- Multi-relaxation-time (MRT) and cumulant variants improve stability significantly.

**Cons:**
- Memory intensive. D3Q19 with single precision: 19 × 4 bytes = 76 bytes per
  cell, times two (double buffering) = 152 bytes/cell. A 256^3 grid is 2.5 GB.
  D3Q27 is worse. This is the primary practical limitation.
- Fixed grid resolution. Adaptive LBM exists but is complex and partially
  breaks the parallelism advantage.
- Compressible by nature. LBM recovers the weakly compressible Navier-Stokes
  equations. At high Mach numbers (fast flow relative to lattice speed of sound),
  compressibility artifacts appear. Mitigated by keeping flow velocities below
  ~0.1 lattice units.
- Free surface handling requires additional tracking (e.g., volume-of-fluid
  style flags per cell). Less natural than particle methods.
- Stability at high Reynolds numbers (turbulent flow) requires careful choice
  of collision operator (MRT, cumulant, or regularized).
- Less intuitive than Navier-Stokes. Hard to reason about in terms of familiar
  fluid quantities until you understand the connection.
- Boundary conditions for complex moving objects (rigidbody coupling) require
  interpolated bounce-back schemes that are more involved than simple bounce-back.

### Analysis

**PBF is out.** Its position-based, non-physical approach conflicts directly with
the north star. When we say "if the simulation feels real," we mean it—PBF fakes
physics, and that fakeness bleeds through in how fluid interacts with EM fields
and thermal gradients. There's no physical pressure to couple with.

**SPH is possible but painful.** It handles free surfaces beautifully but makes EM
coupling difficult (noisy pressure, no natural grid for field computation). We'd
need IISPH or DFSPH for incompressibility, plus a separate grid for EM, plus
interpolation between them.

**The real contest is FLIP/PIC vs LBM.**

| Criterion                    | FLIP/PIC    | LBM         |
|------------------------------|-------------|-------------|
| Free surface / splashing     | Excellent   | Moderate    |
| GPU parallelism              | Good        | Excellent   |
| EM coupling (shared grid)    | Good        | Excellent   |
| Thermal coupling             | Moderate    | Excellent   |
| Memory efficiency            | Moderate    | Poor        |
| Extensibility (new physics)  | Moderate    | Excellent   |
| Physical accuracy            | Good        | Good        |
| Implementation complexity    | High        | Moderate    |
| Algorithmic simplicity       | Low         | High        |
| Visual quality               | Excellent   | Good        |

LBM wins on the criteria that matter most for this engine: EM coupling, thermal
coupling, extensibility, GPU performance, and algorithmic simplicity. Its main
weakness—free surfaces—can be supplemented with a particle spray/splash layer.

FLIP/PIC wins on visual quality and free-surface handling. It's the safer, more
proven choice for "fluid that looks amazing."

**Recommendation: LBM as the primary fluid solver**, with particle-based spray/foam
tracking at free surfaces for visual detail. The extensibility argument is decisive:
adding MHD, thermal convection, and multi-phase behavior to LBM is modular and
well-studied. Adding these to FLIP/PIC would require significantly more custom work.

The memory cost of LBM is a real concern. Mitigation strategies:
- Sparse grid (only allocate cells containing fluid or near boundaries)
- Reduced lattice (D3Q19 instead of D3Q27, or even D3Q15 where accuracy permits)
- Compressed distribution storage (exploiting symmetries)
- GPU VRAM is growing fast—8-24 GB is common in 2026

---

## 3. Electromagnetic Solver

### Full Maxwell's Equations (FDTD)

**How it works:** Finite-Difference Time-Domain. Discretize Maxwell's curl
equations on a staggered Yee grid. E and B fields are updated alternately.
Supports full wave propagation.

**Pros:**
- Complete electromagnetic physics. Wave propagation, radiation, diffraction,
  interference, resonance, waveguides.
- Well-understood algorithm, extensive literature.
- GPU parallelizable (similar regularity to LBM).
- Time-domain naturally handles transient phenomena.

**Cons:**
- **Computationally prohibitive for our use case.** The grid must resolve the
  shortest wavelength. For 60 Hz AC current, the wavelength is ~5,000 km. Even
  for kHz-range phenomena, wavelengths are hundreds of meters. Our game world
  is meters across. The grid resolution needed for mechanical detail (cm-scale)
  would model EM waves at GHz frequencies, wasting enormous computation on
  physics irrelevant to gameplay.
- Timestep is constrained by the Courant condition tied to the speed of light.
  For a 1cm grid cell: dt < 1cm / (3×10^8 m/s) ≈ 33 picoseconds. At 33ps per
  step, reaching 1 second of simulation requires ~30 billion steps. **Completely
  impractical.**
- The gameplay-relevant EM phenomena (eddy currents, induction, MHD, static
  fields) are all quasi-static. Full wave propagation adds nothing to gameplay
  while adding enormous cost.

**Verdict: Eliminated.** The physics we care about doesn't require wave propagation,
and the computational cost is prohibitive by many orders of magnitude.

### Quasi-Static Field Solver

**How it works:** Assume fields establish instantaneously (valid when the system
size is much smaller than the EM wavelength, which it always is for our game world
at relevant frequencies). Solve:
- Electrostatics: Poisson equation ∇²φ = -ρ/ε for electric potential
- Magnetostatics: ∇²A = -μJ for magnetic vector potential
- Eddy currents: Diffusion equation ∂B/∂t = (1/μσ)∇²B for magnetic field
  in conductors

**Pros:**
- Sufficient for ALL gameplay-relevant phenomena: eddy currents, electromagnetic
  induction, Lorentz forces, MHD, Joule heating, electromagnetic braking.
- The governing equations (Poisson, diffusion) are structurally identical to
  equations in fluid simulation. Shared solver infrastructure is possible.
- Grid-based, so couples naturally with LBM fluid solver (same grid!).
- Orders of magnitude cheaper than FDTD.
- Physically accurate within the quasi-static approximation (which is valid for
  our scenario—game world is meters, not kilometers).

**Cons:**
- No wave propagation. No radio, radar, microwave heating, antenna design, or
  resonant cavity effects. These could be cool gameplay mechanics that we're
  giving up.
- Still requires solving PDEs on a grid, which has non-trivial cost (though
  much less than FDTD).
- Eddy current computation in arbitrarily-shaped conductors requires solving the
  vector diffusion equation, which is more complex than scalar Poisson.
- Material properties (conductivity, permeability) must be defined per-cell,
  adding to memory requirements.

### Circuit-Level Simulation (Modified Nodal Analysis)

**How it works:** Represent the electrical system as a graph of nodes and
components. Each component (resistor, capacitor, inductor, voltage/current source)
contributes to a system of linear equations. Solve the matrix equation for node
voltages and branch currents.

**Pros:**
- Extremely fast. Sparse matrix solve is O(N) to O(N^1.5) where N is the number
  of nodes. Can handle thousands of components in microseconds.
- **No artificial propagation delay.** Unlike Minecraft redstone (which ticks at
  10Hz and propagates one block per tick, making circuits sluggish and requiring
  "tick manipulation" hacks), MNA solves the entire circuit simultaneously. A
  1000-node circuit reaches steady state in one solve. Signal doesn't "travel"
  through wires—it's computed globally, as real circuit theory dictates (Kirchhoff's
  laws are instantaneous). This means player-built circuits feel responsive and
  correct, not like they're fighting an update clock.
- Natural abstraction for discrete components that players place: wires, resistors,
  batteries, capacitors, switches, LEDs.
- Players think in terms of circuits. This matches their mental model.
- Well-understood algorithms (SPICE has existed since 1973).
- Time-stepping for reactive components (capacitors, inductors) is straightforward
  with trapezoidal integration. Transient behavior (RC charging curves, LC
  oscillations) emerges naturally from the math, not from artificial tick delays.
- Because we're not an open-world game with infinite terrain, the circuit graph
  is bounded and predictable. We can budget for thousands of nodes comfortably,
  far beyond what players will realistically build.

**Cons:**
- No spatial fields. A wire has no surrounding magnetic field. A current has no
  Lorentz force on nearby conductors. This eliminates all the interesting physics:
  no eddy currents, no MHD, no induction at a distance, no electromagnetic braking.
- Lumped-element assumption: all physics is concentrated at discrete points. Can't
  model distributed effects like skin effect, proximity effect, or field fringing.
- Alone, this is fast and correct but limited in scope. The spatial field solver
  is what elevates this beyond "fancy redstone" into real EM physics.

### Hybrid: Circuit + Quasi-Static Fields

**How it works:** Two coupled subsystems:
1. Circuit solver handles discrete components (wires, batteries, resistors, etc.)
   using Modified Nodal Analysis.
2. Quasi-static field solver handles spatial EM effects on a grid: magnetic
   fields from current-carrying conductors, eddy currents in bulk conductors,
   Lorentz forces in conductive fluids.

The two are coupled: circuit currents generate fields (Biot-Savart → grid),
and field-induced EMFs feed back into the circuit (Faraday's law → voltage sources
in the circuit).

**Pros:**
- Best of both worlds. Discrete components are fast (circuit solver). Spatial
  phenomena are physically accurate (field solver). Each handles what it's best at.
- Players can build circuits (intuitive) AND observe spatial field effects
  (impressive, educational).
- Performance is scalable: circuit solver runs only where there are components,
  field solver runs only where there are conductors/fields.
- The field solver grid can be the same grid as the fluid solver, enabling
  trivial MHD coupling.
- Can progressively enhance: start with circuit only, add field solver for
  specific regions/scenarios.

**Cons:**
- Coupling interface is non-trivial. Where does "circuit" end and "field" begin?
  A long wire could be a circuit element (lumped resistance + inductance) or a
  spatially-resolved conductor on the field grid. Need clear rules.
- Two different solvers to implement and maintain.
- Edge cases in coupling: what happens when a conductive fluid bridges two
  circuit nodes? The fluid becomes a spatially-resolved resistor that the circuit
  solver must account for.
- More complex than either approach alone.

### Analysis

FDTD is eliminated on computational grounds—it's not close.

The real question is: **do we need spatial fields, or are circuits enough?**

The answer comes from the north star and the game mechanics we identified:

| Mechanic                     | Circuit Only | Quasi-Static Fields |
|------------------------------|-------------|---------------------|
| Wires, batteries, switches   | ✓           | ✓                   |
| Eddy current braking         | ✗           | ✓                   |
| MHD (pump with no moving parts)| ✗         | ✓                   |
| Electromagnetic induction    | Lumped only | Full spatial         |
| Joule heating → convection   | Point source| Spatially resolved   |
| Conductive fluid in circuit  | ✗           | ✓                   |
| Electromagnet spatial force  | ✗           | ✓                   |

Every interesting cross-domain interaction requires spatial fields. Circuit-only
gets us Minecraft redstone. The hybrid approach gets us real physics.

**Recommendation: Hybrid circuit + quasi-static field solver.** The circuit solver
handles player-placed discrete components (cheap, fast, intuitive). The quasi-static
field solver handles spatial EM on the grid (accurate, couples with fluid). The
coupling interface needs careful design but is tractable.

The quasi-static field equations to solve:
- **Electric potential:** ∇·(σ∇φ) = -∂ρ/∂t (current continuity)
- **Magnetic field from currents:** ∇²A = -μ₀J (vector Poisson, or Biot-Savart
  for thin wires)
- **Eddy currents:** ∂B/∂t = (1/μσ)∇²B (magnetic diffusion in conductors)
- **Lorentz force:** F = J × B (feeds into fluid and rigidbody)
- **Joule heating:** Q = J·E = σ|E|² (feeds into thermal)

These are all grid-based PDEs that share solver infrastructure with the fluid
pressure solve.

---

## 4. Spatial Representation: Unified Grid vs Separate

### Option A: Single Unified Grid

All domains (fluid, EM, thermal) share one regular grid at one resolution.

**Pros:**
- Coupling is trivial. Fluid velocity, pressure, EM fields, temperature—all at
  the same grid points. No interpolation. F_lorentz = J × B is a simple
  per-cell multiplication.
- One spatial data structure. One GPU buffer layout. One set of grid dimensions.
- Cache-friendly access. When computing MHD forces, the fluid and EM data for
  a cell are at predictable memory offsets.
- Simplest implementation by far.

**Cons:**
- Resolution is locked. EM fields might be smooth and need only 64^3, while
  fluid turbulence needs 256^3. The unified grid must be 256^3, wasting memory
  and computation on EM cells that carry negligible information.
- Domain extent must cover everything. If EM fields extend 10 meters but fluid
  occupies a 1-meter tank, the grid must cover 10 meters at fluid resolution.
  Enormously wasteful.
- Rigidbody doesn't fit naturally on a grid. Rigid bodies have arbitrary shape,
  position, and orientation—they need a separate representation regardless.

### Option B: Separate Representations, Coupled via Interpolation

Each domain uses its own optimal representation: LBM grid for fluid, separate EM
grid (potentially different resolution), particles or shapes for rigidbody.

**Pros:**
- Each domain at optimal resolution. Fluid at 256^3, EM at 64^3, thermal at 128^3.
- Memory efficient. Domains only occupy the space they need.
- Can use different grid extents per domain.
- Easier to add new domains without restructuring existing ones.

**Cons:**
- Coupling requires interpolation between grids (trilinear, etc.). Interpolation
  introduces numerical diffusion and can violate conservation laws.
- Synchronization overhead. Must explicitly transfer data between representations.
- More complex memory management. Multiple GPU buffers with different layouts.
- Interpolation at domain boundaries can create artifacts—energy might be gained
  or lost at the coupling interface.

### Option C: Shared Primary Grid + Domain-Specific Extensions

One primary grid at the "reference" resolution. Domains that need different
resolution use derived grids (coarsened or refined) with explicit transfers.
Rigidbody uses its own shape representation but reads/writes forces from/to the grid.

**Pros:**
- Primary grid provides a common spatial reference.
- EM can use a coarsened version (every 4th cell) for efficiency.
- Fluid uses the primary grid at full resolution.
- Coupling points are well-defined: coarsen/refine operations are explicit.
- Rigidbody ↔ grid coupling via immersed boundary or volume fraction methods.

**Cons:**
- Coarsening/refinement adds implementation complexity.
- Still some interpolation involved (rigidbody ↔ grid).
- Need to ensure conservation during coarsen/refine transfers.

### Option D: Adaptive Multi-Resolution (Octree / AMR)

Use an octree or Adaptive Mesh Refinement to allocate high resolution only where
physics is active or interesting.

**Pros:**
- Optimal memory usage. Fine resolution near fluid surfaces, coarse in bulk.
- Handles large domains efficiently.
- Can refine around interesting physics (near conductors, at fluid interfaces).

**Cons:**
- Dramatically more complex implementation.
- GPU-unfriendly. Octrees have irregular memory access patterns that kill GPU
  throughput. This directly conflicts with LBM's main strength (regular parallelism).
- Refinement criteria need tuning and can oscillate.
- Neighbor access (needed for every LBM stream step) becomes indirect and slow.
- Implementation effort is very high. This is a project in itself.
- **Premature optimization.** We don't know where resolution matters until we
  have the basic engine running.

### Analysis

Option D (adaptive) is premature for v1 and hostile to LBM's GPU performance model.
We can revisit it later.

Option B (fully separate) adds coupling complexity that isn't justified if our EM
and fluid grids can reasonably share a resolution.

Option A (unified) is the simplest and guarantees exact coupling, but wastes memory
if domains need very different resolutions.

**Recommendation: Option C (shared primary grid) for v1, designed to evolve.**

Concrete proposal:
- One primary 3D grid (say 128^3 to 256^3) shared between fluid (LBM) and EM.
- Per-cell data: LBM distributions (fluid), electric potential, magnetic field
  components, temperature, material properties (conductivity, permeability).
- EM fields can optionally use a coarser sub-grid (every 2nd or 4th cell) if
  profiling shows the EM solve is a bottleneck. But start unified.
- Rigidbody is a separate system. Rigid bodies are represented as shapes
  (convex hulls, signed distance fields). Their presence on the grid is
  represented as solid cells (bounce-back in LBM, boundary conditions in EM).
  Forces on rigid bodies are computed by integrating grid quantities (pressure,
  Lorentz force, fluid drag) over the body's surface cells.

This gives us the simplest correct coupling for v1, with clear paths to optimize
later.

---

## 5. GPU Compute Strategy

### wgpu (WebGPU API via Rust)

**How it works:** Cross-platform GPU abstraction layer. Provides compute shaders
(WGSL language), render pipeline, and buffer management. Backends: Vulkan (Linux,
Windows, Android), Metal (macOS, iOS), DX12 (Windows), WebGPU (browsers).

**Pros:**
- Rust-native. First-class API with zero-cost abstractions. No FFI boundaries.
- Cross-platform by default. Write once, run on Vulkan/Metal/DX12/WebGPU.
- WebGPU browser target enables web demos, which is valuable for community
  building and marketing.
- Validation layer catches errors at development time (similar to Vulkan
  validation layers, but integrated).
- Active development by the gfx-rs team, with Mozilla and Google contributing.
  Production-quality in 2026.
- Compute shader support is solid for our workloads: regular grid access,
  per-cell computation, parallel reductions.

**Cons:**
- WGSL (WebGPU Shading Language) is less expressive than CUDA:
  - No recursion (irrelevant for our kernels).
  - Limited subgroup/warp operations (improving in newer specs).
  - No dynamic shared memory allocation.
- Performance may lag ~10-15% behind hand-tuned Vulkan on specific workloads due
  to abstraction overhead. For compute-heavy simulation, this adds up.
- Some advanced GPU features not yet exposed (cooperative groups, hardware ray
  tracing for rendering, mesh shaders). These may matter for rendering but not
  for simulation.
- Debugging/profiling tools are less mature than CUDA's Nsight or Vulkan's
  RenderDoc (though RenderDoc works with the Vulkan backend).
- Shared memory (workgroup memory) atomics have backend-dependent behavior.
  Need care for P2G-style scatter operations.

### Raw Vulkan (via ash in Rust)

**How it works:** Direct Vulkan API access through the `ash` crate (thin,
zero-overhead Rust bindings).

**Pros:**
- Maximum control over every GPU operation. Can squeeze out every available
  FLOP for critical simulation kernels.
- Full access to all Vulkan extensions and features.
- Mature debugging: RenderDoc, Vulkan validation layers, GPU vendor profilers.
- SPIR-V shader target: can compile from GLSL, HLSL, Slang, or even Rust
  (via rust-gpu).
- Can use descriptor indexing, push constants, and other Vulkan features for
  optimal shader dispatching.

**Cons:**
- Enormous API surface. Creating a single compute pipeline requires ~200 lines
  of setup code (instance, device, queue, command pool, command buffer, pipeline
  layout, descriptor sets, pipeline, etc.).
- Manual synchronization. Pipeline barriers, semaphores, fences—getting these
  wrong causes data races that are nearly impossible to debug.
- Manual memory management. VkBuffer allocation, memory types, staging buffers—
  a significant layer of complexity.
- No macOS support without MoltenVK, which is a translation layer and may not
  support all Vulkan compute features.
- No web target.
- Development velocity is much slower. Every change touches more boilerplate.
- The performance advantage over wgpu is real but narrowing. For our workloads
  (regular grids, per-cell compute), the gap may be <5%.

### CUDA (via cuda-rs or cudarc in Rust)

**How it works:** NVIDIA's proprietary GPU compute platform. Rust bindings exist
(cudarc) but are wrappers around the C API.

**Pros:**
- Best absolute GPU compute performance (on NVIDIA hardware).
- Most mature ecosystem: cuBLAS, cuFFT, cuSPARSE, Thrust, CUB—all available
  for linear algebra, FFTs, sparse solvers.
- Nsight is the gold standard for GPU profiling and debugging.
- Cooperative groups, warp-level primitives, tensor cores—maximum hardware access.
- Vast body of CUDA physics simulation implementations to reference.
- Shared memory, L1/L2 cache control, occupancy tuning—fine-grained optimization.

**Cons:**
- **NVIDIA only.** Excludes:
  - All Macs (Apple dropped NVIDIA support in 2019). macOS is a significant
    gaming/creative platform.
  - AMD GPUs (significant market share on both desktop and console).
  - Intel Arc GPUs.
  - All mobile GPUs.
  - Web browsers.
- Vendor lock-in. NVIDIA controls the platform, pricing, and roadmap.
- Rust integration is second-class. Kernels must be written in CUDA C++,
  compiled separately with nvcc, and linked via FFI.
- Licensing: CUDA toolkit is free but proprietary. Can't ship the runtime
  without agreeing to NVIDIA's terms.
- Would need a completely separate backend for non-NVIDIA platforms, effectively
  doubling the GPU code.

**Verdict: Eliminated for primary compute.** The platform exclusivity is disqualifying.
We cannot ship a game that doesn't run on Macs or AMD GPUs.

### Metal (Apple only)

Included for completeness. Same platform-exclusivity problem as CUDA, in reverse.
Not viable as a primary compute API. However, wgpu uses Metal as its macOS backend,
so we get Metal performance via wgpu automatically.

### Analysis

The real contest is **wgpu vs raw Vulkan.**

For this project, development velocity matters more than squeezing the last 5-10%
of GPU performance. We're building a novel coupled physics engine—the research and
iteration speed of getting kernels working correctly is more valuable than
hand-optimizing them from the start.

wgpu gives us:
- Cross-platform including web (huge for demos and community)
- Rust-native API (no FFI, works with the borrow checker)
- Sufficient performance for our workloads
- Faster iteration

Raw Vulkan gives us:
- ~5-10% more compute performance (narrowing)
- Full feature access
- Better profiling tools
- Much slower development

**Recommendation: wgpu for v1.** If profiling reveals that the wgpu abstraction
is a bottleneck (unlikely for our regular-grid workloads), we can drop to raw
Vulkan for specific hot paths while keeping wgpu for everything else. wgpu's
Vulkan backend means this transition is incremental, not a rewrite.

Critical validation needed: before committing, we should prototype an LBM kernel
in wgpu compute shaders and benchmark it against a reference CUDA implementation
to verify the performance gap is acceptable.

---

## 6. Timestep Strategy

### Option A: Single Unified Timestep

All domains advance by the same dt every step.

**How it works:** Pick the smallest dt required by any domain. All domains step
together. Cross-domain coupling is implicit—everything is synchronized.

**Pros:**
- Simplest possible implementation.
- Perfect synchronization. No interpolation or extrapolation between domains.
- Deterministic by construction.
- Easy to reason about correctness.

**Cons:**
- The smallest dt dominates. For LBM fluid at a usable resolution, dt might be
  ~0.1ms (10,000 steps per second of simulation). Making the rigidbody solver
  and EM circuit solver also run at 10kHz is wasteful—they converge well at
  120-500 Hz.
- Performance cost scales with the fastest domain, applied to all domains.
  If fluid needs 10,000 steps/sec and EM field diffusion needs 5,000, but
  rigidbody only needs 240, we're doing ~40x unnecessary rigidbody work.

### Option B: Per-Domain Sub-Stepping

Each domain advances at its own natural frequency within a shared outer timestep.

**How it works:** Outer loop runs at a fixed frequency (e.g., 120 Hz). Within each
outer step:
1. Exchange coupling forces between domains (snapshot)
2. Fluid sub-steps: N_fluid times at dt_fluid
3. EM sub-steps: N_em times at dt_em
4. Rigidbody sub-steps: N_rigid times at dt_rigid
5. Update coupling quantities for next outer step

**Pros:**
- Each domain runs at optimal frequency. Fluid: 1-10 kHz. EM diffusion: 500 Hz-5 kHz.
  Rigidbody: 120-480 Hz. Circuit: effectively instant (algebraic solve).
- Much more efficient than unified timestep.
- Outer step provides a clear synchronization point.
- Deterministic at the outer step level (game logic sees consistent state).

**Cons:**
- Cross-domain coupling is delayed by up to one outer timestep. A Lorentz force
  computed at the start of the outer step stays constant throughout fluid sub-steps.
  At 120 Hz outer step, this is ~8ms of stale coupling. For slow-evolving fields,
  this is fine. For fast transients, it could cause artifacts.
- Need to interpolate coupling quantities during sub-steps for smooth behavior
  (linear interpolation of forces between outer steps).
- Order of domain evaluation within the outer step matters (fluid before EM?
  EM before fluid?). Different orderings have different stability properties.
- Implementation complexity is moderate but manageable.

### Option C: Fully Adaptive Timestep

Each domain chooses its timestep based on local conditions (CFL condition, error
estimates, etc.). Timesteps vary between frames.

**Pros:**
- Optimal efficiency. Small steps only when the physics demands it.
- Automatically handles stiff systems (sudden changes, shocks).
- Can slow down gracefully under heavy load.

**Cons:**
- Non-deterministic. Different hardware speeds, floating-point ordering, and
  initial conditions can lead to different simulation trajectories. This breaks
  replays, networked multiplayer, and puzzle verification.
- Players notice variable simulation speed. A puzzle might solve differently on
  different machines.
- Much harder to debug (can't reproduce exact states).
- Complex implementation with many edge cases.

**Verdict on adaptive: Not suitable for a game.** Determinism matters for puzzles
and for players' trust in the simulation. "I built the same machine and it behaved
differently" is unacceptable.

### Option D: Fixed Outer Step + Per-Domain Sub-Stepping (Refined)

Like Option B but with explicit coupling interpolation and careful ordering.

**How it works:**
```
Fixed outer dt = 1/120 sec (or 1/60, configurable)

Each outer step:
  1. Snapshot coupling state:
     - Fluid → EM: current density J from conductive fluid motion
     - EM → Fluid: Lorentz force F = J × B, Joule heating Q
     - Fluid → Rigid: pressure forces, drag
     - Rigid → Fluid: boundary velocity (for moving objects in fluid)
     - EM → Rigid: electromagnetic force, eddy current torque
     - Rigid → EM: conductor position/velocity (for induction)

  2. EM field solve (sub-step as needed):
     - Update magnetic diffusion (eddy currents) with fixed J from step 1
     - Solve electric potential with current boundary conditions
     - N_em sub-steps, typically 4-16

  3. Fluid solve (sub-step as needed):
     - LBM stream+collide with Lorentz force from step 2
     - N_fluid sub-steps, typically 8-64
     - Accumulate forces on rigid body boundary cells

  4. Rigidbody solve:
     - Integrate with accumulated fluid + EM forces
     - Constraint solving (contacts, joints)
     - N_rigid sub-steps, typically 1-4
     - Update boundary cells on grid for next outer step

  5. Thermal update:
     - Diffuse temperature
     - Apply Joule heating source term
     - Update material properties if temperature-dependent
```

**Pros:**
- Deterministic. Fixed outer step, fixed sub-step counts.
- Efficient. Each domain at appropriate frequency.
- Clear coupling interface. The snapshot-solve-update pattern makes data flow
  explicit and debuggable.
- Ordering (EM → Fluid → Rigid) follows physical causality: fields drive fluid
  forces, fluid drives rigid body motion.
- Can tune sub-step counts per domain based on profiling.

**Cons:**
- Coupling is still outer-step-delayed. Fast electromagnetic transients (e.g.,
  switching a strong electromagnet on/off) won't be felt by the fluid until the
  next outer step.
- More implementation complexity than unified timestep.
- Sub-step counts are configuration parameters that need tuning.

### Analysis

Option D is the right approach. It gives us:
- **Determinism** (critical for puzzles and player trust)
- **Efficiency** (each domain at its natural rate)
- **Clear coupling interface** (debuggable, testable)
- **Explicit ordering** (no ambiguity about data flow)

The coupling delay of one outer step (8ms at 120Hz) is acceptable for gameplay.
Physical phenomena at frequencies above 60Hz are not perceptible to players.

**Recommendation: Option D. Fixed outer step at 120Hz with per-domain sub-stepping.**

Sub-step budget per outer step (initial estimates, to be tuned):
- EM field: 4-16 sub-steps (depending on diffusion timescale)
- Fluid (LBM): 8-64 sub-steps (depending on resolution and flow speed)
- Rigidbody: 1-4 sub-steps
- Circuit: 1 solve per outer step (algebraic, effectively instant)
- Thermal: 1-4 sub-steps (diffusion is slow)

---

## Summary of Recommendations

| Decision               | Recommendation                          | Confidence |
|------------------------|-----------------------------------------|------------|
| Language               | Rust                                    | High       |
| Fluid simulation       | LBM (primary) + particle spray          | Medium     |
| EM solver              | Hybrid: circuit + quasi-static fields   | High       |
| Spatial representation | Shared primary grid + rigidbody shapes  | Medium     |
| GPU compute            | wgpu                                    | High       |
| Timestep strategy      | Fixed outer 120Hz + per-domain sub-step | High       |

"Medium" confidence items (fluid method, spatial representation) should be validated
with prototypes before committing to the full engine build.

---

## Open Questions

1. **Rigidbody solver:** Build from scratch or use Rapier (Rust physics library)?
   Rapier is mature but may not integrate cleanly with our grid-based coupling.

2. **Rendering:** Separate from simulation. What rendering approach? Forward vs
   deferred, how to render fluid surfaces (screen-space, marching cubes, ray
   marching), how to visualize EM fields for the player.

3. **Grid resolution vs. world size trade-off:** What's the target? A 256^3 grid
   at 1cm resolution gives a 2.56m cube. Is that big enough for puzzles?
   512^3 gives 5.12m but demands 8x the memory and compute.

4. **Serialization / determinism:** Puzzles need exact reproducibility. IEEE 754
   floating-point is deterministic per-platform, but cross-platform determinism
   (AMD vs NVIDIA GPU) requires careful handling. Do we need cross-platform
   determinism?

5. **Editor / UI framework:** What do we use for the "always live" level editor
   UI? egui (Rust-native immediate mode) is the obvious choice.

6. **Audio:** Physics-driven audio (fluid sounds from simulation state, electrical
   hum from current, mechanical clunks from rigid body contacts) would be incredible
   for immersion. How deep do we go?
