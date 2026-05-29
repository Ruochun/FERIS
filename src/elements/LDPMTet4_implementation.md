# LDPM-TET4 GPU Implementation — Theory-to-Code Analysis

> **Repository:** Ruochun/FERIS  
> **Reference:** Cusatis, Pelessone & Mencarelli (2011), *"Lattice Discrete Particle
> Model (LDPM) for failure behavior of concrete I: Theory"*,
> Cement & Concrete Composites 33(9): 881–890.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mesh Discretisation — Particles, TET4, and Unique Edges](#2-mesh-discretisation--particles-tet4-and-unique-edges)
3. [Reference Geometry — Facet Frame and Area](#3-reference-geometry--facet-frame-and-area)
   - 3.1 [Path A: TET4-derived geometry (Setup)](#31-path-a-tet4-derived-geometry-setup)
   - 3.2 [Path B: File-based override (SetupFromMesh)](#32-path-b-file-based-override-setupfrommesh)
4. [Nodal State — 6 DOFs per Particle](#4-nodal-state--6-dofs-per-particle)
5. [Strain Computation — compute_p](#5-strain-computation--compute_p)
6. [Constitutive Law — Cusatis Tensile-Shear Damage](#6-constitutive-law--cusatis-tensile-shear-damage)
   - 6.1 [Damage driving strain](#61-damage-driving-strain)
   - 6.2 [History variable — irreversibility](#62-history-variable--irreversibility)
   - 6.3 [Crack and failure definition](#63-crack-and-failure-definition)
   - 6.4 [Traction law — asymmetric tension/compression](#64-traction-law--asymmetric-tensioncompression)
   - 6.5 [Rotational moments — always elastic](#65-rotational-moments--always-elastic)
   - 6.6 [Code implementation — full function signature](#66-code-implementation--full-function-signature)
7. [Force Assembly — Facet Tractions to Nodal Forces](#7-force-assembly--facet-tractions-to-nodal-forces)
8. [Mass Matrix](#8-mass-matrix)
9. [Leapfrog Time Integration](#9-leapfrog-time-integration)
10. [Crack-Distance Visualisation — Projecting onto Sub-Facets](#10-crack-distance-visualisation--projecting-onto-sub-facets)
11. [Complete Workflow Summary](#11-complete-workflow-summary)
12. [Theory-Symbol to Code-Variable Reference Table](#12-theory-symbol-to-code-variable-reference-table)

---

## 1. Overview

The Lattice Discrete Particle Model (LDPM) represents a concrete specimen as a set of
rigid **particles** (nodes) whose interactions are carried by Voronoi-polyhedral
**facets** between adjacent particle pairs. The key physical idea is that fracture
is not prescribed globally but emerges spontaneously when enough individual facets
lose their load-carrying capacity.

The implementation supports **two constitutive law modes**:

1. **Full LDPM model** (activated via `SetLDPMParams`): tensile fracture with mode-mixity,
   compressive yielding/hardening/densification, pressure-dependent friction.
2. **Legacy simplified model** (via `SetMaterial` + `SetDamageParams`): single tensile-shear
   damage variable, elastic compression.

This implementation runs the full LDPM workflow on the GPU. The main types involved are:

| Type | File | Role |
|---|---|---|
| `GPU_LDPMTet4_Data` | `LDPMTet4Data.cuh` / `LDPMTet4Data.cu` | Stores all per-particle and per-edge GPU arrays; owns Setup/Teardown |
| `LDPMParams` | `LDPM.cuh` | Struct holding all 24 material parameters for the full model |
| `ldpm_tet4_full_constitutive` | `LDPM.cuh` | Device function: full LDPM constitutive update (fracture + compression + friction) |
| `ldpm_fracture_boundary` | `LDPM.cuh` | Device function: tensile-shear fracture with mode-mixity softening |
| `ldpm_compress_boundary` | `LDPM.cuh` | Device function: compressive yielding / hardening / pore collapse |
| `ldpm_shear_boundary` | `LDPM.cuh` | Device function: pressure-dependent frictional shear limit |
| `ldpm_tet4_cusatis_traction` | `LDPM.cuh` | Device function: legacy simplified constitutive law (backward compat) |
| `compute_p` | `LDPMTet4DataFunc.cuh` | Device function: strains → constitutive dispatch → tractions for one edge |
| `compute_internal_force` | `LDPMTet4DataFunc.cuh` | Device function: scatter tractions to nodal forces |
| `LeapfrogSolver` | `LeapfrogSolver.cu` | Explicit time integrator; dispatches all CUDA kernels |

---

## 2. Mesh Discretisation — Particles, TET4, and Unique Edges

### Theory

LDPM places aggregate particles at random positions and connects them by a 3-D
Delaunay triangulation (TET4 mesh). Every unique edge of the TET4 mesh is one
**interaction strut** between two particles. Each strut is associated with a Voronoi
polyhedral facet that sits between the two particles.

### Implementation

`GPU_LDPMTet4_Data::Setup()` iterates over every TET element and all six possible
local edges `{(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}`. A
`std::map<pair<int,int>, vector<int>>` keyed by sorted node pairs accumulates which
TET elements share each unique edge:

```cpp
// LDPMTet4Data.cu — Setup(), step 5
static const int LOCAL_EDGES[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

std::map<std::pair<int,int>, std::vector<int>> edge_to_tets;
for (int e = 0; e < n_elem; e++) {
    for (int le = 0; le < 6; le++) {
        int a = tet_connectivity(e, LOCAL_EDGES[le][0]);
        int b = tet_connectivity(e, LOCAL_EDGES[le][1]);
        if (a > b) std::swap(a, b);
        edge_to_tets[{a, b}].push_back(e);
    }
}
n_edge = static_cast<int>(edge_to_tets.size());
```

After this loop, `n_edge` unique particle–particle struts have been identified.
The solver treats each strut as one "element": `get_n_beam()` returns `n_edge`.

---

## 3. Reference Geometry — Facet Frame and Area

### Theory

For each strut (i, j), LDPM defines an **interaction facet** whose key geometric
quantities are:

| Symbol | Meaning |
|---|---|
| l₀ | Reference (undeformed) edge length ‖**x**ⱼ − **x**ᵢ‖ |
| **n** | Unit normal along **x**ⱼ − **x**ᵢ (the strut direction) |
| **m**, **l** | Two orthogonal tangent vectors spanning the facet plane |
| A | Projected Voronoi facet area |

All reference geometry is computed once at setup time and stored as read-only
per-edge GPU arrays. It is **never updated** during the simulation, consistent
with LDPM's **linearised (small-displacement) kinematics**: strains are
engineering measures (Δu / l₀), so the reference frame is by definition
the correct frame to project onto throughout.  This is *not* a
Total-Lagrangian continuum formulation — LDPM is a discrete lattice model,
not FEA, and its strain measures are first-order (engineering) quantities,
not Green–Lagrange strains.

### 3.1 Path A: TET4-derived geometry (Setup)

When no external mesh files are provided, `Setup()` computes all geometry
algebraically from the particle positions.

**Reference normal and tangent frame:**

```cpp
// LDPMTet4Data.cu — Setup(), step 6
Real dx = h_x(nj) - h_x(ni);
Real dy = h_y(nj) - h_y(ni);
Real dz = h_z(nj) - h_z(ni);
const Real l0_val = std::sqrt(dx*dx + dy*dy + dz*dz);   // l₀

const Real nx = dx * inv_l0;   // n = (xⱼ − xᵢ) / l₀
const Real ny = dy * inv_l0;
const Real nz = dz * inv_l0;

// m: cardinal axis least parallel to n, then l = n × m
Real mx, my, mz;
if (std::abs(nx) <= std::abs(ny) && std::abs(nx) <= std::abs(nz)) {
    mx = 0;   my = nz;  mz = -ny;
} else if (std::abs(ny) <= std::abs(nz)) {
    mx = -nz; my = 0;   mz = nx;
} else {
    mx = ny;  my = -nx; mz = 0;
}
// normalise m, then l = n × m
const Real lvx = ny*mz - nz*my;
const Real lvy = nz*mx - nx*mz;
const Real lvz = nx*my - ny*mx;
```

**Voronoi facet area** (sum of two sub-triangles per shared TET):

```cpp
// edge midpoint
const Real mid[3] = {0.5*(xi+xj), 0.5*(yi+yj), 0.5*(zi+zj)};

for (int t : tets) {
    // c_ijk = centroid of the TET face containing edge (i,j,k)
    // c_ijl = centroid of the TET face containing edge (i,j,l)
    // c_tet = centroid of the TET
    area_sum += triangle_area(mid, c_ijk, c_tet);
    area_sum += triangle_area(mid, c_ijl, c_tet);
}
h_facet_area[edge_idx] = area_sum;
```

where `triangle_area(a, b, c)` = ½ |(**b**−**a**) × (**c**−**a**)|.

### 3.2 Path B: File-based override (SetupFromMesh)

When the Chrono Workbench preprocessor has generated mesh files, `SetupFromMesh()`
replaces the TET4-derived facet area and tangent frame with richer values read
from `facets.dat`. Each row of that file describes one Voronoi **sub-facet**:

```
Tet  IDx IDy IDz  Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  mF
```

where `p` is the unit normal along the strut direction, `q` the first tangent, and
`s = p × q` the second tangent.

The override loop matches each sub-facet to its global edge by comparing `p` against
all six local edge directions of the owning TET:

```cpp
// LDPMTet4Data.cu — SetupFromMesh()
for (int le = 0; le < 6; le++) {
    const Real d = (px*dx + py*dy + pz*dz) * inv_len;  // dot(p, edge direction)
    if (std::abs(d) > best_abs_dot) {
        best_abs_dot = std::abs(d);
        best_le = le;
        best_sign = (na < nb) ? d : -d;   // sign relative to canonical ni→nj
    }
}
// accumulate projected area
new_area[eidx] += parea;

// store tangent frame from first matching sub-facet (sign-adjusted)
const Real sign_m = (best_sign > 0) ? 1 : -1;
new_m[eidx*3+0] = sign_m * q0;   // m may need sign flip
new_l[eidx*3+0] = s0;            // l = s never needs a flip (see proof below)
```

> **Why the sign flip on m but not l?**
> The canonical edge direction always points from the lower-index node to the
> higher-index one. If a sub-facet's `p` is anti-parallel to that direction then
> the file tangent `q` must be negated so that `l = n_canonical × m` still equals
> `s`. The second tangent `s = p × q` never needs negation because both `p` and `q`
> flip simultaneously: `(−p) × (−q) = p × q = s`. This is verified in-code with
> the inline proof comment in `SetupFromMesh()`.

---

## 4. Nodal State — 6 DOFs per Particle

### Theory

Each particle carries **6 degrees of freedom**: three translational positions
(x, y, z) and three linearised rotations (θ_x, θ_y, θ_z). The conjugate resultants
are translational forces (f_x, f_y, f_z) and rotational moments (m_x, m_y, m_z).

### Implementation

Six separate GPU device arrays hold the per-particle state, each of length `n_nodes`:

```cpp
// LDPMTet4Data.cuh — device accessors
__device__ Map<VectorXR> x12()     { return Map<VectorXR>(d_x12,  n_nodes); }
__device__ Map<VectorXR> rx12()    { return Map<VectorXR>(d_rx12, n_nodes); }  // θ_x

__device__ Map<VectorXR> f_int()   { return Map<VectorXR>(d_f_int_t, n_nodes*3); }
__device__ Map<VectorXR> f_int_r() { return Map<VectorXR>(d_f_int_r, n_nodes*3); }
```

In `LeapfrogSolver` the velocity array layout for `TYPE_LDPM_TET4` is:

```
da_v_[0 … 3*n_nodes−1]          ← translational half-step velocities
da_v_[3*n_nodes … 6*n_nodes−1]   ← rotational half-step velocities  (v_rot)
```

and the mass/inertia array as:

```
da_mass_lump_[0 … n_nodes−1]         ← translational lumped mass  mᵢ
da_mass_lump_[n_nodes … 2*n_nodes−1]  ← rotational inertia Iᵢ
```

---

## 5. Strain Computation — compute_p

### Theory

For each strut (i, j), the engineering strains are projections of the
**relative displacement** Δ**u** and the **rotation difference** Δ**θ** onto the
three facet frame directions (n, m, l):

| Strain | Formula | Physical meaning |
|---|---|---|
| e_N | (Δu · **n**) / l₀ | Normal opening (+) / closing (−) |
| e_M | (Δu · **m**) / l₀ | Shear sliding, m-direction |
| e_L | (Δu · **l**) / l₀ | Shear sliding, l-direction |
| κ_T | (Δθ · **n**) / l₀ | Twist |
| κ_M | (Δθ · **m**) / l₀ | Bending about m |
| κ_L | (Δθ · **l**) / l₀ | Bending about l |

where Δ**u** = (**x**ⱼ − **x**ᵢ) − l₀ **n** is the displacement of the current
edge vector relative to the reference (undeformed) configuration, and l₀ and **n**
are fixed reference-configuration quantities consistent with LDPM's linearised
kinematics.

### Implementation

`compute_p()` in `LDPMTet4DataFunc.cuh` is called once per edge per time step:

```cpp
// LDPMTet4DataFunc.cuh — translational strains
Real r[3];
r[0] = d_data->x12()(nj) - d_data->x12()(ni);   // current edge vector
r[1] = d_data->y12()(nj) - d_data->y12()(ni);
r[2] = d_data->z12()(nj) - d_data->z12()(ni);

// Relative displacement from reference: Δu = r − l₀·n_ref
const Real du0 = r[0] - l0 * n0;
const Real du1 = r[1] - l0 * n1;
const Real du2 = r[2] - l0 * n2;

const Real e_N = (du0*n0 + du1*n1 + du2*n2) * inv_l0;
const Real e_M = (du0*m0 + du1*m1 + du2*m2) * inv_l0;
const Real e_L = (du0*l_0 + du1*l_1 + du2*l_2) * inv_l0;

// Rotational strains
const Real dt0 = d_data->rx12()(nj) - d_data->rx12()(ni);   // Δθ
const Real dt1 = d_data->ry12()(nj) - d_data->ry12()(ni);
const Real dt2 = d_data->rz12()(nj) - d_data->rz12()(ni);

const Real kappa_T = (dt0*n0 + dt1*n1 + dt2*n2) * inv_l0;
const Real kappa_M = (dt0*m0 + dt1*m1 + dt2*m2) * inv_l0;
const Real kappa_L = (dt0*l_0 + dt1*l_1 + dt2*l_2) * inv_l0;
```

> **Linearised-kinematics note:** All reference geometry (`l0`, `n`, `m`, `l`) is
> precomputed once and stored in read-only per-edge arrays. The subtraction
> `r − l₀·n` always measures strain relative to the undeformed configuration, which
> is correct for LDPM's engineering (first-order) strain measures.  This is *not*
> the same as a Total-Lagrangian continuum FEA formulation (which would use the
> deformation gradient **F** and Green–Lagrange strains); LDPM is a discrete lattice
> model whose linearised kinematics assume small displacements and small rotations.

---

## 6. Constitutive Law

The constitutive law is the core of the failure model.  `compute_p()` in
`LDPMTet4DataFunc.cuh` dispatches to one of two constitutive paths based on
`d_data->use_full_ldpm()` (set to `true` by `SetLDPMParams()`):

- **Full LDPM model** → `ldpm_tet4_full_constitutive()` in `LDPM.cuh`
- **Legacy model** → `ldpm_tet4_cusatis_traction()` in `LDPM.cuh`

### 6.0 Constitutive dispatch in `compute_p()`

```cpp
// LDPMTet4DataFunc.cuh
if (d_data->use_full_ldpm()) {
    // Strain increments relative to stored accumulated state
    const Real d_eps_N = e_N - statev[0];
    const Real d_eps_M = e_M - statev[1];
    const Real d_eps_L = e_L - statev[2];
    const Real eps_V = e_N;  // volumetric strain approximation

    ldpm_tet4_full_constitutive(d_eps_N, d_eps_M, d_eps_L,
        kappa_T, kappa_M, kappa_L, eps_V, l0,
        d_data->ldpm_params(), statev,
        t_N, t_M, t_L, m_T, m_M, m_L);

    // Write back state + maintain legacy kappa/omega for VTK output
} else {
    ldpm_tet4_cusatis_traction(e_N, e_M, e_L,
        kappa_T, kappa_M, kappa_L,
        E_N, E_T, E_kT, E_kM, E_kL, sigma_t, H_t,
        kappa_in, kappa_out, omega_out,
        t_N, t_M, t_L, m_T, m_M, m_L);
}
```

---

### 6.A Full LDPM Model (`ldpm_tet4_full_constitutive`)

This implements the complete Cusatis et al. (2011) facet model with three
constitutive mechanisms: tensile fracture, compressive pore collapse, and
pressure-dependent friction.

#### 6.A.1 Parameter set (`LDPMParams`)

The full model uses 24 parameters stored in the `LDPMParams` struct:

| Category | Parameters |
|---|---|
| Elastic | `E0`, `alpha`, `E_kT`, `E_kM`, `E_kL`, `rho` |
| Tensile | `sigma_t`, `sigma_s`, `n_t`, `l_t`, `r_s`, `k_t` |
| Compression | `sigma_c0`, `H_c0`, `H_c1`, `kc0`, `kc1`, `kc2`, `kc3`, `beta`, `E_d` |
| Friction | `mu_0`, `mu_inf`, `sigma_N0` |
| Control | `elastic_flag` |

Set via `GPU_LDPMTet4_Data::SetLDPMParams(const LDPMParams& params)`.

#### 6.A.2 Per-edge state vector (16 components)

Each edge stores a persistent state vector (`LDPM_N_STATEV = 16`):

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | eps_N | accumulated normal strain |
| 1 | eps_M | accumulated shear M strain |
| 2 | eps_L | accumulated shear L strain |
| 3 | sigma_N | current normal stress |
| 4 | sigma_M | current shear M stress |
| 5 | sigma_L | current shear L stress |
| 6 | max_eps_N | maximum normal strain history |
| 7 | max_eps_T | maximum shear magnitude history |
| 8 | eff_strain | effective strain |
| 9 | eff_stress | effective stress |
| 10 | W_int | internal work |
| 11 | crack_w | crack opening displacement |
| 12-14 | reserved | for eigenstrain support |
| 15 | W_diss | dissipated energy |

#### 6.A.3 Tensile fracture — `ldpm_fracture_boundary()`

When `eps_N > 0`, the tensile/fracture branch is active. The function computes:

1. **Mode-mixity angle:** `omega = atan(eps_N / (sqrt(alpha) * eps_T))`
2. **Effective strength** with mixed-mode dependence on `r_st = sigma_s / sigma_t`
3. **Softening modulus:** `H0 = Hs/alpha + (Ht - Hs/alpha) * (2*omega/pi)^n_t`
   where `Ht = 2*E0 / (l_t/l0 - 1)` and `Hs = r_s * E0`
4. **Exponential softening:** `sigma_bt = sigma0 * exp(-H0 * max(eps_max - eps0, 0) / sigma0)`
5. **Unloading** via `k_t` parameter
6. **Stress projection:** `sigma_N = sigma_fr * eps_N / eps_Q`,
   `sigma_M = alpha * sigma_fr * eps_M / eps_Q`

```cpp
// LDPM.cuh — ldpm_fracture_boundary()
const Real strs_Q = ldpm_fracture_boundary(eps_N, eps_M, eps_L, l0,
    statev[6], statev[7], statev[8], statev[9], params);
sigma_N = strs_Q * eps_N / eps_Q;
sigma_M = alpha * strs_Q * eps_M / eps_Q;
sigma_L = alpha * strs_Q * eps_L / eps_Q;
```

#### 6.A.4 Compressive boundary — `ldpm_compress_boundary()`

When `eps_N <= 0`, the compressive boundary is applied independently:

1. **Deviatoric strain:** `eps_D = eps_N - eps_V`
2. **Modified volumetric strain:** `eps_DV = eps_V + beta * eps_D`
3. **Confinement ratio:** `r_DV = |eps_D| / (eps_V - eps_v0)` (or `/eps_v0` when `eps_V > 0`)
4. **Confinement-sensitive hardening:** `Hc = (H_c0 - H_c1) / (1 + kc2 * max(r_DV - kc1, 0)) + H_c1`
5. **Three-phase boundary:** elastic → linear hardening → exponential compaction
6. **Densification modulus** `E_d` replaces `E_0` after yielding

```cpp
// LDPM.cuh — ldpm_compress_boundary()
sigma_N = ldpm_compress_boundary(eps_N, eps_V, statev[3], statev[0], params);
```

#### 6.A.5 Frictional shear under compression — `ldpm_shear_boundary()`

When the facet is in compression, shear is limited by a pressure-dependent friction:

- **Shear boundary:** `sigma_bs = sigma_s + (mu0 - mu_inf)*sigma_N0 - mu_inf*sigma_N
  - (mu0 - mu_inf)*sigma_N0 * exp(sigma_N / sigma_N0)`
- Elastic trial clamped to boundary with return mapping to maintain direction

```cpp
// LDPM.cuh — ldpm_shear_boundary()
ldpm_shear_boundary(eps_M, eps_L, sigma_N, statev[4], statev[5],
    statev[1], statev[2], sigma_M, sigma_L, params);
```

#### 6.A.6 Energy bookkeeping and crack opening

After stress update, the function computes:
- **Internal work** (incremental, trapezoidal rule): added to `statev[10]`
- **Dissipated energy** (based on inelastic strain increments): added to `statev[15]`
- **Crack opening displacement:** `w = sqrt(w_N² + w_M² + w_L²)` stored in `statev[11]`

#### 6.A.7 Legacy kappa/omega maintenance

For backward-compatible VTK output, the full model also updates the legacy
`d_kappa[]` and `d_omega[]` arrays:

```cpp
// LDPMTet4DataFunc.cuh — after full constitutive update
d_data->edge_kappa(edge_idx) = statev[8];  // effective strain as kappa proxy
const Real crack_w = statev[11];
const Real eff_crack = (l0 > 0) ? crack_w / l0 : 0;
d_data->edge_omega(edge_idx) = min(eff_crack / max(statev[8], 1e-30), 1.0);
```

---

### 6.B Legacy Model (`ldpm_tet4_cusatis_traction`)

The legacy model is a simplified single-variable tensile-shear damage law. It is
used when `SetLDPMParams()` has **not** been called (i.e., only `SetMaterial` +
`SetDamageParams` are used). It is kept for backward compatibility.

### 6.B.1 Damage driving strain

The model couples the tensile opening strain and the two shear strains into a single
scalar **damage driving strain** e_D:

```
e_D = sqrt( ⟨e_N⟩₊²  +  α (e_M² + e_L²) )
```

where `⟨e_N⟩₊ = max(e_N, 0)` (only tensile opening contributes) and
`α = E_T / E_N` (the shear-to-normal stiffness ratio).

A facet that is sliding with zero normal opening can still accumulate damage as long
as the shear is large enough to drive e_D above the tensile threshold.

```cpp
// LDPM.cuh
const Real e_N_pos = (e_N > Real(0)) ? e_N : Real(0);
const Real alpha   = (E_N > Real(0)) ? E_T / E_N : Real(0);
const Real e_D     = sqrt(e_N_pos*e_N_pos + alpha*(e_M*e_M + e_L*e_L));
```

### 6.B.2 History variable — irreversibility

`κ` (kappa) is the **maximum value of e_D ever reached** by a facet. It is a
persistent per-edge GPU state variable stored in `d_kappa[]` that can only grow
monotonically:

```
κ_new = max(κ_old, e_D)
```

```cpp
// LDPM.cuh
kappa_out = (e_D > kappa_in) ? e_D : kappa_in;
```

This ratchet mechanism means damage is **irreversible**: once a facet has been
strained beyond its elastic limit, κ retains that peak value even if the load is
subsequently removed or reversed.

### 6.B.3 Crack and failure definition

This is the central concept for understanding how fracture is represented in the code.

**Cracking is defined at the facet level**, not at the specimen level. There is no
explicit "crack flag" or "crack surface" variable. Instead, the scalar damage
variable `ω` (omega) for each facet completely encodes its fracture state:

```
ω = 0                                                 if κ ≤ e_t0   (elastic — no damage)
ω = 1 − (e_t0 / κ) · exp(−H_t · (κ − e_t0))         if κ > e_t0   (softening — cracked)
```

where `e_t0 = σ_t / E_N` is the **peak strain** — the strain at which the facet
first reaches its tensile strength σ_t and begins to crack.

The damage variable ω ranges continuously from 0 to 1:

| ω | Physical meaning |
|---|---|
| 0 | Facet is fully intact; behaviour is linear-elastic |
| 0 < ω < 1 | Facet has **cracked** but still transmits a residual (degraded) force — the softening branch |
| ω ≈ 1 | Facet is **fully failed** — it no longer transmits any tensile or shear force |

A **macroscopic crack** in the simulation is an emergent phenomenon: it appears as a
contiguous band or surface of facets that have all reached ω ≈ 1. There is no
prescribed crack path; the crack propagates wherever the damage state variable reaches
the fully-failed condition.

**In code**, a facet is cracked as soon as `kappa > e_t0`. It is fully failed when
`omega ≈ 1`. Both values are read directly from the device arrays `d_kappa[]` and
`d_omega[]` at any time step:

```cpp
// LDPM.cuh — damage variable computation
const Real e_t0 = (E_N > Real(0)) ? sigma_t / E_N : Real(0);

Real omega = Real(0);
if (kappa_out > e_t0 && e_t0 > Real(0)) {
    // Exponential softening: omega grows from 0 toward 1 as kappa grows
    omega = Real(1) - (e_t0 / kappa_out) * exp(-H_t * (kappa_out - e_t0));
    // Clamp to [0, 1] for numerical robustness
    omega = (omega < Real(0)) ? Real(0) : omega;
    omega = (omega > Real(1)) ? Real(1) : omega;
}
omega_out = omega;
```

After the constitutive call, `compute_p()` writes both updated values back to the
per-edge persistent arrays:

```cpp
// LDPMTet4DataFunc.cuh — persist damage state across time steps
d_data->edge_kappa(edge_idx) = kappa_new;
d_data->edge_omega(edge_idx) = omega_new;
```

The parameter `H_t` (softening modulus) controls the rate at which ω approaches 1
after the peak. It is related to the mode-I fracture energy G_ft and a characteristic
crack-bandwidth length l_ch (approximately equal to the strut edge length l₀):

```
H_t ≈ l_ch · σ_t / G_ft
```

A larger H_t means a steeper (more brittle) softening branch; H_t → 0 gives
perfectly plastic behaviour. For normal-strength concrete (representative values given
in the `LDPM.cuh` header):

```
σ_t = 4 N/mm², E_N = 60 000 N/mm², G_ft = 0.05 N/mm, l_ch = 10 mm
→ e_t0 = 6.67 × 10⁻⁵,   H_t ≈ 800
```

### 6.B.4 Traction law — asymmetric tension/compression

The traction law uses ω to degrade facet stiffness, but applies damage
**asymmetrically**:

```
t_N = E_N · e_N                    for e_N ≤ 0   (compression — always fully elastic)
t_N = E_N · e_N · (1 − ω)          for e_N > 0   (tension — degraded by damage)

t_M = E_T · e_M · (1 − ω)          (shear always degraded)
t_L = E_T · e_L · (1 − ω)
```

The asymmetry is physically correct: crack surfaces can close and still carry
compressive contact forces through the concrete matrix, even when the facet has fully
failed in tension and shear.

```cpp
// LDPM.cuh — asymmetric traction
t_N = (e_N > Real(0)) ? E_N * e_N * (Real(1) - omega) : E_N * e_N;
t_M = E_T * e_M * (Real(1) - omega);
t_L = E_T * e_L * (Real(1) - omega);
```

### 6.B.5 Rotational moments — always elastic

The first-order Cusatis formulation does not couple bending or twist to the
tensile-shear damage. Rotational moments remain linear-elastic throughout:

```
m_T = E_kT · κ_T,   m_M = E_kM · κ_M,   m_L = E_kL · κ_L
```

```cpp
// LDPM.cuh
m_T = E_kT * kappa_T;
m_M = E_kM * kappa_M;
m_L = E_kL * kappa_L;
```

### 6.B.6 Code implementation — full function signature

`ldpm_tet4_cusatis_traction()` in `LDPM.cuh` implements everything above as a
`__device__ __forceinline__` function called once per edge per step:

```cpp
__device__ __forceinline__ void ldpm_tet4_cusatis_traction(
    Real e_N, Real e_M, Real e_L,             // translational strains (inputs)
    Real kappa_T, Real kappa_M, Real kappa_L, // rotational strains (inputs)
    Real E_N, Real E_T,                       // translational moduli
    Real E_kT, Real E_kM, Real E_kL,          // rotational moduli
    Real sigma_t,                             // mesoscale tensile strength
    Real H_t,                                 // softening modulus
    Real kappa_in,                            // history variable on entry (input)
    Real& kappa_out,                          // updated history variable (output)
    Real& omega_out,                          // damage variable (output, ∈ [0,1])
    Real& t_N, Real& t_M, Real& t_L,         // tractions (outputs)
    Real& m_T, Real& m_M, Real& m_L          // moments (outputs)
)
```

The results are cached in the per-edge traction array and the per-edge damage arrays:

```cpp
// LDPMTet4DataFunc.cuh — cache results
d_data->edge_kappa(edge_idx) = kappa_new;   // d_kappa[edge_idx]
d_data->edge_omega(edge_idx) = omega_new;   // d_omega[edge_idx]

d_data->facet_t(edge_idx, 0) = t_N;        // d_facet_t[edge_idx*6 + 0]
d_data->facet_t(edge_idx, 1) = t_M;
d_data->facet_t(edge_idx, 2) = t_L;
d_data->facet_t(edge_idx, 3) = m_T;
d_data->facet_t(edge_idx, 4) = m_M;
d_data->facet_t(edge_idx, 5) = m_L;
```

---

## 7. Force Assembly — Facet Tractions to Nodal Forces

### Theory

The net force that strut (i, j) exerts on its two endpoint particles is obtained by
multiplying each traction/moment by the facet area A and projecting back into the
global coordinate system. Newton's third law gives equal and opposite reactions:

```
f_j += +A · (t_N · n + t_M · m + t_L · l)
f_i += −A · (t_N · n + t_M · m + t_L · l)

M_j += +A · (m_T · n + m_M · m + m_L · l)
M_i += −A · (m_T · n + m_M · m + m_L · l)
```

### Implementation

`compute_internal_force()` processes one endpoint of one edge per thread. The
`node_local` parameter (0 = node i, sign −1; or 1 = node j, sign +1) selects
which end is updated. `atomicAdd` is required because multiple edges share nodes
and run concurrently on the GPU:

```cpp
// LDPMTet4DataFunc.cuh
const int global_node = (node_local == 0) ? ni : nj;
const Real sign = (node_local == 0) ? Real(-1) : Real(1);
const Real A = d_data->facet_area(edge_idx);

// Translational force
for (int k = 0; k < 3; k++) {
    const Real f_k = sign * A *
        (t_N * d_data->edge_n_comp(edge_idx, k)
       + t_M * d_data->edge_m_comp(edge_idx, k)
       + t_L * d_data->edge_lv_comp(edge_idx, k));
    atomicAdd(&d_data->f_int()(global_node * 3 + k), f_k);
}

// Rotational moment
for (int k = 0; k < 3; k++) {
    const Real m_k = sign * A *
        (m_T * d_data->edge_n_comp(edge_idx, k)
       + m_M * d_data->edge_m_comp(edge_idx, k)
       + m_L * d_data->edge_lv_comp(edge_idx, k));
    atomicAdd(&d_data->f_int_r()(global_node * 3 + k), m_k);
}
```

The `n_shape_ = 2` parameter registered in the solver tells
`leapfrog_compute_f_int_kernel` that each "element" (edge) has exactly 2 local
nodes, spawning `2 · n_edge` grid-stride iterations.

---

## 8. Mass Matrix

### Theory

LDPM uses a **lumped mass matrix**: each particle's translational mass equals its
density times the Voronoi cell volume surrounding it. The rotational inertia is
scaled from the translational mass by a characteristic length:

```
mᵢ = ρ · Vᵢ                   (Voronoi volume, approximated as ¼ TET sum)
Iᵢ = α · mᵢ · l_min,i²        (α = 0.25,  l_min = shortest attached edge)
```

### Implementation

`CalcMassMatrix()` computes both on the host using a ¼-TET-volume split as an
approximation for the Voronoi cell volume:

```cpp
// LDPMTet4Data.cu — translational lumped mass
const Real vol = std::abs(a[0]*cross_bc[0]
                         + a[1]*cross_bc[1]
                         + a[2]*cross_bc[2]) / Real(6);   // TET volume
node_volumes[n0] += vol / Real(4);   // each TET contributes ¼ to each node
node_volumes[n1] += vol / Real(4);
node_volumes[n2] += vol / Real(4);
node_volumes[n3] += vol / Real(4);
```

```cpp
// rotational inertia:  I_i = ALPHA_ROT * m_i * l_min_i²
//   LDPM_TET4_ALPHA_ROT = 0.25  (defined in LDPMTet4Data.cuh)
h_I_lump[i] = LDPM_TET4_ALPHA_ROT * m_i * node_l_min[i] * node_l_min[i];
```

The translational mass is stored in a diagonal CSR matrix (nnz = n_nodes). The
specialised `leapfrog_compute_lumped_mass_ldpm_tet4_kernel` copies the rotational
inertia into the second half of the solver's mass array:

```cpp
// LeapfrogSolver.cu
d_solver->mass_lump()(node_i)    = m_i;                    // translational
d_solver->mass_inertia()(node_i) = d_data->I_lump(node_i); // rotational
```

---

## 9. Leapfrog Time Integration

### Theory

The leapfrog (central-difference) explicit integrator advances the simulation by
one time step Δt:

```
1.  Compute strains e and κ at all facets  (= compute_p)
2.  Apply constitutive law → tractions t, moments m, updated κ and ω
3.  Assemble nodal forces:  f_int, M_int
4.  Velocity update:
      v_{n+½}  = v_{n−½}  + Δt · M_lump⁻¹  · (f_ext  − f_int)
      ω_{n+½} = ω_{n−½} + Δt · I_lump⁻¹  · (M_ext − M_int)
5.  Enforce BCs: zero velocity at fixed nodes (all 6 DOFs)
6.  Position advance:
      x_{n+1} = xₙ + Δt · v_{n+½}
      θ_{n+1} = θₙ + Δt · ω_{n+½}
```

### Implementation

`LeapfrogSolver::OneStepLeapfrog()` for `TYPE_LDPM_TET4` launches seven sequenced
CUDA kernel groups:

```cpp
// LeapfrogSolver.cu — LDPM_TET4 branch

// Step 1+2: facet strains → constitutive law → tractions (grid-stride over n_edge)
leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 3: assemble internal forces (clear first, then atomicAdd)
leapfrog_clear_f_int_kernel   <<<blocks_dof, threadsPerBlock>>>(typed_data);
leapfrog_compute_f_int_kernel <<<blocks_if,  threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 4a: translational velocity update
leapfrog_update_velocity_kernel    <<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 4b: rotational velocity update
leapfrog_update_velocity_rot_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 5: zero all 6 velocity DOFs at fixed nodes
leapfrog_apply_bc_ldpm_tet4_kernel <<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 6a: advance translational positions
leapfrog_update_position_kernel <<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

// Step 6b: advance rotational positions
leapfrog_update_rotation_kernel <<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);
```

Key velocity-update kernels:

```cpp
// Translational:  v_{n+½,k} += dt * (1/mᵢ) * (f_ext,k − f_int,k)
Real net_force = d_data->f_ext()(tid) - d_data->f_int()(tid);
d_solver->v()(tid) += dt * inv_mass * net_force;

// Rotational:  ω_{n+½,k} += dt * (1/Iᵢ) * (M_ext,k − M_int,k)
Real net_moment = d_data->f_ext_r()(tid) - d_data->f_int_r()(tid);
d_solver->v_rot()(tid) += dt * inv_inertia * net_moment;
```

Position advances:

```cpp
// Translational:  x_{n+1} = xₙ + dt · v_{n+½}
d_data->x12()(tid) += dt * d_solver->v()(tid * 3 + 0);

// Rotational:  θ_{n+1} = θₙ + dt · ω_{n+½}
d_data->rx12()(tid) += dt * d_solver->v_rot()(tid * 3 + 0);
```

---

## 10. Crack-Distance Visualisation — Projecting onto Sub-Facets

### Theory

After each time step the per-edge damage variable ω ∈ [0, 1] and the maximum
effective-strain history κ together define a physical **crack opening distance**
for each interaction strut:

```
crack_distance = ω · κ · l₀    [mm]
```

where l₀ is the reference edge length. This quantity has a clear physical meaning:
it is the width of the opened crack at the strut's Voronoi facet, in millimetres.
For intact facets (ω = 0) it is zero; for a fully failed facet it equals κ · l₀.

To visualise where cracks have opened, the crack distance must be mapped back onto
the Voronoi **sub-facet** triangle mesh, because these triangles are what a human
observer associates with physical crack surfaces.

### Implementation

The `subfacet→edge` lookup table `d_subfacet_edge_idx[]` is built during
`SetupFromMesh()`. A GPU kernel projects the crack distance from the coarser edge
lattice onto the finer sub-facet mesh:

```cpp
// LDPMTet4Data.cu — project_edge_crack_distance_to_subfacets_kernel
__global__ void project_edge_crack_distance_to_subfacets_kernel(
        int n_subfacet,
        const int* __restrict__ subfacet_edge_idx,
        const Real* __restrict__ edge_omega,
        const Real* __restrict__ edge_kappa,
        const Real* __restrict__ edge_l0,
        Real* __restrict__ subfacet_crack_dist) {
    const int sf = blockIdx.x * blockDim.x + threadIdx.x;
    if (sf >= n_subfacet) return;
    const int eidx = subfacet_edge_idx[sf];
    subfacet_crack_dist[sf] = (eidx >= 0)
        ? edge_omega[eidx] * edge_kappa[eidx] * edge_l0[eidx]
        : Real(0);
}
```

The result is a flat array of length `n_subfacet` with values ≥ 0 in mm. Unmatched
sub-facets (entries where `subfacet_edge_idx == -1`) receive a value of 0.

```cpp
// Retrieve to host, then write VTK field "crack_distance" [mm]
VectorXR sf_crack;
element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
WriteLDPMTet4SubfacetMeshToVTK(filename, mesh, "crack_distance", sf_crack);
```

---

## 11. Complete Workflow Summary

```
Chrono Workbench preprocessor
    ↓  6 data files:
       *-data-nodes.dat / particles.dat / tets.dat
       facets.dat / faceFacets.dat / facetsVertices.dat
ReadLDPMTet4MeshFromFiles()
    ↓
GPU_LDPMTet4_Data::SetupFromMesh()
    ├─ Setup():
    │    ├─ Extract n_edge unique particle struts from TET4 connectivity
    │    │    (std::map<pair<int,int>, vector<int>>  keyed on sorted node pair)
    │    ├─ Compute l₀, n̂, m̂, l̂ (local facet frame) per edge
    │    ├─ Compute facet area A per edge (Voronoi sub-triangle sum)
    │    └─ Allocate GPU arrays:
    │         positions (x,y,z), rotations (θx,θy,θz),
    │         forces (f_int, f_ext), moments (f_int_r, f_ext_r),
    │         traction cache (facet_t, 6 × n_edge),
    │         damage history: d_kappa[], d_omega[]  ← initialised to 0
    └─ Override A, m̂, l̂ from facets.dat sub-facet data (richer geometry)
         Build subfacet→edge index table d_subfacet_edge_idx[]
    ↓
SetMaterial(E_N, E_T, E_kT, E_kM, E_kL)    ← legacy path
SetDamageParams(σ_t, H_t)                  ← legacy cracking threshold
   ──  OR  ──
SetLDPMParams(params)                      ← full LDPM model (sets use_full_ldpm = true)
SetDensity(ρ)
SetExternalForce(f_ext)
SetNodalFixed(fixed_nodes)
    ↓
CalcMassMatrix()
    → mᵢ = ρ · Vᵢ   (¼-TET Voronoi approximation)
    → Iᵢ = 0.25 · mᵢ · l_min,i²
    ↓
LeapfrogSolver::Setup()
    → copy lumped mass + rotational inertia to GPU
    ↓
─────────────────── Time loop ───────────────────
    │
    ├─ leapfrog_compute_p_kernel  (one GPU thread per edge):
    │    │
    │    ├─ Compute 6 strains from Δu and Δθ projected onto (n, m, l)
    │    │    e_N = (Δu·n)/l₀,  e_M = (Δu·m)/l₀,  e_L = (Δu·l)/l₀
    │    │    κ_T = (Δθ·n)/l₀,  κ_M = (Δθ·m)/l₀,  κ_L = (Δθ·l)/l₀
    │    │
    │    └─ Constitutive dispatch:
    │
    │         ═══ IF use_full_ldpm (SetLDPMParams path) ═══
    │
    │         ldpm_tet4_full_constitutive():
    │           if eps_N > 0:
    │             sigma_fr = ldpm_fracture_boundary()   ← mode-mixity softening
    │             Project sigma_fr onto (N, M, L) via eps_Q
    │           else:
    │             sigma_N = ldpm_compress_boundary()    ← yield/harden/compact
    │             (sigma_M, sigma_L) = ldpm_shear_boundary()  ← friction
    │           Update energy, crack opening, and 16-component state vector
    │           Maintain legacy kappa/omega for VTK
    │
    │         ═══ ELSE (legacy SetMaterial/SetDamageParams path) ═══
    │
    │         ldpm_tet4_cusatis_traction():
    │           e_D    = sqrt(⟨e_N⟩₊² + α(e_M² + e_L²))
    │           κ_new  = max(κ_old, e_D)            ← irreversible history
    │           e_t0   = σ_t / E_N                  ← cracking threshold
    │           ω = damage from exponential softening
    │           t_N = E_N·e_N·(1−ω) if e_N > 0, else E_N·e_N  (asymmetric)
    │           t_M = E_T·e_M·(1−ω)
    │           t_L = E_T·e_L·(1−ω)
    │           m_T = E_kT·κ_T,  m_M = E_kM·κ_M,  m_L = E_kL·κ_L  (elastic)
    │
    │         Write d_facet_t[edge × 6 + {0..5}]
    │
    ├─ clear f_int and f_int_r
    │
    ├─ leapfrog_compute_f_int_kernel  (one thread per (edge, local_node)):
    │    f_node  += ±A · (t_N·n + t_M·m + t_L·l)   [atomicAdd]
    │    M_node  += ±A · (m_T·n + m_M·m + m_L·l)   [atomicAdd]
    │
    ├─ leapfrog_update_velocity_kernel:
    │    v_{n+½}  += Δt · M_lump⁻¹ · (f_ext − f_int)
    ├─ leapfrog_update_velocity_rot_kernel:
    │    ω_{n+½} += Δt · I_lump⁻¹ · (M_ext − M_int)
    │
    ├─ leapfrog_apply_bc_ldpm_tet4_kernel:
    │    v = ω = 0  at fixed nodes  (all 6 DOFs)
    │
    ├─ leapfrog_update_position_kernel:
    │    x_{n+1} = xₙ + Δt · v_{n+½}
    └─ leapfrog_update_rotation_kernel:
         θ_{n+1} = θₙ + Δt · ω_{n+½}

    Every T_CSV seconds (default 0.5 ms, ~200 points over 0.1 s):
    ├─ RetrievePositionToCPU() + RetrieveInternalForceToCPU()
    └─ Append one row to CSV:
         displacement_mm,stress_MPa
         <mean z-disp of loaded nodes>, <reaction force / A_cross>

    Every T_VTK seconds (default 5 ms, ~20 frames over 0.1 s):
    ├─ WriteLDPMTet4TetMeshToVTK()
    │    → deformed particle positions on TET4 mesh
    └─ ProjectEdgeCrackDistanceToSubfacets()
         crack_distance[sf] = ω[edge] · κ[edge] · l₀[edge]   [mm]
         WriteLDPMTet4SubfacetMeshToVTK("crack_distance", …)
         → VTK field for ParaView crack-surface visualisation
─────────────────────────────────────────────────
```

---

## 12. Theory-Symbol to Code-Variable Reference Table

| Theory symbol | Physical meaning | Device array / accessor | Source file |
|---|---|---|---|
| **x**ᵢ | Particle position | `d_x12`, `d_y12`, `d_z12` | `LDPMTet4Data.cuh` |
| **θ**ᵢ | Particle rotation | `d_rx12`, `d_ry12`, `d_rz12` | `LDPMTet4Data.cuh` |
| l₀ | Reference edge length | `d_l0[edge]` → `edge_l0(e)` | `LDPMTet4Data.cuh` |
| **n** | Facet unit normal | `d_edge_n[edge*3+k]` → `edge_n_comp(e,k)` | `LDPMTet4Data.cuh` |
| **m** | Facet first tangent | `d_edge_m[edge*3+k]` → `edge_m_comp(e,k)` | `LDPMTet4Data.cuh` |
| **l** | Facet second tangent | `d_edge_lv[edge*3+k]` → `edge_lv_comp(e,k)` | `LDPMTet4Data.cuh` |
| A | Voronoi facet area | `d_facet_area[edge]` → `facet_area(e)` | `LDPMTet4Data.cuh` |
| e_N, e_M, e_L | Translational strains | local vars in `compute_p` | `LDPMTet4DataFunc.cuh` |
| κ_T, κ_M, κ_L | Rotational strains | local vars in `compute_p` | `LDPMTet4DataFunc.cuh` |
| α = E_T/E_N | Shear-normal coupling ratio | `LDPMParams::alpha` or local var | `LDPM.cuh` |
| `LDPMParams` | Full model material parameters (24) | `d_ldpm_params` → `ldpm_params()` | `LDPM.cuh` / `LDPMTet4Data.cuh` |
| `statev[0..15]` | Per-edge 16-component state vector | `d_edge_statev[edge*16+k]` → `edge_statev(e,k)` | `LDPMTet4Data.cuh` |
| `use_full_ldpm` | Flag selecting full vs legacy model | `d_use_full_ldpm` → `use_full_ldpm()` | `LDPMTet4Data.cuh` |
| e_D | Damage driving strain (legacy) | local var `e_D` | `LDPM.cuh` |
| σ_t | Mesoscale tensile strength | `LDPMParams::sigma_t` or `*d_sigma_t` | `LDPM.cuh` / `LDPMTet4Data.cuh` |
| E_N | Normal modulus | `LDPMParams::E0` or `*d_E_N` | `LDPM.cuh` / `LDPMTet4Data.cuh` |
| e_t0 = σ_t/E_N | Cracking threshold strain (legacy) | local var `e_t0` | `LDPM.cuh` |
| κ | History max strain (ratchet) | `d_kappa[edge]` → `edge_kappa(e)` | `LDPMTet4Data.cuh` |
| ω | Damage / fracture state (0=intact, 1=failed) | `d_omega[edge]` → `edge_omega(e)` | `LDPMTet4Data.cuh` |
| H_t | Softening modulus (legacy) | `*d_H_t` → `H_t()` | `LDPMTet4Data.cuh` |
| t_N, t_M, t_L | Facet tractions | `d_facet_t[edge*6+0..2]` → `facet_t(e,0..2)` | `LDPMTet4Data.cuh` |
| m_T, m_M, m_L | Facet moments | `d_facet_t[edge*6+3..5]` → `facet_t(e,3..5)` | `LDPMTet4Data.cuh` |
| **f**_int | Translational internal force | `d_f_int_t[node*3+k]` → `f_int()(node*3+k)` | `LDPMTet4Data.cuh` |
| **M**_int | Rotational internal moment | `d_f_int_r[node*3+k]` → `f_int_r()(node*3+k)` | `LDPMTet4Data.cuh` |
| mᵢ | Translational lumped mass | `d_csr_values[i]` → `mass_lump()(i)` | `LeapfrogSolver.cuh` |
| Iᵢ | Rotational inertia | `d_I_lump[i]` → `mass_inertia()(i)` | `LeapfrogSolver.cuh` |
| crack_distance | Physical crack opening width [mm]: ω·κ·l₀ | computed in `ProjectEdgeCrackDistanceToSubfacets` | `LDPMTet4Data.cu` |
| T_VTK | VTK output interval | `static constexpr Real T_VTK` (default 0.005 s) | dogbone demos |
| T_CSV | CSV output interval | `static constexpr Real T_CSV` (default 0.0005 s) | dogbone demos |
