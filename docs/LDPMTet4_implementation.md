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
5. [Strain Computation — compute_ldpm_facet_strain_and_stress](#5-strain-computation--compute_ldpm_facet_strain_and_stress)
6. [Constitutive Law — Full LDPM Model](#6-constitutive-law)
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

The implementation uses the **full LDPM constitutive model** (activated via `SetLDPMParams`):
tensile fracture with mode-mixity, compressive yielding/hardening/densification, and
pressure-dependent friction.

This implementation runs the full LDPM workflow on the GPU. The main types involved are:

| Type | File | Role |
|---|---|---|
| `GPU_LDPMTet4_Data` | `LDPMTet4Data.cuh` / `LDPMTet4Data.cu` | Stores all per-particle and per-interaction GPU arrays; owns Setup/Teardown |
| `LDPMParams` | `LDPM.cuh` | Struct holding all 24 material parameters for the full model |
| `ldpm_tet4_full_constitutive` | `LDPM.cuh` | Device function: full LDPM constitutive update (fracture + compression + friction) |
| `ldpm_fracture_boundary` | `LDPM.cuh` | Device function: tensile-shear fracture with mode-mixity softening |
| `ldpm_compress_boundary` | `LDPM.cuh` | Device function: compressive yielding / hardening / pore collapse |
| `ldpm_shear_boundary` | `LDPM.cuh` | Device function: pressure-dependent frictional shear limit |
| `compute_ldpm_facet_strain_and_stress` | `LDPMTet4DataFunc.cuh` | Device function: strains → constitutive update → tractions for one interaction |
| `compute_internal_force` | `LDPMTet4DataFunc.cuh` | Device function: scatter tractions to nodal forces |
| `LeapfrogSolver` | `LeapfrogSolver.cu` | Explicit time integrator; dispatches all CUDA kernels |

---

## 2. Mesh Discretisation — Particles, TET4, and Sub-facet Interactions

### Theory

LDPM places aggregate particles at random positions and connects them by a 3-D
Delaunay triangulation (TET4 mesh). Each TET contributes 12 sub-facet
interactions: two for each of its six local particle-particle edges. Every
sub-facet keeps its own center, area, local frame, material state, and owning-TET
volumetric strain, matching Chrono's `ChElementLDPM` sections.

### Implementation

`GPU_LDPMTet4_Data::Setup()` iterates over every TET element and all six possible
local edges `{(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)}`. A
`std::map<pair<int,int>, vector<int>>` keyed by sorted node pairs accumulates which
TET elements share each unique edge:

```cpp
// LDPMTet4Data.cu — Setup(), step 5
constexpr int LDPM_TET4_LOCAL_EDGES[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

std::map<std::pair<int,int>, std::vector<int>> edge_to_tets;
for (int e = 0; e < n_elem; e++) {
    for (int le = 0; le < 6; le++) {
        int a = tet_connectivity(e, LDPM_TET4_LOCAL_EDGES[le][0]);
        int b = tet_connectivity(e, LDPM_TET4_LOCAL_EDGES[le][1]);
        if (a > b) std::swap(a, b);
        edge_to_tets[{a, b}].push_back(e);
    }
}
n_edge = static_cast<int>(edge_to_tets.size());
```

After this loop, `Setup()` has identified unique particle-particle struts for
the generic geometry fallback. When `SetupFromMesh()` has `facets.dat` data, it
replaces those fallback interactions with one interaction per sub-facet row.
The legacy member `n_edge` and `get_n_beam()` therefore represent the active
interaction count, which is `n_subfacet` for a loaded Workbench mesh.

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
per-interaction GPU arrays. It is **never updated** during the simulation, consistent
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
__device__ Map<VectorXR> x_cur()     { return Map<VectorXR>(d_x_cur,  n_nodes); }
__device__ Map<VectorXR> rot_x_cur() { return Map<VectorXR>(d_rot_x_cur, n_nodes); }  // θ_x

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

## 5. Strain Computation — compute_ldpm_facet_strain_and_stress

### Theory

For each strut (i, j), the engineering strains are projections of the
**relative displacement at the interaction facet center** Δ**u**_c and the
**rotation difference** Δ**θ** onto the three facet frame directions (n, m, l):

| Strain | Formula | Physical meaning |
|---|---|---|
| e_N | (Δu_c · **n**) / l₀ | Normal opening (+) / closing (−) |
| e_M | (Δu_c · **m**) / l₀ | Shear sliding, m-direction |
| e_L | (Δu_c · **l**) / l₀ | Shear sliding, l-direction |

where Δ**u**_c = (**u**ⱼ + **θ**ⱼ × **r**ⱼ) − (**u**ᵢ + **θ**ᵢ × **r**ᵢ)
is the relative displacement of the two particles at the facet center.  The
vectors **r**ᵢ and **r**ⱼ point from each endpoint's reference position to the
interaction facet center, matching Chrono's LDPM `A`-matrix kinematics.

### Implementation

`compute_ldpm_facet_strain_and_stress()` in `LDPMTet4DataFunc.cuh` is called once per edge per time step
(via the `compute_p` solver-template wrapper). The strain projection itself is
centralized in `compute_ldpm_facet_kinematics()`:

```cpp
// LDPMTet4DataFunc.cuh — compute_ldpm_facet_kinematics()
const Real ui_c0 = ui0 + ti1 * ci2 - ti2 * ci1;  // u_i + theta_i x r_i
const Real ui_c1 = ui1 + ti2 * ci0 - ti0 * ci2;
const Real ui_c2 = ui2 + ti0 * ci1 - ti1 * ci0;
const Real uj_c0 = uj0 + tj1 * cj2 - tj2 * cj1;  // u_j + theta_j x r_j
const Real uj_c1 = uj1 + tj2 * cj0 - tj0 * cj2;
const Real uj_c2 = uj2 + tj0 * cj1 - tj1 * cj0;

const Real du0 = uj_c0 - ui_c0;
const Real du1 = uj_c1 - ui_c1;
const Real du2 = uj_c2 - ui_c2;

const Real dt0 = d_data->rot_x_cur()(nj) - d_data->rot_x_cur()(ni);   // Δθ
const Real dt1 = d_data->rot_y_cur()(nj) - d_data->rot_y_cur()(ni);
const Real dt2 = d_data->rot_z_cur()(nj) - d_data->rot_z_cur()(ni);

LDPMFacetKinematics kin;
kin.e_N = (du0*n0 + du1*n1 + du2*n2) * inv_l0;
kin.e_M = (du0*m0 + du1*m1 + du2*m2) * inv_l0;
kin.e_L = (du0*lvec0 + du1*lvec1 + du2*lvec2) * inv_l0;
kin.kappa_T = (dt0*n0 + dt1*n1 + dt2*n2) * inv_l0;
kin.kappa_M = (dt0*m0 + dt1*m1 + dt2*m2) * inv_l0;
kin.kappa_L = (dt0*lvec0 + dt1*lvec1 + dt2*lvec2) * inv_l0;
```

> **Linearised-kinematics note:** All reference geometry (`l0`, `n`, `m`, `l`,
> and the facet-center lever arms) is precomputed once and stored in read-only
> per-interaction arrays. This is *not* the same as a Total-Lagrangian continuum FEA
> formulation (which would use the deformation gradient **F** and Green–Lagrange
> strains); LDPM is a discrete lattice model whose linearised kinematics assume
> small displacements and small rotations.

---

## 6. Constitutive Law

The constitutive law is the core of the failure model.  `compute_ldpm_facet_strain_and_stress()` in
`LDPMTet4DataFunc.cuh` invokes the full LDPM constitutive update via
`ldpm_tet4_full_constitutive()` in `LDPM.cuh`.

### 6.0 Constitutive update in `compute_ldpm_facet_strain_and_stress()`

```cpp
// LDPMTet4DataFunc.cuh
// Strain increments relative to stored accumulated state
const Real d_eps_N = kin.e_N - statev[0];
const Real d_eps_M = kin.e_M - statev[1];
const Real d_eps_L = kin.e_L - statev[2];

// Volumetric strain: precomputed per-tet average (matches CPU reference)
const Real eps_V = d_data->edge_vol_strain(edge_idx);

ldpm_tet4_full_constitutive(d_eps_N, d_eps_M, d_eps_L,
    kin.kappa_T, kin.kappa_M, kin.kappa_L, eps_V, kin.l0,
    d_data->ldpm_params(), statev,
    t_N, t_M, t_L, m_T, m_M, m_L);

// Write back state + maintain kappa/omega for VTK output
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
| Elastic | `E0`, `alpha`, `rho` |
| Tensile | `sigma_t`, `sigma_s`, `n_t`, `l_t`, `r_s`, `k_t` |
| Compression | `sigma_c0`, `H_c0`, `H_c1`, `kc0`, `kc1`, `kc2`, `kc3`, `beta`, `E_d` |
| Friction | `mu_0`, `mu_inf`, `sigma_N0` |
| Control | `elastic_flag` |

Set via `GPU_LDPMTet4_Data::SetLDPMParams(const LDPMParams& params)`.

#### 6.A.2 Per-interaction state vector (16 components)

Each interaction stores a persistent state vector (`LDPM_N_STATEV = 16`):

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

#### 6.A.7 Kappa/omega maintenance for VTK output

The constitutive update also maintains the per-interaction `d_kappa[]` and `d_omega[]`
arrays used for crack-distance visualisation:

```cpp
// LDPMTet4DataFunc.cuh — after full constitutive update
d_data->edge_kappa(edge_idx) = statev[8];  // effective strain as kappa proxy
const Real crack_w = statev[11];
const Real eff_crack = (l0 > 0) ? crack_w / l0 : 0;
d_data->edge_omega(edge_idx) = min(eff_crack / max(statev[8], 1e-30), 1.0);
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

`compute_internal_force()` processes one endpoint of one interaction per thread. The
`node_local` parameter (0 = node i, sign −1; or 1 = node j, sign +1) selects
which end is updated. `atomicAdd` is required because multiple interactions share nodes
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
1.  Compute per-TET volumetric strain and map it onto facet interactions
2.  Compute strains e and κ at all facets  (= compute_ldpm_facet_strain_and_stress)
3.  Apply constitutive law → tractions t, moments m, updated κ and ω
4.  Assemble nodal forces:  f_int, M_int
5.  Velocity update:
      v_{n+½}  = v_{n−½}  + Δt · M_lump⁻¹  · (f_ext  − f_int)
      ω_{n+½} = ω_{n−½} + Δt · I_lump⁻¹  · (M_ext − M_int)
6.  Enforce BCs: zero velocity at fixed nodes (all 6 DOFs)
7.  Position advance:
      x_{n+1} = xₙ + Δt · v_{n+½}
      θ_{n+1} = θₙ + Δt · ω_{n+½}
```

### Implementation

`LeapfrogSolver::OneStepLeapfrog()` for `TYPE_LDPM_TET4` launches seven sequenced
CUDA kernel groups:

```cpp
// LeapfrogSolver.cu — LDPM_TET4 branch

// Step 0: volumetric strain precompute on host-launched LDPM kernels
ldpm_host_data_->ComputeVolumetricStrain();

// Step 1: facet strains → constitutive law → tractions (grid-stride over n_edge)
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
d_data->x_cur()(tid) += dt * d_solver->v()(tid * 3 + 0);

// Rotational:  θ_{n+1} = θₙ + dt · ω_{n+½}
d_data->rot_x_cur()(tid) += dt * d_solver->v_rot()(tid * 3 + 0);
```

---

## 10. Crack-Distance Visualisation — Projecting onto Sub-Facets

### Theory

After each time step the per-interaction damage variable ω ∈ [0, 1] and the maximum
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
    │    ├─ Extract unique-edge fallback interactions from TET4 connectivity
    │    │    (std::map<pair<int,int>, vector<int>>  keyed on sorted node pair)
    │    ├─ Compute l₀, n̂, m̂, l̂ (local facet frame) per edge
    │    ├─ Compute facet area A per edge (Voronoi sub-triangle sum)
    │    └─ Allocate GPU arrays:
    │         positions (x,y,z), rotations (θx,θy,θz),
    │         forces (f_int, f_ext), moments (f_int_r, f_ext_r),
    │         traction cache (facet_t, LDPM_FACET_N_COMPONENTS × n_edge),
    │         damage history: d_kappa[], d_omega[]  ← initialised to 0
    └─ Replace fallback interactions with one interaction per facets.dat row
         Preserve each sub-facet's center, A, p̂, q̂, ŝ, state, and owning TET
    ↓
SetLDPMParams(params)                      ← full LDPM model
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
    ├─ ComputeVolumetricStrain():
    │    ├─ Compute eps_V = (V_cur - V_ref) / (3 * V_ref) per TET
    │    └─ Average per-TET eps_V onto each edge via edge-to-TET CSR mapping
    │
    ├─ leapfrog_compute_p_kernel  (one GPU thread per edge):
    │    │
    │    ├─ Compute 6 strains from Δu and Δθ projected onto (n, m, l)
    │    │    e_N = (Δu·n)/l₀,  e_M = (Δu·m)/l₀,  e_L = (Δu·l)/l₀
    │    │    θ enters Δu_c through θ × r_center
    │    │
    │    └─ Constitutive update (ldpm_tet4_full_constitutive):
    │           if eps_N > 0:
    │             sigma_fr = ldpm_fracture_boundary()   ← mode-mixity softening
    │             Project sigma_fr onto (N, M, L) via eps_Q
    │           else:
    │             sigma_N = ldpm_compress_boundary()    ← yield/harden/compact
    │             (sigma_M, sigma_L) = ldpm_shear_boundary()  ← friction
    │           Update energy, crack opening, and 16-component state vector
    │           Maintain kappa/omega for VTK
    │
    │         Write d_facet_t[edge × LDPM_FACET_N_COMPONENTS + LDPMFacetComponent]
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
| **x**ᵢ | Particle position | `d_x_cur`, `d_y_cur`, `d_z_cur` | `LDPMTet4Data.cuh` |
| **θ**ᵢ | Particle rotation | `d_rot_x_cur`, `d_rot_y_cur`, `d_rot_z_cur` | `LDPMTet4Data.cuh` |
| l₀ | Reference edge length | `d_l0[edge]` → `edge_l0(e)` | `LDPMTet4Data.cuh` |
| **n** | Facet unit normal | `d_edge_n[edge*3+k]` → `edge_n_comp(e,k)` | `LDPMTet4Data.cuh` |
| **m** | Facet first tangent | `d_edge_m[edge*3+k]` → `edge_m_comp(e,k)` | `LDPMTet4Data.cuh` |
| **l** | Facet second tangent | `d_edge_lv[edge*3+k]` → `edge_lv_comp(e,k)` | `LDPMTet4Data.cuh` |
| A | Voronoi facet area | `d_facet_area[edge]` → `facet_area(e)` | `LDPMTet4Data.cuh` |
| e_N, e_M, e_L | Translational strains | `LDPMFacetKinematics` from `compute_ldpm_facet_kinematics` | `LDPMTet4DataFunc.cuh` |
| κ_T, κ_M, κ_L | Rotation-difference placeholders retained for API compatibility | `LDPMFacetKinematics` from `compute_ldpm_facet_kinematics` | `LDPMTet4DataFunc.cuh` |
| α = E_T/E_N | Shear-normal coupling ratio | `LDPMParams::alpha` or local var | `LDPM.cuh` |
| `LDPMParams` | Full model material parameters (24) | `d_ldpm_params` → `ldpm_params()` | `LDPM.cuh` / `LDPMTet4Data.cuh` |
| `statev[0..15]` | Per-edge 16-component state vector | `d_edge_statev[edge*16+k]` → `edge_statev(e,k)` | `LDPMTet4Data.cuh` |
| σ_t | Mesoscale tensile strength | `LDPMParams::sigma_t` | `LDPM.cuh` |
| E_N | Normal modulus | `LDPMParams::E0` | `LDPM.cuh` |
| κ | History max strain (ratchet) | `d_kappa[edge]` → `edge_kappa(e)` | `LDPMTet4Data.cuh` |
| ω | Damage / fracture state (0=intact, 1=failed) | `d_omega[edge]` → `edge_omega(e)` | `LDPMTet4Data.cuh` |
| t_N, t_M, t_L | Facet tractions | `d_facet_t[edge*LDPM_FACET_N_COMPONENTS + LDPM_FACET_T_*]` → `facet_t(e, LDPM_FACET_T_*)` | `LDPMTet4Data.cuh` |
| m_T, m_M, m_L | Facet moments | `d_facet_t[edge*LDPM_FACET_N_COMPONENTS + LDPM_FACET_M_*]` → `facet_t(e, LDPM_FACET_M_*)` | `LDPMTet4Data.cuh` |
| **f**_int | Translational internal force | `d_f_int_t[node*3+k]` → `f_int()(node*3+k)` | `LDPMTet4Data.cuh` |
| **M**_int | Rotational internal moment | `d_f_int_r[node*3+k]` → `f_int_r()(node*3+k)` | `LDPMTet4Data.cuh` |
| mᵢ | Translational lumped mass | `d_csr_values[i]` → `mass_lump()(i)` | `LeapfrogSolver.cuh` |
| Iᵢ | Rotational inertia | `d_I_lump[i]` → `mass_inertia()(i)` | `LeapfrogSolver.cuh` |
| crack_distance | Physical crack opening width [mm]: ω·κ·l₀ | computed in `ProjectEdgeCrackDistanceToSubfacets` | `LDPMTet4Data.cu` |
| T_VTK | VTK output interval | `static constexpr Real T_VTK` (default 0.005 s) | dogbone demos |
| T_CSV | CSV output interval | `static constexpr Real T_CSV` (default 0.0005 s) | dogbone demos |
