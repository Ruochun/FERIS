# TLFEA Architecture

## Overview

TLFEA (Total Lagrangian Finite Element Analysis) is a GPU-accelerated FEA / particle-mechanics framework for solid mechanics and discrete particle simulations.

## Design Philosophy

- **Multiple element types**: continuum FEA (FEAT4, FEAT10) and lattice particle (LDPM)
- **Multiple solver strategies**: quasi-static iterative (Nesterov, AdamW), explicit dynamic (Leapfrog), and static linear (cuSPARSE)
- **Clean architecture**: each element type is self-contained; solvers are generic over element types via type-dispatch

## Directory Structure

```
TLFEA/
├── src/
│   ├── elements/       # Finite element / particle element implementations
│   │   ├── ElementBase.h              # Abstract base; ElementType enum
│   │   ├── FEAT4Data.cu/cuh           # TET4 continuum element (4-node)
│   │   ├── FEAT4DataFunc.cuh          # Device functions for FEAT4
│   │   ├── FEAT10Data.cu/cuh          # TET10 continuum element (10-node)
│   │   ├── FEAT10DataFunc.cuh         # Device functions for FEAT10
│   │   ├── ANCF3243Data.cu/cuh        # ANCF cable element (3-node)
│   │   ├── ANCF3443Data.cu/cuh        # ANCF beam element (4-node)
│   │   ├── LDPM4Data.cu/cuh           # [DEPRECATED] 3-DOF LDPM on TET4
│   │   ├── LDPM4DataFunc.cuh          # [DEPRECATED] Device functions
│   │   ├── LDPMTet4Data.cu/cuh        # 6-DOF LDPM on TET4 (recommended)
│   │   └── LDPMTet4DataFunc.cuh       # Device functions for LDPMTet4
│   │
│   ├── solvers/        # Solvers
│   │   ├── SolverBase.h               # Abstract base
│   │   ├── SyncedNesterov.cu/cuh      # Nesterov accelerated gradient solver
│   │   ├── SyncedAdamW.cu/cuh         # AdamW optimizer-based solver
│   │   └── LeapfrogSolver.cu/cuh      # Explicit central-difference integrator
│   │
│   ├── materials/      # Constitutive models
│   │   ├── SVK.cuh                    # St. Venant-Kirchhoff hyperelastic
│   │   ├── LDPM.cuh                   # LDPM 3-DOF facet law (deprecated)
│   │   └── LDPMTet4.cuh               # LDPM 6-DOF facet + moment law
│   │
│   └── utils/          # Utilities
│       ├── types.h                     # Real, MatrixXR, VectorXR
│       ├── cpu_utils.h/cc             # Mesh I/O
│       ├── cuda_utils.h               # CUDA error-check macros
│       ├── quadrature_utils.h         # Gauss quadrature
│       └── mesh_utils.h/cc            # Mesh manipulation
│
├── examples/                           # Example programs
│   ├── beam_simulations/               # Cantilever-beam examples (all solvers)
│   │   ├── beam_simulation.cc          # TET10, Nesterov (dynamic)
│   │   ├── beam_linear_static.cc       # TET10, linear static
│   │   ├── beam_linear_static_t4.cc    # TET4,  linear static
│   │   ├── test_leapfrog_t4.cc         # TET4,  Leapfrog (dynamic)
│   │   ├── test_ldpm_t4.cc             # LDPM4, Leapfrog [DEPRECATED element]
│   │   ├── test_ldpm_tet4.cc           # LDPMTet4, Leapfrog (recommended)
│   │   └── README_BEAM.md
│   └── solver_tests/                   # Focused solver tests
│       ├── test_feat10_nesterov.cc     # FEAT10 + Nesterov
│       └── test_feat10_adamw.cc        # FEAT10 + AdamW
│
├── data/               # Test meshes
│   └── meshes/
│       ├── T4/                         # TET4 beam meshes
│       └── T10/                        # TET10 beam meshes
│
└── docs/               # Documentation
    ├── ARCHITECTURE.md  (this file)
    ├── BUILDING.md
    ├── GETTING_STARTED.md
    └── TROUBLESHOOTING.md
```

## Element Types

### `FEAT4` / `FEAT10` — Continuum FEA

TET4 (4-node) and TET10 (10-node) tetrahedral elements using a St. Venant-Kirchhoff
(SVK) hyperelastic material law.  Each node carries 3 translational DOFs.
Best suited for problems where continuum stress and deformation fields are of interest.

### `LDPMTet4` (TYPE_LDPM_TET4) — 6-DOF Lattice Discrete Particle Model *(recommended)*

Implements the Lattice Discrete Particle Model on a TET4 Delaunay triangulation.
Every unique edge (i, j) of the mesh is one discrete "strut" with an associated
Voronoi facet area.

**DOF layout** — 6 DOFs per particle:
- Translational: `x, y, z`
- Rotational (linearised): `rx (θ_x), ry (θ_y), rz (θ_z)`

**Constitutive law** (`src/materials/LDPMTet4.cuh`) — 5 moduli:

| Strain | Formula | Traction / moment | Modulus |
|--------|---------|-------------------|---------|
| Normal | `e_N = (Δu · n) / l₀` | `t_N = E_N · e_N` | `E_N` |
| Shear m | `e_M = (Δu · m) / l₀` | `t_M = E_T · e_M` | `E_T` |
| Shear l | `e_L = (Δu · l) / l₀` | `t_L = E_T · e_L` | `E_T` |
| Twist | `κ_T = (Δθ · n) / l₀` | `m_T = E_κT · κ_T` | `E_κT` |
| Bending m | `κ_M = (Δθ · m) / l₀` | `m_M = E_κM · κ_M` | `E_κM` |
| Bending l | `κ_L = (Δθ · l) / l₀` | `m_L = E_κL · κ_L` | `E_κL` |

where `Δu = u_j − u_i` (translational displacement difference) and
`Δθ = θ_j − θ_i` (rotation vector difference), both projected onto the
reference facet frame `(n, m, l)`.

**Rotational inertia**: `I_lump = α · m_lump · l_min²` with `α = 0.25` by default
(configurable via `LDPM_TET4_ALPHA_ROT`).

**LeapfrogSolver integration**:
- Velocity array layout: `[v_trans (3·n) | v_rot (3·n)]` (6·n total)
- Mass array layout: `[m_trans (n) | I_rot (n)]` (2·n total)
- New kernels: `leapfrog_update_velocity_rot`, `leapfrog_apply_bc_ldpm_tet4`,
  `leapfrog_update_rotation`, `leapfrog_compute_lumped_mass_ldpm_tet4`

### `LDPM4` (TYPE_LDPM4_DEPRECATED) — 3-DOF LDPM *(deprecated)*

Original 3-translational-DOF LDPM.  Superseded by `LDPMTet4`.  Retained for
backward-compatibility only; may be removed in a future release.

## Solvers

### `LeapfrogSolver` — Explicit Central-Difference Integrator

The preferred solver for dynamic LDPM simulations.  Advances the state by one
time step using the velocity-Verlet (leapfrog) scheme:

```
v_{n+½} = v_{n−½} + dt · M_lump⁻¹ · (f_ext − f_int)
x_{n+1} = x_n    + dt · v_{n+½}
```

For `TYPE_LDPM_TET4` the rotational DOFs are integrated with the same scheme
using the per-node rotational inertia `I_lump` in place of `m_lump`.

CFL stability condition (translational):
```
dt_crit = l_min / c_N   where   c_N = √(E_N / ρ)
```
Recommended safety factor: 0.5.

### `SyncedNesterov` / `SyncedAdamW` — Quasi-static/Dynamic Optimizers

First-order momentum-based iterative solvers.  Suitable for continuum FEA
elements (FEAT4, FEAT10) where energy minimisation converges to equilibrium.
Not designed for LDPM lattice models.

### Linear Static Solver (cuSPARSE)

Solves `K · u = f` directly using the sparse linear solver from cuSPARSE.
Gives the exact static equilibrium solution in a single solve.

## Why LDPM Results Differ from Linear/AdamW Solvers

LDPM (on any solver) and continuum FEA (FEAT4/FEAT10) are **different physical
models** intentionally:

| Aspect | LDPM (LDPMTet4) | Continuum FEA (FEAT4/FEAT10) |
|--------|-----------------|------------------------------|
| Discretisation | Discrete particle lattice | Continuum field |
| Interaction law | Facet tractions on Voronoi cells | Gauss-point stress |
| Rotations | Explicit particle rotations (θ) | Embedded in displacement gradient |
| Solver | Leapfrog (explicit dynamic) | Nesterov/AdamW (quasi-static) or linear static |

Additionally, Leapfrog shows **transient wave propagation** (oscillatory response),
whereas a linear static solver shows the **equilibrium state** and Nesterov/AdamW
perform a quasi-static approach to equilibrium.  Different results are expected
and correct.

## GPU Acceleration

All element and solver kernels run on the GPU via CUDA:
- **Grid-stride loops** for element-level parallelism
- **atomicAdd** for thread-safe nodal force assembly (LDPM)
- **cuSPARSE / cuBLAS** for the linear static solver
- No cooperative groups required for the Leapfrog kernels

## Extension Points

1. **New element type**: inherit `ElementBase`, implement virtual methods,
   add a `TYPE_*` enum value, add dispatch branches in `LeapfrogSolver`
2. **New solver**: inherit `SolverBase`, implement `Solve()`
3. **New material**: add a `__device__ __forceinline__` function in `src/materials/`,
   call it from the element's `compute_p` device function

## References

- LDPM: Cusatis, G., Pelessone, D., Mencarelli, A. (2011). "Lattice Discrete
  Particle Model (LDPM) for failure behaviour of concrete." *Cement and Concrete
  Composites*, 33(9), 881–890.
- LeapFrog / velocity-Verlet: Verlet, L. (1967). "Computer 'Experiments' on
  Classical Fluids." *Physical Review*, 159(1), 98–103.
- St. Venant-Kirchhoff: standard nonlinear elasticity reference
- Nesterov acceleration: Nesterov, Y. (1983). "A method for solving the convex
  programming problem with convergence rate O(1/k²)"
