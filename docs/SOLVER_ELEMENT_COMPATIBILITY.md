# Solver–Element Compatibility Matrix

This document is the authoritative reference for which element types are
supported by each solver class.  **Any time a new solver or element is added,
or an existing pairing is enabled/disabled, this file must be updated to
reflect the change.**  See `AGENTS.md` for the full rule set that governs
modifications to this repository.

> **Note on physical formulations**: Not all elements use the same physics.
> FEAT4 and FEAT10 implement **Total Lagrangian (TL) continuum FEA**.
> ANCF3243 and ANCF3443 implement **ANCF large-deformation structural mechanics**
> (a distinct formulation — not classical TL).
> LDPMTet4 implements **Lattice Discrete Particle Model (LDPM)** — a mesoscale
> particle method that is **not FEA at all**.  See `docs/ARCHITECTURE.md` for
> the full physical description of each element type.

---

## Element Types

| Enum value | Class | Shape | DOFs per node | Physics description |
|---|---|---|---|---|
| `TYPE_3243` | `GPU_ANCF3243_Data` | ANCF cable, 3-node | 8 generalized coords | Flexible cable / slender beam: large deformation, large rotation continuum beam using Absolute Nodal Coordinate Formulation (3 position + 5 slope DOFs per node) |
| `TYPE_3443` | `GPU_ANCF3443_Data` | ANCF shell, 4-node | 16 generalized coords | Flexible thin shell: large deformation, large rotation plate/shell using Absolute Nodal Coordinate Formulation (3 position + 13 slope DOFs per node) |
| `TYPE_T4` | `GPU_FEAT4_Data` | TET4, 4-node | 3 translational | Continuum nonlinear solid mechanics using linear tetrahedral elements; St. Venant–Kirchhoff (SVK) or Mooney–Rivlin hyperelastic material |
| `TYPE_T10` | `GPU_FEAT10_Data` | TET10, 10-node | 3 translational | Continuum nonlinear solid mechanics using quadratic tetrahedral elements; SVK or Mooney–Rivlin hyperelastic material |
| `TYPE_LDPM_TET4` | `GPU_LDPMTet4_Data` | TET4 Delaunay lattice, 6-DOF particles | 3 translational + 3 rotational | Lattice Discrete Particle Model (LDPM): mesoscale particle interaction on Voronoi facets; tensile-shear damage constitutive law suitable for concrete-like materials |

---

## Solver Classes

### `LinearStaticSolver<TData>` (template)

**Physics**: Linear quasi-static (small-strain) equilibrium — solves `K·u = f_ext`
once via a GPU conjugate-gradient loop (cuBLAS + cuSPARSE).  Valid only for
small displacements; does not account for geometric nonlinearity.

| Element type | Supported? | Notes |
|---|---|---|
| `TYPE_T10` (`GPU_FEAT10_Data`) | ✅ | Convenience alias: `LinearStaticSolverT10` |
| `TYPE_T4`  (`GPU_FEAT4_Data`)  | ✅ | Convenience alias: `LinearStaticSolverT4` |
| `TYPE_3243` | ❌ | ANCF beam elements are not continuum solid; no tangent stiffness kernel |
| `TYPE_3443` | ❌ | ANCF shell elements are not continuum solid; no tangent stiffness kernel |
| `TYPE_LDPM_TET4` | ❌ | LDPM is a discrete lattice model; no global stiffness matrix |

*`LinearStaticSolver` is a class template, so unsupported types are rejected at
compile time rather than runtime.*

---

### `SyncedNesterovSolver`

**Physics**: Quasi-static / dynamic equilibrium via Nesterov accelerated
gradient descent (augmented Lagrangian for constraints).  Suited for
continuum FEA and ANCF elements where an energy landscape exists.

| Element type | Supported? | Notes |
|---|---|---|
| `TYPE_3243` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_3443` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_T10`  | ✅ | |
| `TYPE_T4`   | ✅ | |
| `TYPE_LDPM_TET4` | ❌ | LDPM uses explicit dynamic integration; use `LeapfrogSolver` |

---

### `SyncedAdamWSolver`

**Physics**: Quasi-static / dynamic equilibrium via AdamW first-order optimizer
(augmented Lagrangian for constraints, cooperative GPU launch).  Suited for
continuum FEA and ANCF elements.

| Element type | Supported? | Notes |
|---|---|---|
| `TYPE_3243` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_3443` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_T10`  | ✅ | |
| `TYPE_T4`   | ✅ | |
| `TYPE_LDPM_TET4` | ❌ | LDPM uses explicit dynamic integration; use `LeapfrogSolver` |

---

### `SyncedAdamWNocoopSolver`

**Physics**: Same as `SyncedAdamWSolver` but uses a non-cooperative GPU launch
strategy (no `cudaLaunchCooperativeKernel`).  Keeps biased first/second moment
vectors (AdamW) and supports an optional pre-computed constraint Jacobian pointer.

| Element type | Supported? | Notes |
|---|---|---|
| `TYPE_3243` | ✅ | Calls `CalcDsDuPre()` on setup; constraint pointer retrieved if constraints are set up |
| `TYPE_3443` | ✅ | Calls `CalcDsDuPre()` on setup; constraint pointer retrieved if constraints are set up |
| `TYPE_T10`  | ✅ | Constraint pointer retrieved if constraints are set up |
| `TYPE_T4`   | ✅ | Constraint pointer retrieved if constraints are set up |
| `TYPE_LDPM_TET4` | ❌ | LDPM uses explicit dynamic integration; use `LeapfrogSolver` |

---

### `LeapfrogSolver`

**Physics**: Explicit central-difference (leapfrog / velocity-Verlet) time
integration for nonlinear solid mechanics and LDPM particle dynamics.
Advances the simulation by one time step per `Solve()` call.

For `TYPE_LDPM_TET4` both translational and rotational DOFs are integrated;
the velocity buffer is `6·n_coef` and the mass buffer is `2·n_coef`.

| Element type | Supported? | Notes |
|---|---|---|
| `TYPE_3243` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_3443` | ✅ | Calls `CalcDsDuPre()` on setup |
| `TYPE_T10`  | ✅ | |
| `TYPE_T4`   | ✅ | |
| `TYPE_LDPM_TET4` | ✅ | **Primary use case.** 6-DOF leapfrog with rotational inertia |

---

## Material Models

The material model is set on the element (not the solver) before calling `Solve()`.

| Enum | Class | Applicable elements | Description |
|---|---|---|---|
| `MATERIAL_MODEL_SVK` | `SVK.cuh` | `TYPE_T4`, `TYPE_T10` | St. Venant–Kirchhoff (SVK) hyperelastic — suitable for moderate strains |
| `MATERIAL_MODEL_MOONEY_RIVLIN` | `MooneyRivlin.cuh` | `TYPE_T4`, `TYPE_T10` | Mooney–Rivlin hyperelastic — suitable for large-strain rubber-like materials |
| LDPM facet law | `LDPM.cuh` | `TYPE_LDPM_TET4` | Cusatis LDPM tensile-shear coupled damage — suitable for concrete-like materials |

---

## Quick-Reference Table (all combinations)

|  | LinearStatic | SyncedNesterov | SyncedAdamW | SyncedAdamWNocoop | Leapfrog |
|---|---|---|---|---|---|
| `TYPE_3243` | ❌ | ✅ | ✅ | ✅ | ✅ |
| `TYPE_3443` | ❌ | ✅ | ✅ | ✅ | ✅ |
| `TYPE_T4`   | ✅ | ✅ | ✅ | ✅ | ✅ |
| `TYPE_T10`  | ✅ | ✅ | ✅ | ✅ | ✅ |
| `TYPE_LDPM_TET4` | ❌ | ❌ | ❌ | ❌ | ✅ |
