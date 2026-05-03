# Examples

The examples are split into two categories:

## `beam_simulations/` — Cantilever-beam problems

A single cantilever-beam geometry exercised with every element type and solver.
Useful for cross-comparing physical models and solver strategies.

| Executable | Element | Solver | Notes |
|---|---|---|---|
| `beam_simulation` | FEAT10 (TET10) | SyncedNesterov | Dynamic quasi-static |
| `beam_linear_static` | FEAT10 (TET10) | Linear (cuSPARSE) | Static equilibrium |
| `beam_linear_static_t4` | FEAT4 (TET4) | Linear (cuSPARSE) | Static equilibrium |
| `test_leapfrog_t4` | FEAT4 (TET4) | Leapfrog | Explicit dynamic |
| `test_ldpm_tet4` | LDPMTet4 (6-DOF) | Leapfrog | **Recommended LDPM example** |

See [`README_BEAM.md`](beam_simulations/README_BEAM.md) for mesh details and
[`README_LDPM_TET4.md`](beam_simulations/README_LDPM_TET4.md) for the 6-DOF
LDPM element documentation.

## `solver_tests/` — Solver capability tests

Focused tests for the gradient-based iterative solvers on a simple solid block,
independent of any specific geometry.

| Executable | Element | Solver |
|---|---|---|
| `test_feat10_nesterov` | FEAT10 | SyncedNesterov |
| `test_feat10_adamw` | FEAT10 | SyncedAdamW |

## Why results differ between examples

Different examples intentionally model different physics:

- **LDPM (LDPMTet4 / Leapfrog)** — discrete particle lattice, explicit dynamic; shows wave propagation / transient response
- **Linear static** — static equilibrium solution, no dynamics
- **Nesterov / AdamW** — quasi-static approach to equilibrium using continuum FEA
- **FEAT4 Leapfrog** — continuum TET4 element, same explicit dynamic scheme as LDPM but different physics (continuum vs. lattice)

These differences are expected and correct.
