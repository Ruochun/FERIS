# LDPM-TET4 Gap Roadmap

> **Purpose:** Track the implementation status of the full LDPM model in FERIS.

---

## Status: FULLY IMPLEMENTED

The full LDPM constitutive model has been implemented in FERIS, closing all gaps
identified below. The implementation matches the CPU reference in chrono-mechanics
(`src/chrono_ldpm/ChMaterialVECT.cpp` and `ChElementLDPM.cpp`).

All refinements are complete, including per-sub-facet kinematics and volumetric
strain mapping. Each loaded Workbench sub-facet retains its own center, frame,
area, state, and owning-TET `eps_V = (V_cur - V_ref) / (3 * V_ref)`, matching
`ChElementLDPM`.

---

## 1. Current State vs Target State

| Area | Status | Implementation |
|---|---|---|
| Elasticity | ✅ Done | `E_0`, `alpha` in `LDPMParams`; `E_k*` retained for API compatibility |
| Tension | ✅ Done | Mode-mixity dependent softening via `ldpm_fracture_boundary()` |
| Compression | ✅ Done | Yielding, hardening, pore collapse via `ldpm_compress_boundary()` |
| Shear under compression | ✅ Done | Pressure-dependent friction via `ldpm_shear_boundary()` |
| Facet state | ✅ Done | 16-component per-interaction state vector (`LDPM_N_STATEV`) |
| Rotational response | ✅ Done | Elastic only (unchanged, first-order LDPM) |
| Backward compatibility | ✅ Done | Legacy `ldpm_tet4_cusatis_traction()` removed (superseded by full model) |

---

## 2. Gap 1 — Parameter Model (CLOSED)

Full `LDPMParams` struct with:
- Elastic: `E0`, `alpha`, `rho` (`E_k*` retained for API compatibility)
- Tensile: `sigma_t`, `sigma_s`, `n_t`, `l_t`, `r_s`, `k_t`
- Compression: `sigma_c0`, `H_c0`, `H_c1`, `kc0`, `kc1`, `kc2`, `kc3`, `beta`, `E_d`
- Friction: `mu_0`, `mu_inf`, `sigma_N0`
- Control: `elastic_flag`

Set via `GPU_LDPMTet4_Data::SetLDPMParams(const LDPMParams& params)`.

---

## 3. Gap 2 — Facet History Variables (CLOSED)

Per-interaction state vector with 16 components tracking:
- Accumulated strains [0-2]
- Current stresses [3-5]
- History maxima [6-7]
- Effective strain/stress [8-9]
- Energy quantities [10, 15]
- Crack opening [11]
- Reserved for eigenstrain [12-14]

---

## 4. Gap 3 — Constitutive Law Branching (CLOSED)

`ldpm_tet4_full_constitutive()` in `LDPM.cuh` implements:
1. Tensile/fracture branch when `eps_N > 0`
2. Compressive boundary when `eps_N <= 0`
3. Frictional shear boundary when `eps_N <= 0`

Branching logic mirrors `ChMaterialVECT::ComputeStress()` from chrono-mechanics.

---

## 5. Gap 4 — Rotational Response (CLOSED — No Change Needed)

Rotational nodal residuals now come from Chrono-style facet tractions acting
through reference facet-center lever arms; no separate `m = E_k * kappa`
couple-stress law is used in the Chrono-matching path.

---

## 6. Gap 5 — Output and Verification (CLOSED)

- Effective strain/stress exposed via state vector
- Crack opening displacement computed per interaction
- Legacy kappa/omega maintained for VTK visualization
- 7-case single-tet benchmark updated with full LDPM parameters

---

## 7. Remaining Refinements (Low Priority)

| Item | Description | Status |
|---|---|---|
| Volumetric strain mapping | Compute `eps_V = (V_cur - V_ref) / (3 * V_ref)` per TET and map it to each owned sub-facet interaction | ✅ Done |

Each step computes the true volumetric strain per TET element. Workbench
sub-facet interactions use their owning TET's value, matching the CPU reference
`ChElementLDPM`; generic `Setup()` unique-edge interactions retain a sharing-TET
average as a fallback.

---

## Literature basis

1. Cusatis, G., Pelessone, D., & Mencarelli, A. (2011). *Lattice Discrete Particle Model (LDPM) for failure behavior of concrete. I: Theory*. Cement and Concrete Composites, 33(9), 881-890.
2. chrono-mechanics (base_dev): `src/chrono_ldpm/ChMaterialVECT.cpp`
