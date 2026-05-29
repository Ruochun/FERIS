# LDPM-TET4 Gap Roadmap

> **Purpose:** Describe the gap between the current FERIS LDPM branch and the target modern LDPM implementation, in implementation order.

---

## Table of Contents

1. [Current State vs Target State](#1-current-state-vs-target-state)
2. [Gap 1 — Parameter Model](#2-gap-1--parameter-model)
3. [Gap 2 — Facet History Variables](#3-gap-2--facet-history-variables)
4. [Gap 3 — Constitutive Law Branching](#4-gap-3--constitutive-law-branching)
5. [Gap 4 — Rotational Response and What Stays the Same](#5-gap-4--rotational-response-and-what-stays-the-same)
6. [Gap 5 — Output and Verification](#6-gap-5--output-and-verification)
7. [Suggested Implementation Order](#7-suggested-implementation-order)

---

## 1. Current State vs Target State

| Area | Current branch | Target model |
|---|---|---|
| Elasticity | `E_N`, `E_T`, `E_kT`, `E_kM`, `E_kL` | keep elastic skeleton, but align with canonical `E_0`, `alpha`, and LDPM parameter groups |
| Tension | single scalar tensile/shear damage law | canonical LDPM tensile boundary with mode-mixity terms |
| Compression | purely elastic | compressive yielding, hardening, pore collapse, compaction, densification |
| Shear under compression | only reduced by `omega` | pressure-dependent frictional/shear law |
| Facet state | `kappa`, `omega` only | separate tensile, compressive, and sliding histories |
| Rotational response | elastic only | usually still elastic only in first-order LDPM |
| Sub-facet metadata | mostly stored only | used where needed for fuller LDPM behavior and diagnostics |
| Crack output | scalar proxy `omega * kappa * l0` | opening/sliding measures tied to the actual inelastic update |

---

## 2. Gap 1 — Parameter Model

### Current simplification

The current public constitutive interface has only:

- `E_N`, `E_T`, `E_kT`, `E_kM`, `E_kL`
- `sigma_t`, `H_t`
- `rho`

### Missing to reach target

Add the full LDPM parameter groups for:

- tensile mixed-mode softening: `l_t`, `r_st`, `n_t`, `r_s`,
- compression / compaction: `sigma_c0`, `H_c0`, `H_c1`, `kappa_c0`, `kappa_c1`, `kappa_c2`, `kappa_c3`, `beta`, `E_d`,
- friction: `mu_0`, `mu_inf`, `sigma_N0`,
- optional unloading parameters: `k_t`, `k_s`, `k_c`.

### Result of this step

FERIS will be able to represent the **same parameter space as the intended modern LDPM model**, rather than only a tensile-damage subset.

---

## 3. Gap 2 — Facet History Variables

### Current simplification

Each edge stores only:

- `kappa` = max tensile/shear driving strain
- `omega` = scalar damage

### Missing to reach target

Introduce additional per-facet state for:

- compression / compaction history,
- frictional sliding / plastic slip history,
- mode-dependent constitutive thresholds,
- any branch or yield-state information needed by the chosen update.

### Result of this step

The implementation will be able to remember **different irreversible mechanisms**, instead of forcing all inelasticity into one scalar damage variable.

---

## 4. Gap 3 — Constitutive Law Branching

### Current simplification

`ldpm_tet4_cusatis_traction(...)` currently does only this:

- tensile/shear damage driven by `e_N`, `e_M`, `e_L`,
- elastic compression,
- elastic rotational moments.

### Missing to reach target

Refactor the constitutive kernel so it has explicit branches for:

1. tensile opening / softening,
2. compressive yielding and compaction,
3. frictional sliding under compression,
4. mixed-mode transitions.

### Result of this step

The facet law becomes a **true LDPM constitutive update**, instead of a single softening equation wrapped around an otherwise elastic response.

---

## 5. Gap 4 — Rotational Response and What Stays the Same

### Current behavior

Rotational moments are linear elastic through `E_kT`, `E_kM`, and `E_kL`.

### Target behavior

For the standard first-order LDPM upgrade, this part should mostly stay the same. The main modernization work is in the **translational facet constitutive law**, not in adding rotational damage.

### Result of this step

This keeps the roadmap realistic: not everything must change. The major gap is the missing compression / friction / mixed-mode translational constitutive logic.

---

## 6. Gap 5 — Output and Verification

### Current simplification

Current diagnostics are mainly:

- nodal forces/moments,
- per-edge traction cache,
- `kappa`,
- `omega`,
- projected `crack_distance`.

### Missing to reach target

Add outputs that reflect the richer constitutive state, such as:

- compressive state indicators,
- sliding / friction state indicators,
- branch-aware facet response summaries,
- improved crack opening / sliding measures.

Also add verification cases that separately exercise:

- pure tension,
- confined compression,
- compression-shear friction,
- mixed-mode transitions.

### Result of this step

The model becomes much easier to validate against the intended modern LDPM behavior.

---

## 7. Suggested Implementation Order

### Phase 1 — Expand the parameter API ✅ DONE

Added `SetLDPMParams()` with full compression, friction, and mixed-mode tension parameters. Legacy `SetDamageParams()` preserved for backward compatibility.

### Phase 2 — Expand per-facet state storage ✅ DONE

Added per-edge `e_N_comp` (compressive strain history) alongside existing `kappa` and `omega`.

### Phase 3 — Replace the simplified constitutive kernel ✅ DONE

New `ldpm_tet4_constitutive_update()` in `LDPM.cuh` implements:
- Tensile fracturing with mode-mixity dependent boundary strength (r_st, n_t)
- Compressive yielding, hardening, and densification (sigma_c0, Hc0_ratio, kappa_c0, Ed_ratio)
- Pressure-dependent friction under compression (mu_0, mu_inf, sigma_N0)

Legacy `ldpm_tet4_cusatis_traction()` wrapper preserved for backward compatibility.

### Phase 4 — Upgrade post-processing and tests

Expose the richer state in outputs and add focused single-facet / single-tet verification cases.

### Phase 5 — Align demos with the target model ✅ DONE (single-tet)

`ldpm_singletet.cc` updated to use `SetLDPMParams()` with full Table-4 parameter set. Dogbone demos continue to use `SetDamageParams()` (legacy behavior preserved via defaults).

---

## Bottom line

The main gaps (Phases 1–3, 5) have been closed:

- The **facet constitutive model** now implements full LDPM tension/compression/friction/compaction.
- The **parameter API** supports the full modern LDPM parameter set.
- The **single-tet demo** uses the complete parameter set from the reference document.
- **Backward compatibility** is maintained for legacy demos via `SetDamageParams()`.

Remaining work: Phase 4 (richer post-processing outputs for compressive/friction state).

---

## Literature basis

This gap definition is based on comparing the current FERIS implementation to the canonical LDPM formulation and its calibration framework in:

1. Cusatis, G., Pelessone, D., & Mencarelli, A. (2011). *Lattice Discrete Particle Model (LDPM) for failure behavior of concrete. I: Theory*. Cement and Concrete Composites, 33(9), 881-890. https://doi.org/10.1016/j.cemconcomp.2011.02.011  
2. Cusatis, G., Pelessone, D., & Mencarelli, A. (2011). *Lattice Discrete Particle Model (LDPM) for failure behavior of concrete. II: Calibration and validation*. Cement and Concrete Composites, 33(9), 891-905. https://doi.org/10.1016/j.cemconcomp.2011.02.010
