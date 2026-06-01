# LDPM-TET4 Target Implementation Summary

> **Goal:** Document the target modern LDPM facet model that FERIS implements.

---

## Table of Contents

1. [Target Scope](#1-target-scope)
2. [Core Constitutive Mechanisms](#2-core-constitutive-mechanisms)
3. [Target Parameter Set](#3-target-parameter-set)
4. [Target Facet State Variables](#4-target-facet-state-variables)
5. [Target Constitutive Update Structure](#5-target-constitutive-update-structure)
6. [Rotational Moments in the Target Model](#6-rotational-moments-in-the-target-model)
7. [Target Use of Existing Mesh Data](#7-target-use-of-existing-mesh-data)
8. [Expected Outputs](#8-expected-outputs)

---

## 1. Target Scope

The FERIS LDPM implementation matches the **canonical modern LDPM facet formulation** as implemented in the chrono-mechanics CPU code (`ChMaterialVECT.cpp`):

- elastic facet response in normal and shear directions,
- tensile fracturing / softening with mode-mixity dependence,
- compressive pore collapse, hardening, and rehardening / densification,
- pressure-dependent frictional sliding in shear,
- distinct history variables for tensile damage, compressive compaction, and sliding,
- continued support for the GPU edge-based geometry and leapfrog infrastructure.

---

## 2. Core Constitutive Mechanisms

### 2.1 Elastic regime

Normal modulus `E_0` and shear modulus `E_T = alpha * E_0`.

### 2.2 Tensile fracturing (FractureBC)

When `eps_N > 0`, the model computes:

- Mode-mixity angle: `omega = atan(eps_N / (sqrt(alpha) * eps_T))`
- Effective strength with mixed-mode dependence on the ratio `r_st = sigma_s / sigma_t`
- Softening modulus: `H0 = Hs/alpha + (Ht - Hs/alpha) * (2*omega/pi)^n_t`
  where `Ht = 2*E0 / (l_t/l0 - 1)` and `Hs = r_s * E0`
- Exponential softening: `sigma_bt = sigma0 * exp(-H0 * max(eps_max - eps0, 0) / sigma0)`
- Unloading via `k_t` parameter
- Stress projection: `sigma_N = sigma_fr * eps_N / eps_Q`, `sigma_M = alpha * sigma_fr * eps_M / eps_Q`

### 2.3 Compression / pore collapse / compaction (CompressBC)

When `eps_N <= 0`:

- Deviatoric strain: `eps_D = eps_N - eps_V`
- Modified volumetric strain: `eps_DV = eps_V + beta * eps_D`
- Confinement ratio: `r_DV = |eps_D| / (eps_V - eps_v0)` (or `/eps_v0` when `eps_V > 0`)
- Confinement-sensitive hardening: `Hc = (H_c0 - H_c1) / (1 + kc2 * max(r_DV - kc1, 0)) + H_c1`
- Three-phase boundary: elastic → linear hardening → exponential compaction
- Densification modulus `E_d` replaces `E_0` after yielding

### 2.4 Frictional shear under compression (ShearBC)

When the facet is compressed:

- Shear boundary: `sigma_bs = sigma_s + (mu0 - mu_inf) * sigma_N0 - mu_inf * sigma_N - (mu0 - mu_inf) * sigma_N0 * exp(sigma_N / sigma_N0)`
- Elastic trial clamped to boundary with return mapping to maintain direction

### 2.5 Constitutive branching

The model branches on `eps_N`:
- `eps_N > 0` → tensile/fracture path (coupled tension-shear via effective strain)
- `eps_N <= 0` → independent compression (CompressBC) + friction (ShearBC)

---

## 3. Target Parameter Set

### 3.1 Elastic parameters

- `E_0`: facet normal modulus
- `alpha`: shear-to-normal modulus ratio
- `E_kT`, `E_kM`, `E_kL`: rotational stiffness
- `rho`: density

### 3.2 Tensile-fracture parameters

- `sigma_t`: mesoscale tensile strength
- `sigma_s`: mesoscale shear strength (defines `r_st = sigma_s / sigma_t`)
- `l_t`: tensile characteristic length
- `n_t`: softening exponent (mode-mixity shape)
- `r_s`: shear softening modulus ratio
- `k_t`: tensile unloading parameter

### 3.3 Compression / compaction parameters

- `sigma_c0`: initial compressive yielding threshold
- `H_c0`: initial hardening modulus
- `H_c1`: final hardening modulus
- `kc0`: transitional strain ratio (defines `eps_c1 = eps_c0 * kc0`)
- `kc1`: deviatoric strain threshold ratio
- `kc2`: deviatoric damage parameter
- `kc3`: volumetric strain parameter (defines `eps_v0 = eps_c0 * kc3`)
- `beta`: volumetric-deviatoric coupling
- `E_d`: densification stiffness

### 3.4 Friction parameters

- `mu_0`: initial friction coefficient
- `mu_inf`: asymptotic friction coefficient
- `sigma_N0`: transitional stress for the friction law

### 3.5 Elastic analysis flag

- `elastic_flag`: if true, bypasses all inelastic updates (useful for debugging)

---

## 4. Target Facet State Variables

Each edge stores 16 state variables (LDPM_N_STATEV = 16):

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

---

## 5. Target Constitutive Update Structure

1. Compute strain increments from current total strains minus stored accumulated strains
2. Branch on sign of normal strain:
   - Positive → tensile/fracture regime (coupled via effective strain)
   - Non-positive → compression boundary + friction boundary (independent)
3. Update energy quantities (internal work, dissipated energy, crack opening)
4. Store updated state vector
5. Return tractions and elastic rotational moments

---

## 6. Rotational Moments in the Target Model

Rotational moments remain **elastic-only** (first-order LDPM):

- `m_T = E_kT * kappa_T`
- `m_M = E_kM * kappa_M`
- `m_L = E_kL * kappa_L`

---

## 7. Target Use of Existing Mesh Data

The implementation uses:
- particle positions and TET connectivity
- sub-facet projected areas and tangent frames
- reference edge length `l0` (used in softening modulus calculation)

---

## 8. Volumetric Strain

The target model uses a volumetric strain `eps_V` in the compressive boundary
computation. FERIS computes `eps_V` as a true per-TET volumetric strain:

```cpp
eps_V_tet = (V_cur - V_ref) / (3 * V_ref)
```

Each time step, `GPU_LDPMTet4_Data::ComputeVolumetricStrain()` computes this
value for every TET from the current particle positions, then averages the
values across all TETs sharing each edge through the precomputed edge-to-TET CSR
mapping. The per-edge averaged value is stored in `d_edge_vol_strain` and read by
`compute_ldpm_facet_strain_and_stress()` as `edge_vol_strain(edge_idx)`.

This replaces the earlier per-facet approximation `eps_V = eps_N` and matches
the volume-strain approach used by `ChElementLDPM::ComputeVolume()` in the CPU
reference.

---

## 9. Expected Outputs

The model reports:
- facet tractions (t_N, t_M, t_L) and moments (m_T, m_M, m_L)
- per-edge effective strain and stress history
- crack opening displacement
- internal work and dissipated energy
- legacy kappa/omega for backward-compatible visualization

---

## 10. References

1. Cusatis, G., Pelessone, D., & Mencarelli, A. (2011). *Lattice Discrete Particle Model (LDPM) for failure behavior of concrete. I: Theory*. Cement and Concrete Composites, 33(9), 881-890.
2. chrono-mechanics (base_dev): `src/chrono_ldpm/ChMaterialVECT.cpp` — CPU reference implementation.
