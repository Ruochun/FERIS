# LDPM-TET4 Current Implementation Summary

> **Repository:** Ruochun/FERIS  
> **Scope:** What the current LDPM branch implements.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mesh and Facet Geometry](#2-mesh-and-facet-geometry)
3. [Particle State and Solver Coupling](#3-particle-state-and-solver-coupling)
4. [Facet Kinematics](#4-facet-kinematics)
5. [Constitutive Response](#5-constitutive-response)
6. [Force, Moment, and Mass Assembly](#6-force-moment-and-mass-assembly)
7. [API](#7-api)
8. [Outputs and History Variables](#8-outputs-and-history-variables)

---

## 1. Overview

The FERIS `TYPE_LDPM_TET4` element is a **GPU edge-based lattice model** built on a TET4 Delaunay mesh. Each unique particle-particle edge is treated as one LDPM interaction, with:

- 3 translational DOFs per particle,
- 3 rotational DOFs per particle,
- one facet frame `(n, m, l)` and one projected facet area per edge,
- explicit leapfrog time integration.

The implementation supports **two constitutive law modes**:

1. **Full LDPM model** (activated via `SetLDPMParams`): tensile fracture with mode-mixity, compressive yielding/hardening/densification, pressure-dependent friction.
2. **Legacy simplified model** (via `SetMaterial` + `SetDamageParams`): single tensile-shear damage variable, elastic compression.

---

## 2. Mesh and Facet Geometry

Same as before: TET4-derived edges with optional facets.dat override for area/tangent frame.

---

## 3. Particle State and Solver Coupling

Each particle carries 6 DOFs (3 translations + 3 rotations). LeapfrogSolver integrates with prescribed velocity BCs.

---

## 4. Facet Kinematics

Small-strain / small-rotation linearized kinematic model:
- `e_N, e_M, e_L` from relative displacement projected onto `(n, m, l)`
- `kappa_T, kappa_M, kappa_L` from rotation difference projected onto `(n, m, l)`

---

## 5. Constitutive Response

### 5.1 Full LDPM Model (when `SetLDPMParams` is called)

**Tensile regime** (`eps_N > 0`):
- Mode-mixity angle `omega = atan(eps_N / (sqrt(alpha) * eps_T))`
- Effective strength with shear-tensile ratio `r_st = sigma_s / sigma_t`
- Exponential softening with mode-dependent modulus
- Unloading via `k_t` parameter

**Compressive regime** (`eps_N <= 0`):
- Three-phase compressive boundary: elastic → hardening → exponential compaction
- Confinement-sensitive hardening modulus `Hc`
- Densification modulus `E_d` after yielding

**Shear under compression**:
- Pressure-dependent friction: `sigma_bs = f(sigma_N, mu_0, mu_inf, sigma_N0)`
- Elastic predictor clamped to friction boundary

### 5.2 Legacy Model (when only `SetMaterial`/`SetDamageParams` is called)

- Single tensile-shear damage driven by effective strain
- Elastic compression
- No friction

### 5.3 Rotational moments (both modes)

Linear-elastic: `m = E_k * kappa` (no damage coupling).

---

## 6. Force, Moment, and Mass Assembly

Same as before: `f = A * (t_N * n + t_M * m + t_L * l)` via atomicAdd.

---

## 7. API

### Material setup (full LDPM)

```cpp
LDPMParams params{};
params.E0 = 30000.0;
params.alpha = 0.2;
// ... (all 24 parameters)
element_data.SetLDPMParams(params);
```

### Material setup (legacy)

```cpp
element_data.SetMaterial(E_N, E_T, E_kT, E_kM, E_kL);
element_data.SetDamageParams(sigma_t, H_t);
element_data.SetDensity(rho);
```

---

## 8. Outputs and History Variables

### Full LDPM mode
- 16-component per-edge state vector (strains, stresses, history, energies, crack opening)
- Legacy kappa/omega maintained for backward-compatible VTK output

### Legacy mode
- Per-edge `kappa` (max effective strain)
- Per-edge `omega` (scalar damage)
- Projected crack_distance

---

## 9. References

1. Cusatis, G., Pelessone, D., & Mencarelli, A. (2011). *LDPM I: Theory*. Cement and Concrete Composites, 33(9), 881-890.
2. chrono-mechanics (base_dev): `src/chrono_ldpm/ChMaterialVECT.cpp`
