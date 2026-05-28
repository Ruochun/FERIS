# LDPM-TET4 Target Implementation Summary

> **Goal:** Document the target modern LDPM facet model that FERIS should implement next.

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

The target FERIS LDPM implementation should match the **canonical modern LDPM facet formulation** used in the Cusatis LDPM family and later practical implementations:

- elastic facet response in normal and shear directions,
- tensile fracturing / softening with mode-mixity dependence,
- compressive pore collapse, hardening, and rehardening / densification,
- pressure-dependent frictional sliding in shear,
- distinct history variables for tensile damage, compressive compaction, and sliding,
- continued support for the current GPU edge-based geometry and leapfrog infrastructure.

The target is therefore **not** just a stronger tensile damage law. It is a **multi-mechanism facet constitutive model**.

---

## 2. Core Constitutive Mechanisms

### 2.1 Elastic regime

Use the existing facet kinematics, with normal modulus `E_0` and shear modulus derived from the standard LDPM shear-normal ratio `alpha`, i.e. `E_T = alpha * E_0`.

### 2.2 Tensile fracturing

Under opening-dominated loading, the target model should use the canonical LDPM tensile boundary with:

- tensile strength `sigma_t`,
- tensile characteristic length `l_t` (or an equivalent `G_t` form),
- mode-mixity dependence through parameters such as `r_st`, `n_t`, and `r_s`.

### 2.3 Compression / pore collapse / compaction

Under compressive loading, the target model should replace the current elastic compression branch with the standard LDPM compressive boundary, including:

- onset of compressive yielding,
- compressive hardening,
- pore collapse / compaction,
- confinement-sensitive rehardening / densification.

### 2.4 Frictional shear under compression

When the facet is compressed, the target model should use a **pressure-dependent shear limit / friction law**, rather than reducing shear only through the tensile damage variable.

### 2.5 Mixed-mode coupling

The target response should handle tension-shear and compression-shear coupling through the proper LDPM constitutive boundaries instead of a single scalar tensile/shear damage rule.

---

## 3. Target Parameter Set

A practical target parameterization should include the standard LDPM groups below.

### 3.1 Elastic parameters

- `E_0`: facet normal modulus
- `alpha`: shear-to-normal modulus ratio (`E_T = alpha * E_0`)
- rotational stiffness parameters (`beta`-style derived parameters, or explicit `E_kT`, `E_kM`, `E_kL` if FERIS keeps its current API)
- `rho`: density

### 3.2 Tensile-fracture parameters

- `sigma_t`: mesoscale tensile strength
- `l_t`: tensile characteristic length
- `r_st`: shear-to-tensile strength ratio
- `n_t`: softening exponent / mode-mixity softening-shape parameter
- `r_s`: shear softening ratio

### 3.3 Compression / compaction parameters

- `sigma_c0`: initial compressive yielding threshold
- `H_c0`: initial compressive hardening parameter
- `H_c1`: final compressive hardening / rehardening parameter
- `kappa_c0`: strain ratio at the rehardening transition
- `kappa_c1`: deviatoric strain threshold ratio
- `kappa_c2`: deviatoric damage parameter
- `kappa_c3`: volumetric strain parameter
- `beta`: volumetric-deviatoric coupling parameter
- `E_d` or `E_d/E_0`: densification stiffness parameter

### 3.4 Friction parameters

- `mu_0`: low-pressure friction coefficient
- `mu_inf`: high-pressure / asymptotic friction coefficient
- `sigma_N0`: transition stress for the friction law

### 3.5 Optional unloading / hysteresis parameters

If FERIS adopts the full cyclic-capable formulation rather than a monotonic subset, also store:

- `k_t`: tensile unloading parameter
- `k_s`: shear unloading parameter
- `k_c`: compressive unloading parameter

---

## 4. Target Facet State Variables

The target implementation should go beyond the current `(kappa, omega)` pair.

At minimum, each facet should be able to track:

- current normal and shear strain components,
- tensile history / damage variable(s),
- compressive boundary / compaction history,
- shear-slip or frictional plastic history,
- branch-selection or yield-state information,
- derived crack opening / sliding quantities for output.

In practice, this means FERIS should support **different irreversible histories for tensile fracture, compressive compaction, and frictional sliding**, rather than forcing all inelasticity into one scalar damage variable.

---

## 5. Target Constitutive Update Structure

A target facet update should follow this structure:

1. compute facet strains from the current `(n, m, l)` frame,
2. compute the mode indicators needed by LDPM,
3. determine whether the facet is in opening, compression, or mixed mode,
4. evaluate the corresponding LDPM constitutive boundary,
5. update the relevant tensile, compressive, and sliding history variables,
6. return normal traction, two shear tractions, and rotational moments,
7. preserve explicit GPU-friendly storage for post-processing.

The key design change is that **compression and friction cannot remain elastic side branches**. They need their own constitutive update logic and state evolution.

---

## 6. Rotational Moments in the Target Model

The target implementation should keep the current first-order LDPM treatment of rotational moments as **elastic-only** unless FERIS explicitly chooses a different higher-order extension.

That means the target full LDPM upgrade should primarily change:

- the translational facet constitutive response,
- the facet history variables,
- the constitutive branching,

while keeping rotational moments in the same general form:

- twist from `kappa_T`,
- bending from `kappa_M` and `kappa_L`,
- no direct damage coupling in the standard first-order formulation.

---

## 7. Target Use of Existing Mesh Data

The current branch already stores several inputs that a fuller LDPM can exploit better:

- sub-facet projected area,
- sub-facet volumes,
- material flags (`mF`),
- face-facet surface triangles,
- exact sub-facet geometry.

The target implementation should use these data not just for visualization, but also where appropriate for:

- richer material-zone assignment,
- boundary traction handling,
- more faithful sub-facet post-processing,
- constitutive branching that depends on facet metadata.

---

## 8. Expected Outputs

Once the target model is implemented, FERIS should be able to report not only the current edge tractions and scalar damage proxy, but also:

- tensile damage state,
- compressive compaction / crushing state,
- frictional / sliding state,
- more physically meaningful crack opening and sliding measures,
- branch-aware facet diagnostics for verification.

That is the level of functionality expected from a **modern, correct LDPM implementation**, as opposed to the current tensile/shear-only simplification.
