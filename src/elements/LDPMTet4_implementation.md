# LDPM-TET4 Current Implementation Summary

> **Repository:** Ruochun/FERIS  
> **Scope:** What the current LDPM branch actually implements today.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mesh and Facet Geometry](#2-mesh-and-facet-geometry)
3. [Particle State and Solver Coupling](#3-particle-state-and-solver-coupling)
4. [Facet Kinematics](#4-facet-kinematics)
5. [Constitutive Response Actually Implemented](#5-constitutive-response-actually-implemented)
6. [Force, Moment, and Mass Assembly](#6-force-moment-and-mass-assembly)
7. [Mesh-File Data Used vs Stored](#7-mesh-file-data-used-vs-stored)
8. [Outputs and History Variables](#8-outputs-and-history-variables)
9. [What This Implementation Does **Not** Yet Do](#9-what-this-implementation-does-not-yet-do)

---

## 1. Overview

The current FERIS `TYPE_LDPM_TET4` element is a **GPU edge-based lattice model** built on a TET4 Delaunay mesh. Each unique particle-particle edge is treated as one LDPM interaction, with:

- 3 translational DOFs per particle,
- 3 rotational DOFs per particle,
- one facet frame `(n, m, l)` and one projected facet area per edge,
- explicit leapfrog time integration.

The current constitutive law is **not the full modern LDPM facet law**. It implements only:

- linear elastic translational stiffness,
- linear elastic rotational stiffness,
- a **single tensile/shear damage law** driven by `e_N`, `e_M`, and `e_L`,
- no compressive yielding/compaction,
- no pressure-dependent frictional sliding law,
- no additional inelastic state variables beyond `kappa` and `omega`.

So the existing code is best described as a **simplified LDPM-style tensile/shear damage model**, not a full current-production LDPM constitutive implementation.

---

## 2. Mesh and Facet Geometry

`GPU_LDPMTet4_Data::Setup()` builds one interaction per unique TET edge. For each edge it stores:

- endpoint nodes,
- reference length `l0`,
- unit normal `n`,
- two tangents `m` and `l`,
- projected facet area.

`SetupFromMesh()` then optionally overrides the TET-derived facet area and tangential directions with richer sub-facet data from `facets.dat`.

### Geometry path actually used

- `nodes / particles / tets` define particle positions and TET connectivity.
- `facets.dat` is used to override **facet area** and the **tangent frame** (`m`, `l`).
- the edge normal remains the canonical edge direction from particle `i` to `j`.
- `faceFacets.dat` and `facetsVertices.dat` are stored for visualization / future use.

---

## 3. Particle State and Solver Coupling

Each particle carries 6 DOFs:

- translations: `x, y, z`
- rotations: `rx, ry, rz`

`LeapfrogSolver` stores for `TYPE_LDPM_TET4`:

- velocity array: `[v_trans(3*n_nodes) | v_rot(3*n_nodes)]`
- lumped mass array: `[m_trans(n_nodes) | I_rot(n_nodes)]`

The current branch supports:

- fixed-node BCs through `SetNodalFixed(...)`,
- prescribed translational velocities through `SetPrescribedVelocityBC(...)`,
- prescribed angular velocities through `SetPrescribedAngularVelocityBC(...)`,
- initial velocity seeding through `SetNodalVelocity(...)` and `SetNodalAngularVelocity(...)`.

The recommended velocity-driven LDPM workflow is the one documented in `LeapfrogSolver.cuh`: set parameters, call `Setup()`, register prescribed velocities, seed nodal velocities, call `InitialHalfKick()`, then run the solve loop.

---

## 4. Facet Kinematics

For each edge `(i, j)`, `compute_p()` computes:

- translational strain components `e_N, e_M, e_L`,
- rotational strain components `kappa_T, kappa_M, kappa_L`.

The current translational strain measure is the **reference-edge engineering strain**:

```text
delta_u = (x_j - x_i) - l0 * n

e_N = (delta_u · n) / l0
e_M = (delta_u · m) / l0
e_L = (delta_u · l) / l0
```

Rotational strain is based only on the rotation difference:

```text
delta_theta = theta_j - theta_i

kappa_T = (delta_theta · n) / l0
kappa_M = (delta_theta · m) / l0
kappa_L = (delta_theta · l) / l0
```

This is a **small-strain / small-rotation linearized kinematic model**.

---

## 5. Constitutive Response Actually Implemented

The current material API contains only these facet-response parameters:

### Elastic / rotational stiffness

- `E_N`
- `E_T`
- `E_kT`
- `E_kM`
- `E_kL`

### Tensile damage

- `sigma_t`
- `H_t`

### Internal state actually tracked per edge

- `kappa`: maximum attained effective tensile/shear driving strain
- `omega`: scalar damage variable in `[0, 1]`

### Current law

The damage-driving strain is

```text
e_D = sqrt( max(e_N, 0)^2 + (E_T/E_N) * (e_M^2 + e_L^2) )
```

with history update

```text
kappa_new = max(kappa_old, e_D)
```

and exponential softening after the threshold `e_t0 = sigma_t / E_N`:

```text
omega = 0                                                if kappa <= e_t0
omega = 1 - (e_t0 / kappa) * exp(-H_t * (kappa - e_t0)) if kappa > e_t0
```

Facet tractions/moments are then:

```text
t_N = E_N * e_N                    for e_N <= 0
t_N = E_N * e_N * (1 - omega)      for e_N > 0

t_M = E_T * e_M * (1 - omega)
t_L = E_T * e_L * (1 - omega)

m_T = E_kT * kappa_T
m_M = E_kM * kappa_M
m_L = E_kL * kappa_L
```

### Important interpretation

This means the present branch has:

- **tension-only normal softening**,
- **shear degraded by the same scalar damage variable**,
- **purely elastic compression**,
- **purely elastic rotational response**,
- **no compressive plasticity, pore collapse, friction law, or return mapping**.

---

## 6. Force, Moment, and Mass Assembly

For each edge, the code assembles equal-and-opposite nodal resultants:

```text
f = A * (t_N * n + t_M * m + t_L * l)
M = A * (m_T * n + m_M * m + m_L * l)
```

using `atomicAdd` into per-node force and moment arrays.

`CalcMassMatrix()` builds:

- translational lumped mass from a `1/4` split of each TET volume,
- rotational inertia from `I_i = 0.25 * m_i * l_min,i^2`.

So the current inertial model is also simplified and geometry-based.

---

## 7. Mesh-File Data Used vs Stored

### Actively used by the current constitutive/mechanics path

- particle positions
- TET connectivity
- sub-facet projected areas
- sub-facet tangents / normals (only to map tangent frame and area to each edge)
- density

### Stored but not yet used for richer constitutive behavior

- particle diameters
- sub-facet material flags (`mF`)
- sub-facet cell volumes
- face-facet surface triangles
- exact sub-facet vertex coordinates

These fields are available in `GPU_LDPMTet4_Data`, but the present constitutive kernel does not use them to switch material zones, compute confinement-dependent laws, or apply traction/contact laws.

---

## 8. Outputs and History Variables

The current implementation can retrieve and/or visualize:

- nodal positions,
- nodal rotations,
- nodal internal forces,
- nodal internal moments,
- per-edge tractions/moments,
- per-edge `kappa`,
- per-edge `omega`,
- projected sub-facet damage,
- projected sub-facet `crack_distance = omega * kappa * l0`.

`crack_distance` is a **visualization surrogate** built from the scalar damage history. It is useful for plotting, but it is **not** a full vector-valued crack opening/sliding measure from a modern LDPM inelastic update.

---

## 9. What This Implementation Does **Not** Yet Do

Compared with the canonical modern LDPM facet law, the current branch does **not** yet implement:

1. **compressive boundary / pore-collapse law**,
2. **compressive hardening and rehardening/compaction**, 
3. **pressure-dependent frictional shear law under compression**,
4. **separate inelastic slip / plastic strain variables**,
5. **multi-surface or mode-dependent facet updates**,
6. **material-zone switching using `mF` or similar flags**,
7. **a parameterization centered on the full LDPM calibration set**.

In short, the current branch captures the LDPM mesh layout and a useful tensile/shear-damage subset, but **not yet the full modern LDPM constitutive model**.
