# Naming Conventions

This document defines recommended naming conventions for FERIS. The goal is to improve readability across the repository's mixed physics domains (continuum FEA, ANCF, and LDPM) and reduce ambiguity from legacy names such as `x12`, `y12`, and `z12`.

These conventions are intended to guide **new code** and **incremental refactors**. They are not a mandate to perform broad mechanical renames in unrelated files.

## Principles

- Prefer **semantic names** that communicate physical meaning.
- Prefer **explicit state suffixes** such as `_ref` and `_cur` over opaque numeric suffixes.
- Keep **memory-location prefixes** (`h_`, `d_`, `da_`) only when they add useful meaning.
- Use names that work consistently across FEAT, ANCF, and LDPM code.
- Avoid introducing solver-specific ambiguity into generic state variable names.

## Core State Naming

### Positions

When only one configuration is in scope:
- `x`, `y`, `z`

When both reference and current configurations are needed:
- `x_ref`, `y_ref`, `z_ref`
- `x_cur`, `y_cur`, `z_cur`

Grouped alternatives when appropriate:
- `pos_ref`
- `pos_cur`

### Displacements

Use one of:
- `u_x`, `u_y`, `u_z`
- `disp_x`, `disp_y`, `disp_z`

Grouped alternatives:
- `u`
- `disp`

### Translational velocities

Use one of:
- `vx`, `vy`, `vz`
- `vel_x`, `vel_y`, `vel_z`

Grouped alternatives:
- `vel`
- `nodal_vel`

For leapfrog-specific staggered quantities, be explicit:
- `vel_half`
- `vel_nphalf`
- `vel_nmhalf`

### Rotations

For rotational degrees of freedom, prefer:
- `rot_x`, `rot_y`, `rot_z`

Comments and derivations may also use:
- `theta_x`, `theta_y`, `theta_z`

If reference/current distinction is needed:
- `rot_x_ref`, `rot_y_ref`, `rot_z_ref`
- `rot_x_cur`, `rot_y_cur`, `rot_z_cur`

### Angular velocities

Use one of:
- `omega_x`, `omega_y`, `omega_z`
- `ang_vel_x`, `ang_vel_y`, `ang_vel_z`

Grouped alternatives:
- `omega`
- `ang_vel`

## Forces and Moments

### Translational forces
- `f_int`
- `f_ext`

If a distinction is needed:
- `f_int_trans`
- `f_ext_trans`

### Rotational moments
- `m_int`
- `m_ext`

If a distinction is needed:
- `m_int_rot`
- `m_ext_rot`

## Reference Geometry and Directional Quantities

### Lengths and reference measures
- `l0` or `l_ref`
- `edge_l0`
- `edge_length_ref`

### Frame directions
For reference-frame edge/facet directions:
- `n_ref`
- `m_ref`
- `l_ref`

For current-frame variants if ever introduced:
- `n_cur`
- `m_cur`
- `l_cur`

### Edge directions and vectors
- `edge_dir_ref`
- `edge_dir_cur`
- `edge_vec_ref`
- `edge_vec_cur`

## Boundary Conditions and Time Histories

Use explicit names for driven/fixed conditions:
- `fixed_nodes`
- `driven_nodes`
- `prescribed_disp`
- `prescribed_vel`
- `active_prescribed_vel`

For time-history tables:
- `bc_times`
- `bc_disp1`
- `bc_rot1`
- `disp_history`
- `vel_history`

## Memory-space Prefixes

FERIS often uses prefixes indicating host/device ownership. These may be kept where useful:
- `h_` for host-only data
- `d_` for raw device pointers
- `da_` for dual-array wrappers

These prefixes should be combined with semantic names.

Examples:
- `h_x_ref`
- `h_x_cur`
- `d_x_cur`
- `da_x_cur`
- `d_rot_z_cur`
- `da_vel_half`

Avoid pairing memory prefixes with opaque suffixes where possible.

## Legacy Name Guidance

Names such as the following are considered ambiguous and should generally be avoided in new code:
- `x12`
- `y12`
- `z12`
- `rx12`
- `ry12`
- `rz12`

These names may remain in existing interfaces until a local refactor is warranted, but new code should prefer explicit alternatives.

Recommended replacements:
- `x12` → `x_cur`
- `y12` → `y_cur`
- `z12` → `z_cur`
- `rx12` → `rot_x_cur` or `rot_x`
- `ry12` → `rot_y_cur` or `rot_y`
- `rz12` → `rot_z_cur` or `rot_z`

For host/device forms:
- `h_x12` → `h_x_cur`
- `d_x12` → `d_x_cur`
- `da_x12` → `da_x_cur`

## Recommended Repository-wide Standard

Across FEAT, ANCF, and LDPM code, the preferred standard is:
- coordinates: `x_ref`, `y_ref`, `z_ref`, `x_cur`, `y_cur`, `z_cur`
- displacement: `u` / `disp`
- translational velocity: `vel`
- rotation: `rot`
- angular velocity: `omega`
- internal/external translational force: `f_int`, `f_ext`
- internal/external rotational moment: `m_int`, `m_ext`

This vocabulary is broad enough to remain readable across all supported element families.

## Refactoring Policy

To minimize churn:
1. Apply these conventions to **new code**.
2. Prefer clearer **local variable names** even when existing APIs still use legacy names.
3. Refactor legacy public/internal names only when already modifying the surrounding code for a substantive reason.
4. Avoid repository-wide rename-only changes unless explicitly requested.

## Examples

### Preferred local naming in example code

```cpp
VectorXR x_cur, y_cur, z_cur;
element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);
```

Instead of:

```cpp
VectorXR x12, y12, z12;
element_data.RetrievePositionToCPU(x12, y12, z12);
```

### Reference/current distinction

```cpp
const Real dx = x_cur(i) - x_ref(i);
const Real dy = y_cur(i) - y_ref(i);
const Real dz = z_cur(i) - z_ref(i);
```

### Leapfrog-specific velocity naming

```cpp
VectorReal3 prescribed_vel;
VectorXR vel_half;
```

## Scope

This document is guidance for maintainability and consistency. It should be used when reviewing new changes and when making incremental code improvements.