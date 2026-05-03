# LDPM-TET4 Element — 6-DOF Lattice Discrete Particle Model

## Overview

`LDPMTet4` (`TYPE_LDPM_TET4`) is the recommended LDPM element in FERIS.  It
adds three linearised rotational degrees of freedom per particle (θ_x, θ_y, θ_z)
in addition to the three translational ones, enabling a richer representation of
bending and twisting interactions between neighbouring particles.

## Demo

```
examples/beam_simulations/test_ldpm_tet4.cc
```

Simulates a cantilever beam: one end is clamped, a downward distributed force is
applied at the free end, and the system is advanced with the explicit leapfrog
integrator.  VTK snapshots are written every 50 steps.

## Setup Workflow

### A. TET4-mesh-derived geometry (fast, no file data needed)

```cpp
// 1. Build the element from a TET4 mesh
GPU_LDPMTet4_Data element(n_nodes, n_elems);
element.Setup(h_x, h_y, h_z, tet_connectivity);
```

Facet areas and the tangent frame **(m, l)** are computed on-the-fly from the
TET4 Delaunay triangulation using the Voronoi dual construction.

### B. File-based sub-facet geometry (richer, uses Chrono Workbench output)

```cpp
// 1. Read all six Chrono Workbench data files
LDPMTet4Mesh mesh;
std::string err;
ReadLDPMTet4MeshFromFiles("data/meshes/LDPMTet4/Dogbone/LDPMgeo000", mesh, &err);

// 2. Initialise the element from the mesh (SetupFromMesh overrides the
//    tet4-derived facet areas and tangent frame with the richer file values)
GPU_LDPMTet4_Data element;
element.SetupFromMesh(mesh);
```

`SetupFromMesh` reads the Voronoi sub-facet data from `facets.dat`:
- **facet area** per edge ← sum of `pArea` over all its sub-facets
- **m-vector** per edge ← first-tangent `q` from the file (sign-adjusted)
- **l-vector** per edge ← second-tangent `s` from the file

See `examples/Dogbone/dogbone_forcebc.cc` (force-BC) and
`examples/Dogbone/dogbone_velbc.cc` (velocity-BC) for complete tensile-loading demos.

### Shared steps (both paths)

```cpp
// Set 5 material moduli (Pa or Pa·m² for rotational ones)
element.SetMaterial(E_N, E_T, E_kT, E_kM, E_kL);

// Set density (kg/m³)
element.SetDensity(rho);

// Boundary conditions
element.SetNodalFixed(fixed_nodes);     // clamp nodes
element.SetExternalForce(h_f_ext);      // translational force [n_nodes * 3]

// Build lumped mass + rotational inertia
element.CalcMassMatrix();

// Create and run the solver
LeapfrogSolver solver(&element);
LeapfrogParams params{dt};
solver.SetParameters(&params);
solver.Setup();
for (int step = 0; step < n_steps; step++) solver.Solve();
```

## DOF Layout

Each particle has 6 DOFs:

| Array | Size | Meaning |
|-------|------|---------|
| `d_x12, d_y12, d_z12` | `n_nodes` each | Translational positions |
| `d_rx12, d_ry12, d_rz12` | `n_nodes` each | Rotational angles (θ_x, θ_y, θ_z) |
| `d_f_int_t, d_f_ext_t` | `n_nodes * 3` each | Translational force vectors |
| `d_f_int_r, d_f_ext_r` | `n_nodes * 3` each | Rotational moment vectors |

Inside `LeapfrogSolver` the extended arrays are:

| Array | Size | Layout |
|-------|------|--------|
| velocity `d_v_` | `6 * n_nodes` | `[v_x, v_y, v_z, … | ω_x, ω_y, ω_z, …]` |
| mass `d_mass_lump_` | `2 * n_nodes` | `[m_0, …, m_{N-1} | I_0, …, I_{N-1}]` |

## Constitutive Law (`src/materials/LDPMTet4.cuh`)

Every edge (i, j) defines a reference facet frame **(n, m, l)** where **n** is
the unit edge vector and **m**, **l** are two orthogonal tangent directions.

### Translational strains → tractions

```
Δu = (x_j − x_i) − l₀ · n          (relative displacement)
e_N = (Δu · n) / l₀                  t_N = E_N · e_N
e_M = (Δu · m) / l₀                  t_M = E_T · e_M
e_L = (Δu · l) / l₀                  t_L = E_T · e_L
```

### Rotational strains → moments

```
Δθ = θ_j − θ_i                       (rotation difference)
κ_T = (Δθ · n) / l₀                  m_T = E_κT · κ_T   (twist)
κ_M = (Δθ · m) / l₀                  m_M = E_κM · κ_M   (bending, m-dir)
κ_L = (Δθ · l) / l₀                  m_L = E_κL · κ_L   (bending, l-dir)
```

### Force / moment assembly

For each edge the signed facet area `A` is computed from the dual Voronoi cell:

```
Node j:  f_j += +A · (t_N·n + t_M·m + t_L·l)
         τ_j += +A · (m_T·n + m_M·m + m_L·l)

Node i:  f_i += −A · (…)
         τ_i += −A · (…)
```

Assembly uses `atomicAdd` so the kernel is fully parallel over edges.

## Material Parameters

| Symbol | Parameter | Typical value |
|--------|-----------|---------------|
| `E_N` | Normal translational modulus | ≈ 0.6 × E_concrete |
| `E_T` | Shear translational modulus | `α_t · E_N` (α_t ≈ 0.25) |
| `E_κT` | Twist modulus | `β · E_N · l_min²` (β ≈ 0.25) |
| `E_κM` | Bending modulus (m-dir) | same |
| `E_κL` | Bending modulus (l-dir) | same |

The helper `examples/beam_simulations/test_ldpm_tet4.cc` computes `E_κ*` from
`beta_k * E_N * l_min²` where `l_min` is the global minimum edge length.

## Rotational Inertia

```
I_lump[i] = LDPM_TET4_ALPHA_ROT · m_lump[i] · l_min_node[i]²
```

- `LDPM_TET4_ALPHA_ROT = 0.25` (defined in `LDPMTet4Data.cuh`)
- `l_min_node[i]` = minimum edge length incident on node *i*

## CFL Time Step

```
c_N    = sqrt(E_N / rho)
dt_crit = l_min / c_N
dt      = safety_factor · dt_crit   (safety_factor ≈ 0.5)
```

The rotational CFL (`l_min / sqrt(E_κT / (I/A))`) is almost always less
restrictive than the translational CFL.

## Difference from Other Solvers / Elements

LDPM-TET4 with the Leapfrog solver will produce different results than:

- **Linear static solver** — gives the static equilibrium; Leapfrog shows transient dynamics
- **Nesterov / AdamW with FEAT10** — different physical model (continuum FEA vs. discrete particles)

These differences are expected and correct.

## Building

```bash
mkdir -p build && cd build
cmake ..
make test_ldpm_tet4
```

## Running

From the build directory:

```bash
./bin/test_ldpm_tet4
```

Output VTK files: `output_ldpm_tet4_00000.vtk`, `output_ldpm_tet4_00001.vtk`, …

Open in ParaView: File → Open → select group → Apply.
