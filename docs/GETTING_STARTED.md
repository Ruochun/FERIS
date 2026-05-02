# Getting Started with FERIS

This guide will help you get up and running with FERIS quickly.

## What is FERIS?

FERIS is a GPU-accelerated framework for solid mechanics and particle simulations built on CUDA. It provides:

- **Five element types**: FEAT4 (TET4), FEAT10 (TET10), ANCF3243 (cable), ANCF3443 (shell), LDPMTet4 (6-DOF discrete particle)
- **Five solvers**: LinearStaticSolver, SyncedNesterov, SyncedAdamW, SyncedAdamWNocoop, LeapfrogSolver
- **Multiple material models**: St. Venant–Kirchhoff, Mooney–Rivlin, LDPM facet damage law
- **CUDA acceleration** for all element and solver kernels

See [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](SOLVER_ELEMENT_COMPATIBILITY.md) for which element types work with which solvers.

## Installation

### Step 1: Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake libeigen3-dev
```

**CUDA Toolkit:**
Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Step 2: Clone and Build

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Ruochun/FERIS.git
cd FERIS

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Running Your First Simulation

### Continuum FEA (TET10, Nesterov solver)

```bash
cd build
./bin/test_feat10_nesterov
```

This example tests the FEAT10 element infrastructure with the SyncedNesterov solver on a small cube mesh.

### Cantilever Beam (TET10, SyncedNesterov)

```bash
cd build
./bin/beam_simulation
```

Loads a beam mesh from `data/meshes/T10/beam.vtu`, applies a tip load, and runs quasi-static equilibrium.

### Cantilever Beam — Linear Static

```bash
./bin/beam_linear_static     # TET10
./bin/beam_linear_static_t4  # TET4
```

Single-step linear static solve using cuSPARSE conjugate-gradient.

### Cantilever Beam — Explicit Dynamic (Leapfrog)

```bash
./bin/test_leapfrog_t4    # TET4 continuum
./bin/test_ldpm_tet4      # LDPMTet4 6-DOF particle (recommended LDPM example)
```

See [`examples/README.md`](../examples/README.md) for the full list and a description of why results differ across examples.

## Understanding the Code

### Main Components

```cpp
// 1. Create GPU element data structure
GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
gpu_t10_data.Setup(x, y, z, connectivity);

// 2. Set material and density
gpu_t10_data.SetSVK(E, nu);       // St. Venant-Kirchhoff material
gpu_t10_data.SetDensity(rho0);

// 3. Apply boundary conditions and loads
gpu_t10_data.SetNodalFixed(fixed_nodes);
gpu_t10_data.SetExternalForce(f_ext);

// 4. Create solver and solve
SyncedNesterovSolver solver(&gpu_t10_data, n_constraints);
SyncedNesterovParams params = {...};
solver.SetParameters(&params);
for (int i = 0; i < N_STEPS; ++i) solver.Solve();
```

## Troubleshooting

### Build Fails with "nvcc not found"
- Make sure CUDA is installed: `which nvcc`
- Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### Runtime Error: "CUDA out of memory"
- Reduce mesh size
- Use a GPU with more memory
- Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU

### Solver Doesn't Converge
- Increase `max_inner` or `max_outer` iterations
- Adjust `inner_tol` or `outer_tol` tolerances
- Reduce the time step
- Check material parameters are physically reasonable

## Further Reading

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md): System design and component overview
- [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](SOLVER_ELEMENT_COMPATIBILITY.md): Which solver works with which element
- [`docs/BUILDING.md`](BUILDING.md): Detailed build instructions
- [`AGENTS.md`](../AGENTS.md): Rules for coding agents modifying this repo
- [`README.md`](../README.md): Project overview
