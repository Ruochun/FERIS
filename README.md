# TLFEA

Total Lagrangian Finite Element Analysis — A GPU-accelerated FEA and discrete-particle framework.

## Overview

TLFEA provides a GPU-accelerated (CUDA) infrastructure for Total Lagrangian finite element analysis and lattice particle mechanics, featuring:
- **Five element types**: FEAT4 (TET4), FEAT10 (TET10), ANCF3243 (cable), ANCF3443 (shell), LDPMTet4 (discrete particle)
- **Five solvers**: LinearStaticSolver, SyncedNesterov, SyncedAdamW, SyncedAdamWNocoop, LeapfrogSolver
- **Multiple material models**: St. Venant–Kirchhoff, Mooney–Rivlin, LDPM damage law
- **CMake build system**: Easy to build and extend

Based on the [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) project from SBEL at UW-Madison.

## Solver–Element Compatibility

See [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](docs/SOLVER_ELEMENT_COMPATIBILITY.md) for the full matrix.

|  | LinearStatic | SyncedNesterov | SyncedAdamW | SyncedAdamWNocoop | Leapfrog |
|---|---|---|---|---|---|
| FEAT4 (TET4) | ✅ | ✅ | ✅ | ✅ | ✅ |
| FEAT10 (TET10) | ✅ | ✅ | ✅ | ✅ | ✅ |
| ANCF3243 (cable) | ❌ | ✅ | ✅ | ✅ | ✅ |
| ANCF3443 (shell) | ❌ | ✅ | ✅ | ✅ | ✅ |
| LDPMTet4 (particle) | ❌ | ❌ | ❌ | ❌ | ✅ |

## Quick Start

### Requirements

- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- Eigen3 library
- C++17 compatible compiler
- MoPhiEssentials (included as submodule)

### Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Ruochun/TLFEA.git
cd TLFEA

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Build
mkdir build
cd build
cmake ..
make
```

See [docs/BUILDING.md](docs/BUILDING.md) for detailed build instructions.

### Running the Examples

#### Basic Infrastructure Test

```bash
cd build
./bin/test_feat10_nesterov
```

This example tests the FEAT10 element and SyncedNesterov solver infrastructure.

#### Beam Simulation (VTU Mesh Loading)

```bash
cd build
./bin/beam_simulation
```

This example demonstrates loading a mesh from a VTU file and performing realistic engineering analysis:
- Uses MoPhiEssentials' VTU loader to read beam.vtu
- Works with industry-standard VTU mesh format
- Applies boundary conditions and concentrated load
- Simulates cantilever beam deflection
- Exports results to VTK files for visualization

See [examples/README_BEAM.md](examples/README_BEAM.md) for detailed information about this example.

## Project Structure

```
TLFEA/
├── src/
│   ├── elements/      # FEA element implementations (FEAT4, FEAT10, ANCF3243, ANCF3443, LDPMTet4)
│   ├── solvers/       # Solvers (LinearStatic, SyncedNesterov, SyncedAdamW, SyncedAdamWNocoop, Leapfrog)
│   ├── materials/     # Material models (SVK, Mooney-Rivlin, LDPM damage)
│   └── utils/         # Utility functions
├── external/
│   └── MoPhiEssentials/  # Low-level GPU infrastructure (submodule)
├── examples/          # Example programs
├── data/              # Test data and meshes
├── docs/              # Documentation
│   ├── SOLVER_ELEMENT_COMPATIBILITY.md  # Element/solver compatibility matrix
│   └── ...
├── AGENTS.md          # Rules for coding agents modifying this repo
└── CMakeLists.txt     # Build configuration
```

## Dependencies

### MoPhiEssentials

This project uses [MoPhiEssentials](https://github.com/Ruochun/MoPhiEssentials) as a submodule for:
- **GPU Error Handling**: `MOPHI_GPU_CALL` macro for CUDA error checking
- **Logging System**: Thread-safe logging with `MOPHI_INFO`, `MOPHI_WARNING`, `MOPHI_ERROR`
- **Common Utilities**: Shared GPU/CPU data structures and memory management

The integration provides consistent error handling and logging across the codebase.

## Documentation

- [Building Guide](docs/BUILDING.md) - Detailed build and installation instructions
- [Architecture](docs/ARCHITECTURE.md) - System design and component overview

## Features

### Implemented
- ✅ FEAT4 (4-node tetrahedral) element
- ✅ FEAT10 (10-node tetrahedral) element
- ✅ ANCF3243 (flexible cable) element
- ✅ ANCF3443 (flexible shell) element
- ✅ LDPMTet4 (6-DOF lattice discrete particle) element
- ✅ LinearStaticSolver (cuSPARSE conjugate-gradient)
- ✅ SyncedNesterov iterative solver
- ✅ SyncedAdamW iterative solver
- ✅ SyncedAdamWNocoop iterative solver
- ✅ LeapfrogSolver explicit dynamic integrator
- ✅ St. Venant-Kirchhoff hyperelastic material
- ✅ Mooney-Rivlin hyperelastic material
- ✅ LDPM tensile-shear damage material
- ✅ GPU acceleration via CUDA
- ✅ CMake build system
- ✅ VTU mesh loading capability via MoPhiEssentials
- ✅ MoPhiEssentials integration for error handling and logging
- ✅ Solver–element compatibility matrix (`docs/SOLVER_ELEMENT_COMPATIBILITY.md`)
- ✅ Coding agent rules (`AGENTS.md`)

## License

See LICENSE file for details.

## Acknowledgments

This project is based on the [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) research code developed by the Simulation-Based Engineering Laboratory (SBEL) at the University of Wisconsin-Madison.

