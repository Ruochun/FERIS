# LDPM examples and tests

Each scenario keeps its drivers, setup helpers, and documentation together:

| Folder | Content |
|---|---|
| [dogbone](dogbone/README.md) | Force-controlled and velocity-driven tensile specimens |
| [tpb](tpb/README.md) | Notched three-point bending with LDPM, elastic arms, and rigid supports |
| [singletet](singletet/README.md) | Seven prescribed translation/rotation histories on one tetrahedron |
| [coupling](coupling/README.md) | Shared dynamics and constraint regression checks |

Targets are registered in this directory's `CMakeLists.txt`; executables remain
in `build/bin`. See [BUILDING.md](../../docs/BUILDING.md) for configuration.
