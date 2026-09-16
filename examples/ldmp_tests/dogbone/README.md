# Dogbone tensile examples

These examples load the Workbench mesh in `data/meshes/LDPMTet4/Dogbone` and
integrate the LDPM particle translations and rotations with `LeapfrogSolver`.
Both run a 0.1 s tensile simulation with a mesh-dependent CFL time step.

- `dogbone_forcebc`: tensile loading through applied nodal forces.
- `dogbone_velbc`: tensile loading through prescribed top-plate velocity.

They exercise LDPM facet response and damage under specimen-scale loading.
Outputs include deformed tetrahedra, subfacet visualization, and a CSV of
stress versus displacement. Each executable cleans and recreates its own
output folder (`dogbone_forcebc` or `dogbone_velbc`) under the working directory.

After [configuring the build](../../../docs/BUILDING.md), build from the
repository root:

```bash
cmake --build build --target dogbone_forcebc dogbone_velbc -j4
./build/bin/dogbone_forcebc
./build/bin/dogbone_velbc
```

Run from the repository root so the relative mesh paths resolve. These are
simulation examples; they do not assert agreement with a reference curve.
