# Single-tetrahedron LDPM histories

`ldpm_singletet` loads the regular tetrahedron in
`data/meshes/LDPMTet4/SingleTet` and uses `LeapfrogSolver` with prescribed
translational or rotational velocity histories. It isolates facet response
under loading, unloading, and reversal using mm–tonne–s units.

| Case | Prescribed history |
|---|---|
| 1–3 | Node 2 x, node 3 y, or node 4 z translation, respectively |
| 4 | Combined nodal translations producing volumetric loading |
| 5–7 | Node 2 x, node 3 y, or node 4 z rotation, respectively |

Translation histories span 4.8 s; rotation histories span 3 s. The driver
defines the piecewise loading histories and material parameters explicitly.
CSV files record nodal forces, moments, displacements, and subfacet
stress/strain; VTK files show tetrahedra and subfacets. Each run cleans and
recreates `ldpm_singletet_caseN` under the working directory.

After [configuring the build](../../../docs/BUILDING.md), run from the
repository root:

```bash
cmake --build build --target ldpm_singletet -j4
./build/bin/ldpm_singletet 1
```

The optional case number is 1–7 and defaults to 1. This is a history benchmark
for inspecting response curves, rather than an automated reference comparison.
