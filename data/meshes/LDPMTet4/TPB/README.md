# Reference three-point bending meshes

Unmodified input data from [chrono-mechanics TPB](https://github.com/Computational-Mechanics-Material-Models/chrono-mechanics/tree/5c2ae9466120c025829a4eb573c7532dd70a6bc4/template_project_ldpm/TPB),
branch `Chrono_v10_dev`, commit `5c2ae9466120c025829a4eb573c7532dd70a6bc4`.

- `LDPMgeo000-data-*.dat`: all six files from
  `LDPMgeo000NotchedPrism_Semi_Circle000/`, retaining their Workbench notices.
- `TPBT_ElasticPart.inp`: the elastic arm, used twice with x translations
  -525.5 mm and +100 mm.
- `LICENSE.chrono`: upstream repository license.
- `SHA256SUMS`: checksums of the seven original input files.

Coordinates use millimetres. The LDPM core occupies x=0..100, y=0..100,
z=0..200 with the original bottom notch. The assembled beam extends from
x=-525.5 to 625.5. No mesh regeneration, rescaling, or notch approximation
is applied. See [demo documentation](../../../../examples/ldmp_tests/README_TPB.md).
