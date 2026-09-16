# LDPM three-point bending (`ldpm_tpb`)

This demo reproduces the geometry, material inputs, loading history, and
bilateral support layout of the [Chrono TPB example](https://github.com/Computational-Mechanics-Material-Models/chrono-mechanics/blob/5c2ae9466120c025829a4eb573c7532dd70a6bc4/template_project_ldpm/TPB/LDPM_TPBT.cpp)
using FERIS LDPM and explicit leapfrog integration. It includes the elastic
arms: the 100 mm long LDPM region is only the notched middle of the beam.
Build and run commands are in [BUILDING.md](../../../docs/BUILDING.md#ldpm-three-point-bending).

## Physical setup

All quantities use mm–tonne–s, giving N and MPa.

| Part | Setup |
|---|---|
| LDPM core | Original six Workbench data files, x=0..100, y=0..100, z=0..200 mm, with the original bottom notch |
| Elastic arms | Original Abaqus C3D4 mesh twice, translated by -525.5 and +100 mm in x; E=39000 MPa, nu=0.2, rho=2.338e-9 tonne/mm³ |
| Interfaces | Each LDPM particle at x=0 or 100 tied by barycentric interpolation to its elastic-arm surface triangle; particle rotations remain free |
| Supports | Rigid 25×100×10 mm plates centered at (-513,50,-5) and (613,50,-5); density 7.8e-9 tonne/mm³; only each center's z translation restrained |
| Support attachment | Elastic nodes at z=0 within each 25 mm support strip follow that plate's translation and rotation |
| Loading | Core nodes at z=200, x=37.5..62.5: x displacement fixed, z displacement prescribed; y and rotations free |
| CMOD | Change in x separation of the two original notch-mouth particles at y=z=0, x=47.5 and 52.5 |

The support span is 1126 mm. Both supports can slide and rotate; their finite
mass and rotational inertia are included. The top kinematic plate needs no
additional dynamic DOFs because its motion is prescribed.

The signed loading displacement is

```
u_z(t) = -3750 t²                      for 0 <= t <= 0.002 s
         -0.015 - 15 (t - 0.002)       thereafter
```

The default end time is 0.1 s (1.485 mm downward displacement). Each step
uses the exact displacement increment divided by dt, including steps that
cross the ramp transition. The initial velocity and forces are zero.

LDPM inputs match the active reference code: E0=60273, alpha=0.25,
sigma_t=3.44, sigma_s=8.944, n_t=0.4, l_t=500, E_d=60273,
sigma_c0=150, H_c0=24109, H_c1=6027.3, beta=0, kc0=4, kc1=1,
kc2=5, kc3=0.1, mu_0=0.4, mu_inf=0, sigma_N0=600, rho=2.338e-9.
Other unloading/softening parameters are zero. Every imported subfacet has
its own constitutive history. Particle rotation couples through facet force
lever arms; no additional curvature stiffness is introduced.

## Numerical scope

The reference uses implicit HHT and corotational elastic tetrahedra. This
port uses `CoupledLeapfrogSolver` with the shared leapfrog stages for LDPM
and `GPU_FEAT4_Data` arms using **linear small-displacement** SVK response. Rigid plate kinematics and interface constraints are also linear
in displacement/rotation. This is appropriate for small beam deflections;
large-rotation equivalence to Chrono is not claimed.

A mass-weighted bilateral projection couples the free subsystem steps. It
retains elastic-arm compliance, inertia, support sliding, and support
rotation. It does not replace the arms with prescribed core-end motion or
penalty springs. See [implementation notes](../../../docs/LDPMTet4_implementation.md#three-point-bending-demo-coupling)
for the projection equations.

Both material meshes have gravity disabled in the reference. This demo also
omits the tiny rigid-plate gravity loads (the CPU code leaves plate gravity
at the system default). It omits loading-plate inertia from the reported
actuator force: `load_N` is the force applied to the specimen. There is no
contact/friction law in the bilateral support ties.

FERIS retains its existing lumped LDPM masses and scalar particle rotational
inertias. These and the arm mass lumping differ from the CPU mass treatment.
Leapfrog adds no HHT numerical damping. A quantitative CPU/GPU curve match
therefore requires time-step convergence and comparison against an actual
CPU run; this demo does not establish such a match.

The default dt is bounded by 0.15 times the shortest LDPM edge divided by
sqrt(E0/rho), and half the elastic-arm critical-step bound obtained from
absolute row sums of the mass-normalized stiffness. This is a conservative
starting estimate, not a nonlinear stability proof. `--dt-scale 0.5` halves
it for convergence studies. No mass scaling or artificial damping is used.

## Outputs and checks

Each run replaces `./ldpm_tpb/` under the current working directory after
successful model setup. Mesh inputs default to the bundled source-tree data
and can be overridden with `--mesh-dir DIR`.

- `history.csv`: downward deflection, CMOD, specimen loading force, upward
  support reactions, support x sliding, kinetic energy, elastic-arm strain
  energy, maximum crack opening, the legacy damage proxy, and maximum constraint error.
- `core_NNNNN.vtk`: deformed LDPM tetrahedra and displacement.
- `cracks_NNNNN.vtk`: reference subfacet geometry with `crack_distance` in mm.
- `arms_NNNNN.vtk`: deformed elastic arms and displacement.
- `run.txt`: time step, step count, material mode, and mesh sizes.

CSV is sampled approximately every 0.1 ms, VTK every 1 ms, and both include
initial and final states. `--no-vtk` suppresses VTK; `--steps N` caps a run
for smoke testing; `--elastic` disables LDPM inelastic evolution for diagnostics.

Positions are at `time_s`; constraint reactions and facet damage come from
the preceding kick at `reaction_time_s = time_s-dt`. Kinetic energy uses
half-step velocities. The CSV includes only the arms' elastic strain energy,
not total LDPM internal energy. Dynamic load and support reactions need not
balance instantaneously because the specimen and plates have inertia.

The legacy damage proxy is crack strain divided by effective strain, capped at
one. Near zero effective strain it can saturate even for negligible crack
opening; use `max_crack_opening_mm` and the crack VTK field to assess fracture.

The demo rejects missing interface matches, absent supports/loading nodes,
unexpected CMOD nodes, non-positive masses, inaccurate constraint inverses,
non-finite sampled state, and constraint drift above 1e-7 mm. A short run
checks setup and integration, not the full post-peak response.

## Validation performed

`tpb_model_test` checks overlapping and large sparse constraint components,
projection momentum/energy behavior, rigid ties, quadratic face interpolation,
and reference mesh/loading data without a GPU. `coupled_dynamics_test` checks
T4/T10 stiffness/force consistency, rigid modes, positive mass conservation,
prescribed motion, standalone/coupled leapfrog equivalence, rigid-body acceleration,
and a 521-row sparse constraint component on a GPU.

The original implementation (commit `332ab4f`) was validated as follows and
provides the refactoring baseline. GPU checks on the supplied mesh covered initial/final VTK output and nonlinear
runs through 3 ms at dt≈6.9393e-8 s (43,232 steps) and dt≈3.4697e-8 s
(86,463 steps). At 3 ms:

| Quantity | Default dt | Half dt |
|---|---:|---:|
| Downward deflection (mm) | 0.0300000 | 0.0300000 |
| CMOD (mm) | 0.0126163330 | 0.0126163331 |
| Specimen load (N) | 877.374825 | 877.351780 |
| Left support reaction (N) | 443.760762 | 443.749560 |
| Right support reaction (N) | 443.718868 | 443.707678 |

The load difference is 0.00263%; sampled constraint errors stayed below
1e-11 mm. The refined run reached 0.00103606 mm maximum crack opening.
These checks cover the loading-ramp transition and onset of cracking, not
the entire 0.1 s response or quantitative agreement with a CPU reference run.
The existing `dogbone_velbc` and `ldpm_singletet` targets also rebuilt successfully.

After migration to the shared library, the default-step 3 ms run gives
877.374814 N load and 0.012616332954 mm CMOD. Relative to the original baseline,
the load differs by approximately 1.2e-8 and CMOD by less than 5e-11 mm.
The maximum sampled constraint drift is 1.04e-10 mm. The shared solver advances
the element's coordinates directly after projection.
The half-step run (86,463 steps) gives 877.351746 N and 0.012616332972 mm CMOD;
its load differs from the saved half-step baseline by 3.85e-8 relative, and
its sampled constraint drift stays below 3.4e-10 mm.

The existing T4/T10 static beam examples also converge with relative residuals
below 1e-10, and the T4 leapfrog example completes. A 100-step TPB run writes
the expected initial/final VTK files. CUDA memcheck could not attach to the
WSL GPU in this environment; the ordinary GPU tests pass.
