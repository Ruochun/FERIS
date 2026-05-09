/**
 * dogbone_forcebc.cc
 *
 * Dogbone — LDPM-TET4 Tensile Demo with Externally Applied Force (Force-BC)
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Demonstrates the full Cusatis LDPM physics on a dogbone tensile specimen
 * driven by an applied nodal force on the top face.  This is the force-BC
 * counterpart of dogbone_velbc.cc (velocity/kinematic-BC version).
 *
 *   1. Read all six Chrono Workbench LDPM mesh data files (nodes, particles,
 *      tets, facets, faceFacets, facetsVertices).
 *   2. Initialise GPU_LDPMTet4_Data via SetupFromMesh(), which overrides the
 *      tet4-derived facet areas and tangent frame vectors (m, l) with the
 *      richer, precomputed values from the facets.dat file.
 *   3. Set elastic moduli (E_N, E_T, rotational) AND the Cusatis tensile
 *      damage parameters (σ_t, H_t).
 *   4. Fix particles on the z = z_min face (one end of the dogbone).
 *   5. Apply a constant tensile force on the z = z_max face equal to
 *      LOAD_FAC × σ_t × A_cross.  With LOAD_FAC = 0.3 the average stress
 *      is half the mesoscale tensile strength, keeping the specimen below
 *      the static failure load.  Dynamic wave amplification (overshoot ≈ 2×
 *      the static displacement) drives local strains towards the damage
 *      threshold at stress concentrators.
 *   6. Time-march with LeapfrogSolver for T_SIM seconds (default 0.1 s).
 *      The step count and output intervals are computed automatically from
 *      the CFL time step so that the simulation always runs for T_SIM
 *      regardless of mesh refinement.
 *
 * Physics: Cusatis LDPM (2011)
 * ────────────────────────────
 * The constitutive model is NOT classical FEA elasticity but the full
 * mesoscale LDPM damage model:
 *
 *   • Each unique Voronoi edge (particle i ↔ particle j) is one interaction
 *     strut with its own facet area, normal, and tangent frame.
 *   • Strains (e_N, e_M, e_L) are resolved on the facet frame from the
 *     relative translational displacement between i and j.
 *   • A characteristic damage strain:
 *       e_D = sqrt( max(e_N, 0)²  +  (E_T/E_N) * (e_M² + e_L²) )
 *     drives tensile-shear coupled damage.
 *   • History variable κ = max(κ_old, e_D) records the maximum e_D
 *     ever reached (irreversible damage).
 *   • Exponential softening:  ω = 1 - (e_t0/κ) * exp(-H_t*(κ - e_t0))
 *     where e_t0 = σ_t / E_N is the damage threshold strain.
 *   • Tractions: t_N = E_N * e_N * (1-ω)  [tension only],
 *                t_M = E_T * e_M * (1-ω),  t_L = E_T * e_L * (1-ω).
 *   • Rotational moments remain linear-elastic (standard first-order LDPM).
 *
 * VTK output
 * ──────────
 * Two sets of VTK files are written per output frame, plus a single stress-displacement CSV:
 *
 * dogbone_forcebc_tet4_<frame>.vtk
 *   Written every vtk_interval steps (frame 0 = initial undeformed state):
 *   - POINTS  : current (deformed) particle positions
 *   - CELLS   : VTK_TETRA from tets.dat (connectivity unchanged)
 *   - POINT_DATA "diameter"     : aggregate diameter
 *   - POINT_DATA "displacement" : Euclidean displacement magnitude |u|
 *
 * dogbone_forcebc_subfacets_<frame>.vtk   ← KEY OUTPUT FOR LDPM CRACK VISUALIZATION
 *   Written every vtk_interval steps (frame 0 = initial, all zeros):
 *   - POINTS  : exact sub-facet vertex positions (fixed reference geometry)
 *   - CELLS   : VTK_TRIANGLE, one per Voronoi sub-facet
 *   - CELL_DATA "crack_distance" : per-subfacet crack opening displacement [mm]
 *                                  = ω × κ × l₀  (inelastic facet opening)
 *                                  0 → undamaged / elastic
 *                                  ~0.1 mm → fully opened crack at 0.1 s simulation
 *               Each sub-facet's value is projected from its owning LDPM edge via
 *               GPU_LDPMTet4_Data::ProjectEdgeCrackDistanceToSubfacets().
 *
 * dogbone_forcebc_stress_disp.csv
 *   Stress [N/mm² = MPa] vs displacement [mm] at every print interval.
 *   Stress = reaction force at fixed (bottom) nodes / cross-section area.
 *   Displacement = mean z-displacement of loaded (top) nodes.
 *
 * ParaView workflow
 * ─────────────────
 *   1. dogbone_forcebc_tet4_*.vtk        → load as series, color by "displacement"
 *   2. dogbone_forcebc_subfacets_*.vtk   → load as series, color by "crack_distance"
 *                                          use threshold filter (crack_distance > 0) to
 *                                          show only fractured sub-facets
 *   3. dogbone_forcebc_stress_disp.csv   → import in any spreadsheet for the
 *                                          stress-displacement curve
 *
 * Building
 * ────────
 *   mkdir -p build && cd build
 *   cmake ..
 *   make dogbone_forcebc
 *
 * Running (from the build directory)
 * ───────────────────────────────────
 *   ./bin/dogbone_forcebc
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <MoPhiEssentials.h>

#include "../src/elements/LDPMTet4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
#include "../src/utils/ldpm_mesh_utils.h"

using namespace feris;

// ── Simulation parameters ─────────────────────────────────────────────────────

// Concrete-like LDPM-Cusatis material parameters from Cusatis et al. (2011)
// dogbone tensile test benchmark.
//
// Unit system: mm-tonne-s (consistent set used by Cusatis):
//   length [mm], force [N], mass [tonne], time [s], stress [N/mm² = MPa].
//   Dimensional check: 1 N = 1 kg·m/s² = (1e-3 tonne)·(1e3 mm)/s² = 1 tonne·mm/s²  ✓
//   Therefore density must be in tonne/mm³, not kg/mm³.
//   (Using kg/mm³ would make force units kg·mm/s² = milliNewtons, not Newtons.)
//
// Elastic parameters
//   E_N   = E₀ = 60 273 N/mm²           normal modulus
//   alpha = E_T/E_N = 0.25              shear-to-normal ratio
//   beta  = E_k*/E_N/l² = 0.25          rotational coupling (N·mm² = N/mm² × mm²)
//   rho   = 2.338e-9 tonne/mm³          (= 2338 kg/m³)
//
// Cusatis damage parameters
//   sigma_t = 3.44 N/mm²   mesoscale tensile strength
//   G_ft    = 0.0491 N/mm  mode-I fracture energy
//             (consistent with tensile characteristic length l_t = 500 mm,
//              i.e. G_ft = l_t * sigma_t² / (2*E_N) = 500*3.44²/(2*60273) ≈ 0.0491)
//   H_t     ≈ l_ch * sigma_t / G_ft  where l_ch ≈ l_min (per-edge characteristic length)
//   The demo sets H_t from these two inputs at runtime (after l_min is known).
//
// Applied load (see LOAD_FAC/T_SIM/T_VTK constants below for details)

static constexpr Real E_N_VAL = Real(60273.0);   // N/mm²  normal modulus (E₀)
static constexpr Real ALPHA_T = Real(0.25);      // E_T / E_N  (shear-to-normal ratio α)
static constexpr Real BETA_K = Real(0.25);       // rotational coupling
static constexpr Real RHO_VAL = Real(2.338e-9);  // tonne/mm³  (= 2338 kg/m³)
static constexpr Real SIGMA_T_VAL = Real(3.44);  // N/mm²  mesoscale tensile strength σ_t
static constexpr Real G_FT_VAL = Real(0.0491);   // N/mm   mode-I fracture energy G_t
// Applied load and simulation time
//   LOAD_FAC sets the total applied force as a fraction of the nominal
//   quasi-static failure load F_fail = sigma_t * A_cross_approx.
//   With LOAD_FAC = 0.3 the average cross-section stress is 0.3 × sigma_t,
//   which is below the static failure limit.  The dynamic wave amplification
//   (overshoot ≈ 2× the quasi-static equilibrium displacement) drives local
//   strains towards the damage threshold at geometric stress concentrators,
//   activating the Cusatis softening response in the most stressed facets.
//
//   T_SIM  = total physical simulation time [s].
//   T_VTK  = interval between VTK snapshots [s].
//   T_CSV  = interval between stress-displacement CSV rows [s].
//   The step count and output intervals are derived from the CFL time step
//   computed at runtime, so the simulation always covers exactly T_SIM
//   regardless of mesh refinement.

static constexpr Real LOAD_FAC = Real(0.3);  // fraction of static failure load (0.3 = 30% of σ_t × A_cross)
static constexpr Real T_SIM = Real(0.1);     // s — total simulation time
static constexpr Real T_VTK = Real(0.005);   // s — VTK + console output interval (~20 frames)
static constexpr Real T_CSV = Real(0.0005);  // s — CSV stress-displacement output (~200 points)

// Fraction of bounding-box extent used as tolerance when identifying boundary
// particles (nodes on the min/max face of the specimen).
static constexpr Real BC_TOL_FRAC = Real(1e-3);

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // ──────────────────────────────────────────────────────────────────────────
    // 1. Read all six LDPM mesh data files
    // ──────────────────────────────────────────────────────────────────────────
    const std::string prefix = "data/meshes/LDPMTet4/Dogbone/LDPMgeo000";

    std::cout << "Reading LDPM mesh from prefix: " << prefix << "\n";

    LDPMTet4Mesh mesh;
    std::string err;
    if (!ReadLDPMTet4MeshFromFiles(prefix, mesh, &err)) {
        std::cerr << "Error loading mesh: " << err << "\n";
        return 1;
    }

    std::cout << "  Particles : " << mesh.n_particles << "\n";
    std::cout << "  TETs      : " << mesh.n_tets << "\n";
    std::cout << "  Sub-facets: " << mesh.n_subfacets << "\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 2. Compute coordinate bounding box and minimum edge length
    // ──────────────────────────────────────────────────────────────────────────
    Real x_min = std::numeric_limits<Real>::max();
    Real x_max = -std::numeric_limits<Real>::max();
    Real y_min = x_min, y_max = x_max;
    Real z_min = x_min, z_max = x_max;

    for (int i = 0; i < mesh.n_particles; ++i) {
        x_min = std::min(x_min, mesh.particle_x(i));
        x_max = std::max(x_max, mesh.particle_x(i));
        y_min = std::min(y_min, mesh.particle_y(i));
        y_max = std::max(y_max, mesh.particle_y(i));
        z_min = std::min(z_min, mesh.particle_z(i));
        z_max = std::max(z_max, mesh.particle_z(i));
    }

    std::cout << "  Bounding box  x=[" << x_min << ", " << x_max << "]"
              << "  y=[" << y_min << ", " << y_max << "]"
              << "  z=[" << z_min << ", " << z_max << "]\n";

    // Minimum edge length across the entire mesh (needed for CFL and E_k*).
    Real l_min = std::numeric_limits<Real>::max();
    for (int e = 0; e < mesh.n_tets; ++e) {
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                const int na = mesh.tet_connectivity(e, a);
                const int nb = mesh.tet_connectivity(e, b);
                const Real dx = mesh.particle_x(nb) - mesh.particle_x(na);
                const Real dy = mesh.particle_y(nb) - mesh.particle_y(na);
                const Real dz = mesh.particle_z(nb) - mesh.particle_z(na);
                const Real l = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (l < l_min)
                    l_min = l;
            }
        }
    }
    std::cout << "  Min edge length: " << l_min << " mm\n";

    // Rotational moduli [N·mm²] = E_N [N/mm²] * l_min² [mm²]
    const Real E_T_VAL = ALPHA_T * E_N_VAL;
    const Real E_kT_VAL = BETA_K * E_N_VAL * l_min * l_min;
    const Real E_kM_VAL = E_kT_VAL;
    const Real E_kL_VAL = E_kT_VAL;

    // Cusatis softening modulus:  H_t ≈ l_min * sigma_t / G_ft  [1/strain]
    const Real H_T_VAL = l_min * SIGMA_T_VAL / G_FT_VAL;
    std::cout << "  Damage threshold strain e_t0 = " << SIGMA_T_VAL / E_N_VAL << "\n";
    std::cout << "  Cusatis softening modulus H_t = " << H_T_VAL << " (per unit strain)\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Identify fixed and loaded particles
    //    Fixed  : z ≈ z_min  (clamped end)
    //    Loaded : z ≈ z_max  (pulled end — uniform tensile load)
    // ──────────────────────────────────────────────────────────────────────────
    const Real z_range = z_max - z_min;
    const Real tol_z = BC_TOL_FRAC * z_range;

    std::vector<int> fixed_idx, load_idx;
    for (int i = 0; i < mesh.n_particles; ++i) {
        const Real z = mesh.particle_z(i);
        if (std::abs(z - z_min) < tol_z)
            fixed_idx.push_back(i);
        if (std::abs(z - z_max) < tol_z)
            load_idx.push_back(i);
    }

    if (fixed_idx.empty()) {
        std::cerr << "Could not find any particle at z = z_min for fixed BC.\n";
        return 1;
    }
    if (load_idx.empty()) {
        std::cerr << "Could not find any particle at z = z_max for load application.\n";
        return 1;
    }

    std::cout << "  Fixed particles (z ≈ " << z_min << "): " << fixed_idx.size() << "\n";
    std::cout << "  Loaded particles (z ≈ " << z_max << "): " << load_idx.size() << "\n";

    VectorXi h_fixed(static_cast<int>(fixed_idx.size()));
    for (int i = 0; i < static_cast<int>(fixed_idx.size()); ++i)
        h_fixed(i) = fixed_idx[i];

    // Scale applied load: F_fail = sigma_t × A_cross (quasi-static elastic limit).
    //   F_TOTAL = LOAD_FAC × F_fail  (below static limit; overshoot from dynamic
    //                                  wave amplification activates local damage)
    const Real A_cross_approx = (x_max - x_min) * (y_max - y_min);
    const Real F_fail = SIGMA_T_VAL * A_cross_approx;
    const Real F_TOTAL = LOAD_FAC * F_fail;

    std::cout << "  Approx cross-section area: " << A_cross_approx << " mm²\n";
    std::cout << "  Static elastic limit:  F_fail = " << F_fail << " N\n";
    std::cout << "  Applied tensile force: F_TOTAL = " << F_TOTAL << " N  (" << LOAD_FAC << "× elastic limit)\n";

    // External force: F_TOTAL distributed equally as tensile (+z) load.
    const Real F_per_node = F_TOTAL / static_cast<Real>(load_idx.size());
    VectorReal3 h_f_ext(static_cast<size_t>(mesh.n_particles));
    for (auto& f : h_f_ext)
        f = Real3::Zero();
    for (int i : load_idx)
        h_f_ext[static_cast<size_t>(i)](2) = F_per_node;  // +z direction

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Create and configure GPU_LDPMTet4_Data via SetupFromMesh
    //    This uses the file-based sub-facet info for facet areas and tangent
    //    frame vectors (m, l) instead of the tet4-derived approximation.
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPMTet4_Data element_data;

    element_data.SetupFromMesh(mesh);

    std::cout << "  Unique edges: " << element_data.n_edge << "\n";

    element_data.SetMaterial(E_N_VAL, E_T_VAL, E_kT_VAL, E_kM_VAL, E_kL_VAL);

    // Cusatis LDPM damage parameters (tensile-shear coupled).
    // sigma_t : mesoscale tensile strength [N/mm²]
    // H_t     : softening modulus [1/strain]; derived from G_ft and l_min above.
    //   Physical meaning: the larger H_t, the more brittle the softening.
    //   Reduce sigma_t or H_t to slow down / soften the damage response.
    element_data.SetDamageParams(SIGMA_T_VAL, H_T_VAL);

    element_data.SetDensity(RHO_VAL);
    element_data.SetExternalForce(h_f_ext);
    element_data.SetNodalFixed(h_fixed);

    // ──────────────────────────────────────────────────────────────────────────
    // 5. Compute lumped mass
    // ──────────────────────────────────────────────────────────────────────────
    element_data.CalcMassMatrix();

    // ──────────────────────────────────────────────────────────────────────────
    // 6. CFL time step estimate
    //   c_N = sqrt(E_N / rho)  [mm/s]
    //   dt  = safety * l_min / c_N
    // ──────────────────────────────────────────────────────────────────────────
    const Real c_N = std::sqrt(E_N_VAL / RHO_VAL);  // mm/s
    const Real dt_crit = l_min / c_N;
    const Real dt = Real(0.5) * dt_crit;

    std::cout << "  c_N = " << c_N << " mm/s,  dt_crit = " << dt_crit << " s,  dt = " << dt << " s\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Set up and run LeapfrogSolver
    // ──────────────────────────────────────────────────────────────────────────
    LeapfrogSolver solver(&element_data);

    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    // Step count and output intervals derived from CFL dt so the simulation
    // always runs for T_SIM seconds regardless of mesh refinement.
    const int n_steps = static_cast<int>(std::ceil(T_SIM / dt));
    const int vtk_interval = std::max(1, static_cast<int>(std::round(T_VTK / dt)));
    const int csv_interval = std::max(1, static_cast<int>(std::round(T_CSV / dt)));
    const int print_interval = vtk_interval;

    std::cout << "\nRunning " << n_steps << " leapfrog steps (dt = " << dt << " s)...\n";
    std::cout << "  Total simulation time : " << n_steps * dt << " s\n";
    std::cout << "  VTK output every      : " << vtk_interval << " steps (" << T_VTK << " s)\n";
    std::cout << "  CSV output every      : " << csv_interval << " steps (" << T_CSV << " s)\n";
    std::cout << std::fixed << std::setprecision(6);

    // Helpers: build numbered VTK filenames for TET4 and sub-facet outputs.
    auto tet4_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_forcebc_tet4_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };
    auto subfacet_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_forcebc_subfacets_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };

    // Open stress-displacement CSV for writing.
    // Each row: displacement_mm,stress_MPa
    std::ofstream stress_disp_csv("dogbone_forcebc_stress_disp.csv");
    if (!stress_disp_csv.is_open()) {
        std::cerr << "Warning: could not open dogbone_forcebc_stress_disp.csv for writing.\n";
    } else {
        stress_disp_csv << "displacement_mm,stress_MPa\n";
    }

    // ── Write frame 0: initial (undeformed) TET4 and sub-facet (all zeros) ───
    int frame = 0;
    {
        // TET4 mesh
        const std::string fname_t4 = tet4_vtk_name(frame);
        if (!WriteLDPMTet4TetMeshToVTK(fname_t4, mesh, mesh.particle_x, mesh.particle_y, mesh.particle_z)) {
            std::cerr << "Warning: failed to write initial TET4 VTK.\n";
        } else {
            std::cout << "Wrote: " << fname_t4 << "  (initial configuration)\n";
        }

        // Sub-facet mesh with all-zero crack distance (initial, undamaged state).
        VectorXR sf_crack0(mesh.n_subfacets);
        sf_crack0.setZero();
        const std::string fname_sf = subfacet_vtk_name(frame);
        if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, "crack_distance", sf_crack0)) {
            std::cerr << "Warning: failed to write initial sub-facet VTK.\n";
        } else {
            std::cout << "Wrote: " << fname_sf << "  (initial, all crack_distance=0)\n";
        }

        ++frame;
    }

    // Print column header.
    const int n_edge = element_data.n_edge;
    std::cout << "\n"
              << std::setw(8) << "Step" << std::setw(22) << "mean dz (loaded) [mm]" << std::setw(18) << "stress [MPa]"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int step = 0; step < n_steps; ++step) {
        solver.Solve();

        if ((step + 1) % csv_interval == 0 || (step + 1) % vtk_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            // Internal forces for stress computation.
            VectorReal3 f_int;
            element_data.RetrieveInternalForceToCPU(f_int);

            // Mean z-displacement of loaded nodes.
            Real dz_sum = Real(0);
            for (int i : load_idx)
                dz_sum += z_cur(i) - mesh.particle_z(i);
            const Real dz_mean = dz_sum / static_cast<Real>(load_idx.size());

            // Reaction stress at the fixed (bottom) face.
            Real f_reaction_z = Real(0);
            for (int i : fixed_idx)
                f_reaction_z += f_int[static_cast<size_t>(i)](2);
            const Real stress = -f_reaction_z / A_cross_approx;

            // Write to stress-displacement CSV at high frequency.
            if ((step + 1) % csv_interval == 0 && stress_disp_csv.is_open()) {
                stress_disp_csv << dz_mean << "," << stress << "\n";
            }

            // Console progress at VTK (lower) frequency.
            if ((step + 1) % print_interval == 0) {
                std::cout << std::setw(8) << (step + 1) << std::setw(22) << dz_mean << std::setw(18) << stress << "\n";
            }

            // VTK snapshot: TET4 displacement + sub-facet crack distance.
            if ((step + 1) % vtk_interval == 0) {
                const std::string fname_t4 = tet4_vtk_name(frame);
                if (!WriteLDPMTet4TetMeshToVTK(fname_t4, mesh, x_cur, y_cur, z_cur)) {
                    std::cerr << "Warning: failed to write TET4 VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_t4 << "\n";
                }

                VectorXR sf_crack;
                element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
                const std::string fname_sf = subfacet_vtk_name(frame);
                if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, "crack_distance", sf_crack)) {
                    std::cerr << "Warning: failed to write sub-facet VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_sf << "\n";
                }

                ++frame;
            }
        }
    }

    stress_disp_csv.close();

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Final fracture statistics and analytic displacement estimate
    // ──────────────────────────────────────────────────────────────────────────
    {
        VectorXR x_final, y_final, z_final;
        element_data.RetrievePositionToCPU(x_final, y_final, z_final);

        VectorXR omega_final, kappa_final;
        element_data.RetrieveFacetDamageToCPU(omega_final);
        element_data.RetrieveFacetKappaToCPU(kappa_final);

        const std::vector<int>& edge_nodes = element_data.GetEdgeNodes();

        Real dz_sum = Real(0);
        for (int i : load_idx)
            dz_sum += z_final(i) - mesh.particle_z(i);
        const Real dz_final = dz_sum / static_cast<Real>(load_idx.size());

        Real omega_max = Real(0);
        Real crack_dist_max = Real(0);
        int n_damaged = 0;
        for (int e = 0; e < n_edge; ++e) {
            if (omega_final(e) > omega_max)
                omega_max = omega_final(e);
            const int ni = edge_nodes[2 * e + 0];
            const int nj = edge_nodes[2 * e + 1];
            const Real dx0 = mesh.particle_x(nj) - mesh.particle_x(ni);
            const Real dy0 = mesh.particle_y(nj) - mesh.particle_y(ni);
            const Real dz0 = mesh.particle_z(nj) - mesh.particle_z(ni);
            const Real l0_e = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
            const Real cd = omega_final(e) * kappa_final(e) * l0_e;
            if (cd > crack_dist_max)
                crack_dist_max = cd;
            if (omega_final(e) > Real(0.5))
                ++n_damaged;
        }

        const Real L = z_max - z_min;
        const Real delta_analytic = F_TOTAL * L / (E_N_VAL * A_cross_approx);
        const Real t_final = n_steps * dt;

        std::cout << "\n── Final state at t=" << t_final << " s ───\n";
        std::cout << "  Analytic quasi-static tip displacement: " << delta_analytic << " mm\n";
        std::cout << "  Simulated mean tip displacement: " << dz_final << " mm\n";
        std::cout << "  Maximum edge damage omega_max  : " << omega_max << "\n";
        std::cout << "  Max crack opening (ω×κ×l₀)    : " << crack_dist_max << " mm\n";
        std::cout << "  Edges with omega > 0.5         : " << n_damaged << " / " << n_edge << "\n";
        std::cout << "(Non-zero damage indicates Cusatis LDPM fracture is active.)\n";
    }

    element_data.Destroy();

    std::cout << "\nDone.  " << frame << " VTK frame(s) written.\n";
    std::cout << "\nOutput files (two sets per frame + stress-displacement CSV):\n";
    std::cout << "  dogbone_forcebc_tet4_NNNNN.vtk          — deformed TET4 mesh\n";
    std::cout << "    → Color by 'displacement' to visualise particle motion.\n";
    std::cout << "  dogbone_forcebc_subfacets_NNNNN.vtk     — Voronoi sub-facet mesh\n";
    std::cout << "    → Color by 'crack_distance' [mm] (ω×κ×l₀) to visualise fracture.\n";
    std::cout << "    → Use Threshold filter (crack_distance > 0) to show cracked facets.\n";
    std::cout << "  dogbone_forcebc_stress_disp.csv         — stress [MPa] vs displacement [mm]\n";
    std::cout << "    → Plot displacement_mm vs stress_MPa for the load-displacement curve.\n";
    std::cout << "  Load each series: File → Open → select group → Apply → Animation View.\n";
    return 0;
}
