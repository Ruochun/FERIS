/**
 * dogbone_leapfrog.cc
 *
 * Dogbone — LDPM-TET4 Leapfrog Simulation Demo with Cusatis Damage Model
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Demonstrates the full Cusatis LDPM physics on a dogbone tensile specimen:
 *
 *   1. Read all six Chrono Workbench LDPM mesh data files (nodes, particles,
 *      tets, facets, faceFacets, facetsVertices).
 *   2. Initialise GPU_LDPMTet4_Data via SetupFromMesh(), which overrides the
 *      tet4-derived facet areas and tangent frame vectors (m, l) with the
 *      richer, precomputed values from the facets.dat file.
 *   3. Set elastic moduli (E_N, E_T, rotational) AND the Cusatis tensile
 *      damage parameters (σ_t, H_t).
 *   4. Fix particles on the z = z_min face (one end of the dogbone).
 *   5. Apply a tensile load on the z = z_max face scaled to exceed the
 *      elastic strength (σ_t * A_cross) so that visible damage occurs.
 *   6. Time-march with LeapfrogSolver; at each output interval retrieve the
 *      per-edge damage state and write three VTK files.
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
 * Two sets of VTK files are written, one per output frame:
 *
 * dogbone_tet4_<frame>.vtk
 *   Written every vtk_interval steps (frame 0 = initial undeformed state):
 *   - POINTS  : current (deformed) particle positions
 *   - CELLS   : VTK_TETRA from tets.dat (connectivity unchanged)
 *   - POINT_DATA "diameter"     : aggregate diameter
 *   - POINT_DATA "displacement" : Euclidean displacement magnitude |u|
 *
 * dogbone_subfacets_<frame>.vtk   ← KEY OUTPUT FOR LDPM CRACK VISUALIZATION
 *   Written every vtk_interval steps (frame 0 = initial, all zeros):
 *   - POINTS  : exact sub-facet vertex positions (fixed reference geometry)
 *   - CELLS   : VTK_TRIANGLE, one per Voronoi sub-facet
 *   - CELL_DATA "damage" : per-subfacet failure ratio ω ∈ [0, 1]
 *                          ω = 0 → undamaged / elastic
 *                          ω = 1 → fully fractured (zero traction capacity)
 *               Each sub-facet's ω is the damage of its owning LDPM edge,
 *               projected via a GPU scatter kernel (ProjectEdgeDamageToSubfacets).
 *
 * ParaView workflow
 * ─────────────────
 *   1. dogbone_tet4_*.vtk        → load as series, color by "displacement"
 *   2. dogbone_subfacets_*.vtk   → load as series, color by "damage"
 *                                  use threshold filter (damage > 0.5) to
 *                                  show only significantly cracked sub-facets
 *
 * Building
 * ────────
 *   mkdir -p build && cd build
 *   cmake ..
 *   make dogbone_leapfrog
 *
 * Running (from the build directory)
 * ───────────────────────────────────
 *   ./bin/dogbone_leapfrog
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

using namespace tlfea;

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
// Applied load
//   The total tensile force is set to OVERLOAD_FACTOR times the quasi-static
//   elastic failure load F_fail = sigma_t * A_cross_approx so that visible
//   damage appears within the simulation window.

static constexpr Real E_N_VAL = Real(60273.0);    // N/mm²  normal modulus (E₀)
static constexpr Real ALPHA_T = Real(0.25);       // E_T / E_N  (shear-to-normal ratio α)
static constexpr Real BETA_K = Real(0.25);        // rotational coupling
static constexpr Real RHO_VAL = Real(2.338e-9);   // tonne/mm³  (= 2338 kg/m³)
static constexpr Real SIGMA_T_VAL = Real(3.44);   // N/mm²  mesoscale tensile strength σ_t
static constexpr Real G_FT_VAL = Real(0.0491);    // N/mm   mode-I fracture energy G_t
static constexpr Real OVERLOAD_FAC = Real(0.3);  // multiplier on F_fail

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

    // Scale applied load so the average cross-section sees OVERLOAD_FAC × σ_t.
    //   F_fail ≈ σ_t * A_cross  (quasi-static elastic limit)
    //   F_TOTAL = OVERLOAD_FAC * F_fail  (ensures visible damage in the neck)
    const Real A_cross_approx = (x_max - x_min) * (y_max - y_min);
    const Real F_fail = SIGMA_T_VAL * A_cross_approx;
    const Real F_TOTAL = OVERLOAD_FAC * F_fail;

    std::cout << "  Approx cross-section area: " << A_cross_approx << " mm²\n";
    std::cout << "  Static elastic limit:  F_fail = " << F_fail << " N\n";
    std::cout << "  Applied tensile force: F_TOTAL = " << F_TOTAL << " N  (" << OVERLOAD_FAC << "× elastic limit)\n";

    // External force: F_TOTAL distributed equally as tensile (+z) load.
    const Real F_per_node = F_TOTAL / static_cast<Real>(load_idx.size());
    VectorXR h_f_ext(mesh.n_particles * 3);
    h_f_ext.setZero();
    for (int i : load_idx)
        h_f_ext(i * 3 + 2) = F_per_node;  // +z direction

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Create and configure GPU_LDPMTet4_Data via SetupFromMesh
    //    This uses the file-based sub-facet info for facet areas and tangent
    //    frame vectors (m, l) instead of the tet4-derived approximation.
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPMTet4_Data element_data(mesh.n_particles, mesh.n_tets);

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

    const int n_steps = 500;
    const int print_interval = 50;  // console output every N steps
    const int vtk_interval = 50;    // VTK snapshot every N steps

    std::cout << "\nRunning " << n_steps << " leapfrog steps (dt = " << dt << " s)...\n";
    std::cout << std::fixed << std::setprecision(6);

    // Helpers: build numbered VTK filenames for TET4 and sub-facet outputs.
    auto tet4_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_tet4_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };
    auto subfacet_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_subfacets_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };

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

        // Sub-facet mesh with all-zero damage (initial, undamaged state).
        VectorXR sf_dmg0(mesh.n_subfacets);
        sf_dmg0.setZero();
        const std::string fname_sf = subfacet_vtk_name(frame);
        if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, sf_dmg0)) {
            std::cerr << "Warning: failed to write initial sub-facet VTK.\n";
        } else {
            std::cout << "Wrote: " << fname_sf << "  (initial, all damage=0)\n";
        }

        ++frame;
    }

    // Print column header.
    const int n_edge = element_data.n_edge;
    std::cout << "\n"
              << std::setw(8) << "Step" << std::setw(22) << "mean dz (loaded) [mm]" << std::setw(18)
              << "max omega (damage)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (int step = 0; step < n_steps; ++step) {
        solver.Solve();

        if ((step + 1) % print_interval == 0 || (step + 1) % vtk_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            VectorXR omega;
            element_data.RetrieveFacetDamageToCPU(omega);

            // Console progress.
            if ((step + 1) % print_interval == 0) {
                Real dz_sum = Real(0);
                for (int i : load_idx)
                    dz_sum += z_cur(i) - mesh.particle_z(i);
                const Real dz_mean = dz_sum / static_cast<Real>(load_idx.size());

                // Maximum damage across all edges at this time step.
                Real omega_max = Real(0);
                for (int e = 0; e < n_edge; ++e)
                    if (omega(e) > omega_max)
                        omega_max = omega(e);

                std::cout << std::setw(8) << (step + 1) << std::setw(22) << dz_mean << std::setw(18) << omega_max
                          << "\n";
            }

            // VTK snapshot: TET4 displacement + sub-facet damage.
            if ((step + 1) % vtk_interval == 0) {
                const std::string fname_t4 = tet4_vtk_name(frame);
                if (!WriteLDPMTet4TetMeshToVTK(fname_t4, mesh, x_cur, y_cur, z_cur)) {
                    std::cerr << "Warning: failed to write TET4 VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_t4 << "\n";
                }

                VectorXR sf_dmg;
                element_data.ProjectEdgeDamageToSubfacets(sf_dmg);
                const std::string fname_sf = subfacet_vtk_name(frame);
                if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, sf_dmg)) {
                    std::cerr << "Warning: failed to write sub-facet VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_sf << "\n";
                }

                ++frame;
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Final damage statistics and analytic displacement estimate
    // ──────────────────────────────────────────────────────────────────────────
    {
        VectorXR x_final, y_final, z_final;
        element_data.RetrievePositionToCPU(x_final, y_final, z_final);

        VectorXR omega_final;
        element_data.RetrieveFacetDamageToCPU(omega_final);

        Real dz_sum = Real(0);
        for (int i : load_idx)
            dz_sum += z_final(i) - mesh.particle_z(i);
        const Real dz_final = dz_sum / static_cast<Real>(load_idx.size());

        Real omega_max = Real(0);
        int n_damaged = 0;
        for (int e = 0; e < n_edge; ++e) {
            if (omega_final(e) > omega_max)
                omega_max = omega_final(e);
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
        std::cout << "  Edges with omega > 0.5         : " << n_damaged << " / " << n_edge << "\n";
        std::cout << "(Non-zero damage indicates Cusatis LDPM fracture is active.)\n";
    }

    element_data.Destroy();

    std::cout << "\nDone.  " << frame << " VTK frame(s) written.\n";
    std::cout << "\nOutput files (two sets, one per frame):\n";
    std::cout << "  dogbone_tet4_NNNNN.vtk          — deformed TET4 mesh\n";
    std::cout << "    → Color by 'displacement' to visualise particle motion.\n";
    std::cout << "  dogbone_subfacets_NNNNN.vtk     — Voronoi sub-facet mesh with damage\n";
    std::cout << "    → Color by 'damage' (omega ∈ [0,1]) to visualise fracture.\n";
    std::cout << "    → Use Threshold filter (damage > 0.5) to show cracked facets.\n";
    std::cout << "  Load each series: File → Open → select group → Apply → Animation View.\n";
    return 0;
}
