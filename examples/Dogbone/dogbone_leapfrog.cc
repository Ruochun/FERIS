/**
 * dogbone_leapfrog.cc
 *
 * Dogbone — LDPM-TET4 Leapfrog Simulation Demo (file-based sub-facet geometry)
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Demonstrates the upgraded LDPMTet4 element with file-based sub-facet info:
 *
 *   1. Read all six Chrono Workbench LDPM mesh data files (nodes, particles,
 *      tets, facets, faceFacets, facetsVertices).
 *   2. Initialise GPU_LDPMTet4_Data via SetupFromMesh(), which overrides the
 *      tet4-derived facet areas and tangent frame vectors (m, l) with the
 *      richer, precomputed values from the facets.dat file.
 *   3. Fix particles on the z = z_min face (one end of the dogbone).
 *   4. Apply a uniform tensile load on the z = z_max face (opposite end).
 *   5. Time-march with LeapfrogSolver for a small number of steps.
 *   6. Print tip-face mean displacement and write VTK snapshots.
 *
 * The demo is intentionally kept simple (linear-elastic, small strain) so
 * that the output can be compared to an analytic estimate:
 *
 *   delta ≈ F * L / (E_N * A_cross)
 *
 * where L is the gauge length, A_cross the cross-section area, and F the
 * total applied force.  Agreement at early times (before wave reflections
 * contaminate the result) confirms correctness.
 *
 * VTK output
 * ──────────
 * dogbone_subfacets.vtk
 *   Written once before time integration (static reference geometry):
 *   - POINTS  : exact facet-vertex positions (facetsVertices.dat)
 *   - CELLS   : VTK_TRIANGLE per sub-facet
 *   - CELL_DATA "pArea"   : projected facet area
 *   - CELL_DATA "matflag" : material zone (0=mortar, >0=aggregate/ITZ)
 *
 * dogbone_tet4_<frame>.vtk
 *   Written every vtk_interval steps (frame 0 = initial undeformed state):
 *   - POINTS  : current (deformed) particle positions
 *   - CELLS   : VTK_TETRA from tets.dat (connectivity unchanged)
 *   - POINT_DATA "diameter"     : aggregate diameter
 *   - POINT_DATA "displacement" : Euclidean displacement magnitude |u|
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
 *
 * Then open in ParaView:
 *   dogbone_subfacets.vtk   — static sub-facet mesh (color by matflag / pArea)
 *   dogbone_tet4_*.vtk      — animated TET4 mesh    (color by displacement)
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

// Concrete-like LDPM material parameters.
// Mesh coordinates are in mm, so we use the consistent mm/N/s unit system:
//   length [mm], force [N], mass [kg], time [s], stress [N/mm² = MPa].
//   E_N   = 60 GPa  = 60 000 MPa  = 60 000 N/mm²
//   rho   = 2400 kg/m³ = 2.4e-6 kg/mm³
//   The rotational moduli (N·mm²) are derived below from E_N and l_min.

static constexpr Real E_N_VAL  = Real(60000.0);   // N/mm²  (60 GPa)
static constexpr Real ALPHA_T  = Real(0.25);
static constexpr Real BETA_K   = Real(0.25);       // rotational coupling
static constexpr Real RHO_VAL  = Real(2.4e-6);     // kg/mm³
static constexpr Real F_TOTAL  = Real(1000.0);     // total tensile load [N]

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

    std::cout << "  Particles : " << mesh.n_particles  << "\n";
    std::cout << "  TETs      : " << mesh.n_tets        << "\n";
    std::cout << "  Sub-facets: " << mesh.n_subfacets   << "\n";

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
                const Real l  = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (l < l_min) l_min = l;
            }
        }
    }
    std::cout << "  Min edge length: " << l_min << " mm\n";

    // Rotational moduli [N·mm²] = E_N [N/mm²] * l_min² [mm²]
    const Real E_T_VAL  = ALPHA_T * E_N_VAL;
    const Real E_kT_VAL = BETA_K  * E_N_VAL * l_min * l_min;
    const Real E_kM_VAL = E_kT_VAL;
    const Real E_kL_VAL = E_kT_VAL;

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Identify fixed and loaded particles
    //    Fixed  : z ≈ z_min  (clamped end)
    //    Loaded : z ≈ z_max  (pulled end — uniform tensile load)
    // ──────────────────────────────────────────────────────────────────────────
    const Real z_range  = z_max - z_min;
    const Real tol_z    = BC_TOL_FRAC * z_range;

    std::vector<int> fixed_idx, load_idx;
    for (int i = 0; i < mesh.n_particles; ++i) {
        const Real z = mesh.particle_z(i);
        if (std::abs(z - z_min) < tol_z) fixed_idx.push_back(i);
        if (std::abs(z - z_max) < tol_z) load_idx.push_back(i);
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

    // External force: F_TOTAL distributed equally as tensile (+z) load.
    const Real F_per_node = F_TOTAL / static_cast<Real>(load_idx.size());
    VectorXR h_f_ext(mesh.n_particles * 3);
    h_f_ext.setZero();
    for (int i : load_idx)
        h_f_ext(i * 3 + 2) = F_per_node;   // +z direction

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Create and configure GPU_LDPMTet4_Data via SetupFromMesh
    //    This uses the file-based sub-facet info for facet areas and tangent
    //    frame vectors (m, l) instead of the tet4-derived approximation.
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPMTet4_Data element_data(mesh.n_particles, mesh.n_tets);

    element_data.SetupFromMesh(mesh);

    std::cout << "  Unique edges: " << element_data.n_edge << "\n";

    element_data.SetMaterial(E_N_VAL, E_T_VAL, E_kT_VAL, E_kM_VAL, E_kL_VAL);
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
    const Real c_N       = std::sqrt(E_N_VAL / RHO_VAL);  // mm/s
    const Real dt_crit   = l_min / c_N;
    const Real dt        = Real(0.5) * dt_crit;

    std::cout << "  c_N = " << c_N << " mm/s,  dt_crit = " << dt_crit
              << " s,  dt = " << dt << " s\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Set up and run LeapfrogSolver
    // ──────────────────────────────────────────────────────────────────────────
    LeapfrogSolver solver(&element_data);

    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    const int n_steps        = 200;
    const int print_interval = 20;   // console output every N steps
    const int vtk_interval   = 20;   // VTK snapshot every N steps

    std::cout << "\nRunning " << n_steps << " leapfrog steps (dt = " << dt << " s)...\n";
    std::cout << std::fixed << std::setprecision(6);

    // ── Write static sub-facet mesh (reference geometry, written once) ────────
    {
        const std::string sf_vtk = "dogbone_subfacets.vtk";
        std::cout << "Writing sub-facet mesh -> " << sf_vtk << " ...\n";
        if (!WriteLDPMTet4SubfacetMeshToVTK(sf_vtk, mesh)) {
            std::cerr << "Warning: failed to write sub-facet VTK.\n";
        } else {
            std::cout << "  " << mesh.n_facet_vertices << " points, "
                      << mesh.n_subfacets << " VTK_TRIANGLE cells.\n";
        }
    }

    // ── Write frame 0: initial (undeformed) TET4 mesh ────────────────────────
    int frame = 0;
    {
        std::ostringstream fname;
        fname << "dogbone_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
        if (!WriteLDPMTet4TetMeshToVTK(fname.str(), mesh,
                                        mesh.particle_x, mesh.particle_y, mesh.particle_z)) {
            std::cerr << "Warning: failed to write initial TET4 VTK.\n";
        } else {
            std::cout << "Wrote: " << fname.str() << "  (initial configuration)\n";
        }
        ++frame;
    }

    // Print column header.
    std::cout << "\n" << std::setw(8) << "Step"
              << std::setw(22) << "mean dz (loaded) [mm]"
              << "\n";
    std::cout << std::string(30, '-') << "\n";

    for (int step = 0; step < n_steps; ++step) {
        solver.Solve();

        if ((step + 1) % print_interval == 0 || (step + 1) % vtk_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            // Console progress: mean tip-face displacement in z.
            if ((step + 1) % print_interval == 0) {
                Real dz_sum = Real(0);
                for (int i : load_idx)
                    dz_sum += z_cur(i) - mesh.particle_z(i);
                const Real dz_mean = dz_sum / static_cast<Real>(load_idx.size());
                std::cout << std::setw(8) << (step + 1)
                          << std::setw(22) << dz_mean
                          << "\n";
            }

            // VTK snapshot.
            if ((step + 1) % vtk_interval == 0) {
                std::ostringstream fname;
                fname << "dogbone_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
                if (!WriteLDPMTet4TetMeshToVTK(fname.str(), mesh, x_cur, y_cur, z_cur)) {
                    std::cerr << "Warning: failed to write TET4 VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname.str() << "\n";
                }
                ++frame;
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Verify against simple analytic estimate
    //    delta_elastic ≈ F * L / (E_N * A_cross)
    //    We use L = z_max - z_min and a rough cross-section area.
    //    At early times (before reflections) the wave front hasn't reached the
    //    fixed end yet, so the displacement grows as c_N × t; at quasi-static
    //    equilibrium it converges to the analytic delta.  Even without full
    //    convergence the sign and order of magnitude should be correct.
    // ──────────────────────────────────────────────────────────────────────────
    {
        VectorXR x_final, y_final, z_final;
        element_data.RetrievePositionToCPU(x_final, y_final, z_final);

        Real dz_sum = Real(0);
        for (int i : load_idx)
            dz_sum += z_final(i) - mesh.particle_z(i);
        const Real dz_final = dz_sum / static_cast<Real>(load_idx.size());

        // Rough cross-section (rectangular approximation)
        const Real A_cross_approx = (x_max - x_min) * (y_max - y_min);
        const Real L = z_max - z_min;
        const Real delta_analytic = F_TOTAL * L / (E_N_VAL * A_cross_approx);
        const Real t_final = n_steps * dt;

        std::cout << "\nAnalytic (quasi-static) estimate: "
                  << delta_analytic << " mm\n";
        std::cout << "Simulation mean tip displacement at t=" << t_final
                  << " s: " << dz_final << " mm\n";
        std::cout << "(Agreement improves as the wave converges to static equilibrium.)\n";
    }

    element_data.Destroy();

    std::cout << "\nDone.  " << frame << " TET4 VTK file(s) written.\n";
    std::cout << "  Open dogbone_subfacets.vtk in ParaView to inspect the sub-facet mesh\n";
    std::cout << "    → Color by 'matflag' (0=mortar, >0=aggregate/ITZ) or 'pArea'.\n";
    std::cout << "  Open dogbone_tet4_*.vtk in ParaView to see the animated deformed mesh\n";
    std::cout << "    → Color by 'displacement' to visualise particle motion.\n";
    std::cout << "    → Load the series: File → Open → select group → Apply → use Animation View.\n";
    return 0;
}
