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
 *   6. Print tip-face mean displacement to verify the simulation runs.
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

    const int n_steps          = 200;
    const int print_interval   = 20;

    std::cout << "\nRunning " << n_steps << " leapfrog steps (dt = " << dt << " s)...\n";
    std::cout << std::fixed << std::setprecision(6);

    // Print column header.
    std::cout << std::setw(8) << "Step"
              << std::setw(18) << "mean dz (loaded) [mm]"
              << "\n";
    std::cout << std::string(26, '-') << "\n";

    for (int step = 0; step < n_steps; ++step) {
        solver.Solve();

        if ((step + 1) % print_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            Real dz_sum = Real(0);
            for (int i : load_idx)
                dz_sum += z_cur(i) - mesh.particle_z(i);
            const Real dz_mean = dz_sum / static_cast<Real>(load_idx.size());

            std::cout << std::setw(8) << (step + 1)
                      << std::setw(18) << dz_mean
                      << "\n";
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
    std::cout << "\nDone.\n";
    return 0;
}
