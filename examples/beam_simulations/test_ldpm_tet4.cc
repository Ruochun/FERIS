/**
 * LDPM-TET4 (TYPE_LDPM_TET4) Leapfrog Dynamic Simulation Example
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Demonstrates the 6-DOF LDPM element (GPU_LDPMTet4_Data, TYPE_LDPM_TET4)
 * driven by the explicit leapfrog (central-difference) solver on a TET4
 * Delaunay mesh.  Each particle carries 3 translational and 3 rotational
 * degrees of freedom.  The rotational kinematics use linearised / small
 * rotations.
 *
 * Setup overview:
 *   1. Load a TET4 beam mesh.
 *   2. Create GPU_LDPMTet4_Data from the mesh (edges are extracted automatically).
 *   3. Set 5 LDPM-TET4 material parameters (E_N, E_T, E_kT, E_kM, E_kL).
 *   4. Set particle density.
 *   5. Fix nodes on the x = x_min face (cantilever).
 *   6. Apply a downward distributed load across all nodes on x = x_max face.
 *   7. Build the diagonal lumped-mass and rotational-inertia matrix.
 *   8. Create LeapfrogSolver, set parameters, call Setup.
 *   9. Time-march with Solve() and write VTK snapshots every N steps.
 *
 * Material parameters
 * ───────────────────
 *   E_N    – normal translational modulus (Pa)
 *   E_T    – shear translational modulus  (E_T = alpha_t * E_N)
 *   E_kT   – twist modulus     (Pa·m²;  E_kT = beta * E_N * l_min^2)
 *   E_kM   – bending modulus m (Pa·m²;  E_kM = beta * E_N * l_min^2)
 *   E_kL   – bending modulus l (Pa·m²;  same)
 *
 * CFL stability note
 * ──────────────────
 *   Translational:  dt_crit_t = l_min / c_N   where  c_N = sqrt(E_N / rho)
 *   Rotational:     dt_crit_r = l_min / c_r    where  c_r = sqrt(E_kN / (I/A))
 *   The time step is chosen as safety_factor * min(dt_crit_t, dt_crit_r).
 *   In practice the translational condition is almost always the tighter one.
 *
 * VTK output
 * ──────────
 *   Each snapshot is written as output_ldpm_tet4_<frame>.vtk (legacy ASCII).
 *   Load the series in ParaView with File → Open, select the group, and use
 *   the Animation View to step through frames.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <MoPhiEssentials.h>

#include "../src/FEASolver.h"
#include "../src/elements/LDPMTet4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
#include "../src/utils/mesh_utils.h"
#include "../src/utils/quadrature_utils.h"

using namespace feris;

// ── Material and simulation parameters ────────────────────────────────────────
// Normal modulus (Pa)
const Real E_N_val = 7e10;
// Shear modulus: alpha = 0.25 gives quasi-isotropic translational response.
const Real alpha_t = Real(0.25);
const Real E_T_val = alpha_t * E_N_val;
// Rotational moduli: beta scales the bending/twist stiffness relative to E_N.
// beta = 0.25 couples translational and rotational stiffness via l_min.
// The actual values are E_k* = beta * E_N * l_min^2; we set a characteristic
// l_ref below and compute E_k* from it.  If l_ref is the minimum edge length,
// this gives the same scale as the CFL-relevant rotational wave speed.
const Real rho_val = 2700.0;     // kg/m³
const Real beta_k = Real(0.25);  // dimensionless rotational coupling factor

static constexpr Real BOUNDARY_TOLERANCE_FRACTION = 0.0005;

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // ──────────────────────────────────────────────────────────────────────────
    // 1. Load mesh
    // ──────────────────────────────────────────────────────────────────────────
    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu("data/meshes/T4/beam_highres.vtu");
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << "\n";
        return 1;
    }

    if (mesh.NumOwnedTets() == 0) {
        std::cerr << "No TET4 elements found in mesh.\n";
        return 1;
    }

    const int n_nodes = mesh.NumLocalNodes();
    const int n_elems = static_cast<int>(mesh.NumOwnedTets());
    MOPHI_INFO("Mesh: %d nodes, %d TET4 elements", n_nodes, n_elems);

    MatrixXi elements(n_elems, 4);
    for (int e = 0; e < n_elems; e++) {
        for (int k = 0; k < 4; k++) {
            elements(e, k) = mesh.topo.tets[e][k];
        }
    }

    VectorXR h_x(n_nodes), h_y(n_nodes), h_z(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        h_x(i) = mesh.geom.nodes[i].x();
        h_y(i) = mesh.geom.nodes[i].y();
        h_z(i) = mesh.geom.nodes[i].z();
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2. Compute minimum edge length for CFL and rotational moduli
    // ──────────────────────────────────────────────────────────────────────────
    Real l_min = std::numeric_limits<Real>::max();
    for (int e = 0; e < n_elems; e++) {
        for (int a = 0; a < 4; a++) {
            for (int b = a + 1; b < 4; b++) {
                int na = elements(e, a);
                int nb = elements(e, b);
                Real dx = h_x(nb) - h_x(na);
                Real dy = h_y(nb) - h_y(na);
                Real dz = h_z(nb) - h_z(na);
                Real l = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (l < l_min)
                    l_min = l;
            }
        }
    }

    // Rotational moduli: E_k* = beta * E_N * l_min^2  (Pa·m²)
    const Real E_kT_val = beta_k * E_N_val * l_min * l_min;
    const Real E_kM_val = E_kT_val;
    const Real E_kL_val = E_kT_val;

    MOPHI_INFO("l_min = %.4e m,  E_N = %.2e Pa,  E_kT = E_kM = E_kL = %.2e Pa*m^2", l_min, E_N_val, E_kT_val);

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Identify fixed nodes (clamp at x = x_min)
    // ──────────────────────────────────────────────────────────────────────────
    const Real x_min = h_x.minCoeff();
    const Real x_max = h_x.maxCoeff();
    const Real tol_BC = BOUNDARY_TOLERANCE_FRACTION * (x_max - x_min);

    std::vector<int> fixed_idx;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x(i) - x_min) < tol_BC) {
            fixed_idx.push_back(i);
        }
    }
    MOPHI_INFO("Fixed particles (x = x_min = %.4f): %zu", x_min, fixed_idx.size());

    VectorXi h_fixed(static_cast<int>(fixed_idx.size()));
    for (int i = 0; i < static_cast<int>(fixed_idx.size()); i++) {
        h_fixed(i) = fixed_idx[i];
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Identify loaded nodes (all particles on x = x_max face)
    // ──────────────────────────────────────────────────────────────────────────
    std::vector<int> load_idx;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x(i) - x_max) < tol_BC) {
            load_idx.push_back(i);
        }
    }
    if (load_idx.empty()) {
        MOPHI_ERROR("Could not find any particle at x = x_max for load application.");
        return 1;
    }
    MOPHI_INFO("Load particles (x = x_max = %.4f): %zu nodes", x_max, load_idx.size());

    // ──────────────────────────────────────────────────────────────────────────
    // 5. Build and configure GPU_LDPMTet4_Data
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPMTet4_Data element_data(n_nodes, n_elems);

    element_data.Setup(h_x, h_y, h_z, elements);

    MOPHI_INFO("GPU_LDPMTet4_Data set up: %d edges extracted", element_data.n_edge);

    element_data.SetNodalFixed(h_fixed);

    // External force: 1000 N downward distributed across all tip-face particles
    const Real F_total = -1000.0;  // N, in -z direction
    const Real F_per_node = F_total / static_cast<Real>(load_idx.size());
    VectorReal3 h_f_ext(static_cast<size_t>(n_nodes));
    for (auto& f : h_f_ext)
        f = Real3::Zero();
    for (int i : load_idx) {
        h_f_ext[static_cast<size_t>(i)](2) = F_per_node;
    }
    element_data.SetExternalForce(h_f_ext);

    element_data.SetMaterial(E_N_val, E_T_val, E_kT_val, E_kM_val, E_kL_val);
    element_data.SetDensity(rho_val);

    // ──────────────────────────────────────────────────────────────────────────
    // 6. Build diagonal lumped mass and rotational inertia
    // ──────────────────────────────────────────────────────────────────────────
    element_data.CalcMassMatrix();
    MOPHI_INFO("CalcMassMatrix done (translational + rotational inertia, alpha = %.2f)", LDPM_TET4_ALPHA_ROT);

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Estimate CFL time step
    //
    //   c_N    = sqrt(E_N / rho)         translational wave speed
    //   dt_crit = l_min / c_N
    //   dt      = safety_factor * dt_crit
    //
    // The rotational CFL is typically much less restrictive.
    // ──────────────────────────────────────────────────────────────────────────
    const Real c_N = std::sqrt(E_N_val / rho_val);
    const Real dt_crit = l_min / c_N;
    const Real safety_factor = Real(0.5);
    const Real dt = safety_factor * dt_crit;

    MOPHI_INFO("c_N = %.2e m/s,  l_min = %.2e m,  dt_crit = %.2e s,  dt = %.2e s", c_N, l_min, dt_crit, dt);

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Create and configure LeapfrogSolver
    // ──────────────────────────────────────────────────────────────────────────
    LeapfrogSolver solver(&element_data);

    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    MOPHI_INFO("LeapfrogSolver configured (TYPE_LDPM_TET4, dt = %.2e s)", dt);

    // ──────────────────────────────────────────────────────────────────────────
    // 9. Time-march and write VTK snapshots
    // ──────────────────────────────────────────────────────────────────────────
    const int n_steps = 5000;
    const int output_interval = 50;

    MatrixXR nodes_ref(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        nodes_ref(i, 0) = h_x(i);
        nodes_ref(i, 1) = h_y(i);
        nodes_ref(i, 2) = h_z(i);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Starting LDPM-TET4 leapfrog time integration for " << n_steps << " steps...\n";
    std::cout << "VTK snapshots will be written every " << output_interval << " steps.\n";

    int frame = 0;

    // Write initial (undeformed) state as frame 0.
    {
        std::ostringstream fname;
        fname << "output_ldpm_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
        ANCFCPUUtils::WriteFEAT4ToVTK(fname.str(), nodes_ref, elements, h_x, h_y, h_z);
        std::cout << "Wrote: " << fname.str() << "  (initial configuration)\n";
        ++frame;
    }

    for (int step = 0; step < n_steps; step++) {
        solver.Solve();

        if ((step + 1) % output_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            // Console progress: mean tip-face displacement
            Real dz_sum = Real(0);
            for (int i : load_idx) {
                dz_sum += z_cur(i) - h_z(i);
            }
            const Real dz_mean = dz_sum / static_cast<Real>(load_idx.size());
            std::cout << "Step " << step + 1 << " | mean tip displacement (z): " << dz_mean << "\n";

            // VTK snapshot
            std::ostringstream fname;
            fname << "output_ldpm_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
            ANCFCPUUtils::WriteFEAT4ToVTK(fname.str(), nodes_ref, elements, x_cur, y_cur, z_cur);
            std::cout << "Wrote: " << fname.str() << "\n";
            ++frame;
        }
    }

    std::cout << "\nSimulation complete.  " << frame << " VTK file(s) written.\n";
    std::cout << "Open output_ldpm_tet4_*.vtk in ParaView to visualize the deformed configuration.\n";

    element_data.Destroy();
    MOPHI_INFO("Done");
    return 0;
}
