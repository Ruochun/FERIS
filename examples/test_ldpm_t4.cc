/**
 * LDPM4 Leapfrog Dynamic Simulation Example
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Demonstrates the 3-DOF translational LDPM element (GPU_LDPM4_Data) driven
 * by the explicit leapfrog (central-difference) solver on a TET4 Delaunay
 * mesh.
 *
 * Setup overview:
 *   1. Load a TET4 beam mesh.
 *   2. Create GPU_LDPM4_Data from the mesh (edges are extracted automatically).
 *   3. Set LDPM material parameters (E_N, E_T) and particle density.
 *   4. Fix nodes on the x = x_min face (cantilever).
 *   5. Apply a downward distributed load across all nodes on the x = x_max face.
 *   6. Build the diagonal lumped-mass matrix (CalcMassMatrix).
 *   7. Create LeapfrogSolver, set parameters, call Setup.
 *   8. Time-march with Solve() and write VTK snapshots every N steps.
 *
 * VTK output
 * ──────────
 *   Each snapshot is written as output_ldpm_<frame>.vtk (legacy ASCII VTK).
 *   Load the series in ParaView with File → Open, select the group, and use
 *   the Animation View to step through frames.
 *
 * CFL stability note
 * ──────────────────
 *   dt_crit = l_min / c_N   where   c_N = sqrt(E_N / rho)
 *
 * For LDPM, E_N is the facet normal modulus (not Young's modulus), and c_N is
 * the "wave speed" propagating through the discrete lattice.  A safety factor
 * of 0.5 is used here.
 */

#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <MoPhiEssentials.h>

#include "../src/FEASolver.h"
#include "../src/elements/LDPM4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
#include "../src/utils/mesh_utils.h"
#include "../src/utils/quadrature_utils.h"

using namespace tlfea;

// ── Material and simulation parameters ────────────────────────────────────────
// Normal modulus (Pa):  E_N drives axial stiffness of each interaction facet.
// Shear modulus (Pa):   E_T = alpha * E_N controls shear / bending stiffness.
// alpha = 0.25 is a typical value for quasi-isotropic LDPM response.
const Real E_N_val = 7e7;
const Real alpha_t = 0.25;
const Real E_T_val = alpha_t * E_N_val;
const Real rho_val = 2700.0;  // kg/m³

// Fraction of beam length used as tolerance for detecting end nodes.
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
    // 2. Identify fixed nodes (clamp at x = x_min)
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
    // 3. Identify loaded nodes (all particles on x = x_max face)
    //
    // Note: a single arbitrary corner node would carry a disproportionately
    // small lumped mass (few connected elements), turning the same applied
    // force into a much higher acceleration than the surrounding nodes.
    // Distributing the resultant evenly across the entire free-end face
    // gives a physically correct distributed tip load and avoids the spike.
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
    // 4. Build and configure GPU_LDPM4_Data
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPM4_Data element_data(n_nodes, n_elems);

    // Setup extracts unique edges, computes reference geometry and facet areas.
    element_data.Setup(h_x, h_y, h_z, elements);

    MOPHI_INFO("LDPM4 element data set up: %d edges extracted", element_data.n_edge);

    element_data.SetNodalFixed(h_fixed);

    // External force: 1 kN downward distributed across all tip-face particles
    const Real F_total = -1000.0;  // N, in -z direction
    const Real F_per_node = F_total / static_cast<Real>(load_idx.size());
    VectorXR h_f_ext(n_nodes * 3);
    h_f_ext.setZero();
    for (int i : load_idx) {
        h_f_ext(i * 3 + 2) = F_per_node;
    }
    element_data.SetExternalForce(h_f_ext);

    element_data.SetMaterial(E_N_val, E_T_val);
    element_data.SetDensity(rho_val);

    // ──────────────────────────────────────────────────────────────────────────
    // 5. Build diagonal lumped mass matrix
    // ──────────────────────────────────────────────────────────────────────────
    element_data.CalcMassMatrix();
    MOPHI_INFO("CalcMassMatrix done");

    // ──────────────────────────────────────────────────────────────────────────
    // 6. Estimate CFL time step
    //
    //   c_N    = sqrt(E_N / rho)          wave speed in the discrete lattice
    //   l_min  = shortest edge length     (computed below from mesh geometry)
    //   dt_crit = l_min / c_N
    //   dt      = safety_factor * dt_crit
    // ──────────────────────────────────────────────────────────────────────────
    Real c_N = std::sqrt(E_N_val / rho_val);
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
    const Real dt_crit = l_min / c_N;
    const Real safety_factor = 0.5;
    const Real dt = safety_factor * dt_crit;

    MOPHI_INFO("c_N = %.2e m/s,  l_min = %.2e m,  dt_crit = %.2e s,  dt = %.2e s", c_N, l_min, dt_crit, dt);

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Create and configure LeapfrogSolver
    // ──────────────────────────────────────────────────────────────────────────
    LeapfrogSolver solver(&element_data);

    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    MOPHI_INFO("LeapfrogSolver configured (dt = %.2e s)", dt);

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Time-march and write VTK snapshots
    // ──────────────────────────────────────────────────────────────────────────
    const int n_steps = 2000;
    const int output_interval = 50;  // write one VTK frame every N steps

    // Reference node matrix needed by WriteFEAT4ToVTK (initial positions).
    MatrixXR nodes_ref(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        nodes_ref(i, 0) = h_x(i);
        nodes_ref(i, 1) = h_y(i);
        nodes_ref(i, 2) = h_z(i);
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Starting LDPM4 leapfrog time integration for " << n_steps << " steps...\n";
    std::cout << "VTK snapshots will be written every " << output_interval << " steps.\n";

    int frame = 0;

    // Write initial (undeformed) state as frame 0.
    {
        std::ostringstream fname;
        fname << "output_ldpm_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
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
            fname << "output_ldpm_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
            ANCFCPUUtils::WriteFEAT4ToVTK(fname.str(), nodes_ref, elements, x_cur, y_cur, z_cur);
            std::cout << "Wrote: " << fname.str() << "\n";
            ++frame;
        }
    }

    std::cout << "\nSimulation complete.  " << frame << " VTK file(s) written.\n";
    std::cout << "Open output_ldpm_*.vtk in ParaView to visualize the deformed configuration.\n";

    element_data.Destroy();
    MOPHI_INFO("Done");
    return 0;
}
