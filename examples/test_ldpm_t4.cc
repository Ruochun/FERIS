/**
 * LDPM4 Leapfrog Dynamic Simulation Example
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
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
 *   5. Apply a downward point load at the first node on the x = x_max face.
 *   6. Build the diagonal lumped-mass matrix (CalcMassMatrix).
 *   7. Create LeapfrogSolver, set parameters, call Setup.
 *   8. Time-march with Solve() and print tip displacement every N steps.
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
#include <MoPhiEssentials.h>

#include "../src/FEASolver.h"
#include "../src/elements/LDPM4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
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

    std::vector<int> fixed_idx;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x(i) - x_min) < 1e-8) {
            fixed_idx.push_back(i);
        }
    }
    MOPHI_INFO("Fixed particles (x = x_min = %.4f): %zu", x_min, fixed_idx.size());

    VectorXi h_fixed(static_cast<int>(fixed_idx.size()));
    for (int i = 0; i < static_cast<int>(fixed_idx.size()); i++) {
        h_fixed(i) = fixed_idx[i];
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Identify loaded node (first particle at x = x_max)
    // ──────────────────────────────────────────────────────────────────────────
    int load_node = -1;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x(i) - x_max) < 1e-8) {
            load_node = i;
            break;
        }
    }
    if (load_node < 0) {
        MOPHI_ERROR("Could not find a particle at x = x_max for load application.");
        return 1;
    }
    MOPHI_INFO("Load particle: %d  (x = %.4f)", load_node, x_max);

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Build and configure GPU_LDPM4_Data
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPM4_Data element_data(n_nodes, n_elems);

    // Setup extracts unique edges, computes reference geometry and facet areas.
    element_data.Setup(h_x, h_y, h_z, elements);

    MOPHI_INFO("LDPM4 element data set up: %d edges extracted", element_data.n_edge);

    element_data.SetNodalFixed(h_fixed);

    // External force: 1 kN downward at tip
    VectorXR h_f_ext(n_nodes * 3);
    h_f_ext.setZero();
    h_f_ext(load_node * 3 + 2) = -1000.0;  // -z direction
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
    // 8. Time-march
    // ──────────────────────────────────────────────────────────────────────────
    const int n_steps = 200;
    const int print_interval = 50;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Starting LDPM4 leapfrog time integration for " << n_steps << " steps...\n";

    for (int step = 0; step < n_steps; step++) {
        solver.Solve();

        if ((step + 1) % print_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);
            std::cout << "Step " << step + 1 << " | load node displacement:"
                      << "  dx = " << (x_cur(load_node) - h_x(load_node))
                      << "  dy = " << (y_cur(load_node) - h_y(load_node))
                      << "  dz = " << (z_cur(load_node) - h_z(load_node)) << "\n";
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 9. Print final nodal positions
    // ──────────────────────────────────────────────────────────────────────────
    VectorXR x_final, y_final, z_final;
    element_data.RetrievePositionToCPU(x_final, y_final, z_final);

    std::cout << std::fixed << std::setprecision(17);
    std::cout << "\nFinal x:\n";
    for (int i = 0; i < x_final.size(); i++) {
        std::cout << x_final(i) << " ";
    }
    std::cout << "\n\nFinal y:\n";
    for (int i = 0; i < y_final.size(); i++) {
        std::cout << y_final(i) << " ";
    }
    std::cout << "\n\nFinal z:\n";
    for (int i = 0; i < z_final.size(); i++) {
        std::cout << z_final(i) << " ";
    }
    std::cout << "\n";

    element_data.Destroy();
    MOPHI_INFO("Done");
    return 0;
}
