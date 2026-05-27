/**
 * TET4 Leapfrog Dynamic Simulation Example
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * This driver loads a TET4 beam mesh, clamps nodes on the x = x_min plane,
 * applies a downward point load on one node, and advances the simulation
 * using the explicit leapfrog (central-difference) solver. It demonstrates
 * the full setup workflow intended for LDPM models built on TET4 elements:
 *
 *   1. Read mesh, build GPU_FEAT4_Data
 *   2. Set material (SVK), density, and optional damping
 *   3. CalcDnDuPre() and CalcMassMatrix()
 *   4. Create LeapfrogSolver, call Setup() + SetParameters()
 *   5. Time-march with Solve()
 *   6. Retrieve and print nodal positions
 */

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <MoPhiEssentials.h>

#include "../src/FEASolver.h"
#include "../src/elements/FEAT4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
#include "../src/utils/quadrature_utils.h"

using namespace feris;

// Material and geometry parameters
const Real E_mod = 7e7;    // Young's modulus (Pa)
const Real nu_val = 0.33;  // Poisson's ratio
const Real rho0 = 2700.0;  // Density (kg/m^3)
const Real eta_d = 0.0;    // Viscous damping coefficient (Rayleigh shear, 0 = undamped)
const Real lam_d = 0.0;    // Viscous damping coefficient (Rayleigh bulk,  0 = undamped)

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // -------------------------------------------------------------------------
    // 1. Load mesh
    // -------------------------------------------------------------------------
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

    VectorXR h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        h_x12(i) = mesh.geom.nodes[i].x();
        h_y12(i) = mesh.geom.nodes[i].y();
        h_z12(i) = mesh.geom.nodes[i].z();
    }

    // -------------------------------------------------------------------------
    // 2. Identify fixed nodes (clamp at x = x_min)
    // -------------------------------------------------------------------------
    Real x_min = h_x12.minCoeff();
    std::vector<int> fixed_idx;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x12(i) - x_min) < 1e-8) {
            fixed_idx.push_back(i);
        }
    }
    MOPHI_INFO("Fixed nodes (x = x_min): %zu", fixed_idx.size());

    VectorXi h_fixed_nodes(static_cast<int>(fixed_idx.size()));
    for (int i = 0; i < static_cast<int>(fixed_idx.size()); i++) {
        h_fixed_nodes(i) = fixed_idx[i];
    }

    // -------------------------------------------------------------------------
    // 3. Identify loaded node (first node at x = x_max)
    // -------------------------------------------------------------------------
    Real x_max = h_x12.maxCoeff();
    int load_node = -1;
    for (int i = 0; i < n_nodes; i++) {
        if (std::abs(h_x12(i) - x_max) < 1e-8) {
            load_node = i;
            break;
        }
    }
    if (load_node < 0) {
        MOPHI_ERROR("Could not find a node at x = x_max for load application.");
        return 1;
    }
    MOPHI_INFO("Load node: %d  (x = %.4f)", load_node, x_max);

    // -------------------------------------------------------------------------
    // 4. Build and configure GPU_FEAT4_Data
    // -------------------------------------------------------------------------
    GPU_FEAT4_Data element_data(n_elems, n_nodes);
    element_data.Initialize();

    element_data.SetNodalFixed(h_fixed_nodes);

    VectorReal3 h_f_ext(static_cast<size_t>(n_nodes));
    for (auto& f : h_f_ext)
        f = Real3::Zero();
    h_f_ext[static_cast<size_t>(load_node)](2) = -1000.0;  // 1 kN downward
    element_data.SetExternalForce(h_f_ext);

    const VectorXR& quad_x = Quadrature::tet1pt_x;
    const VectorXR& quad_y = Quadrature::tet1pt_y;
    const VectorXR& quad_z = Quadrature::tet1pt_z;
    const VectorXR& quad_w = Quadrature::tet1pt_weights;

    element_data.Setup(quad_x, quad_y, quad_z, quad_w, h_x12, h_y12, h_z12, elements);

    element_data.SetSVK(E_mod, nu_val);
    element_data.SetDensity(rho0);
    element_data.SetDamping(eta_d, lam_d);

    // -------------------------------------------------------------------------
    // 5. Precompute reference quantities and consistent mass matrix
    // -------------------------------------------------------------------------
    element_data.CalcDnDuPre();
    MOPHI_INFO("CalcDnDuPre done");

    element_data.CalcMassMatrix();
    MOPHI_INFO("CalcMassMatrix done");

    // -------------------------------------------------------------------------
    // 6. Create and configure LeapfrogSolver
    // -------------------------------------------------------------------------
    LeapfrogSolver solver(&element_data);

    // Critical time step for stability. For an SVK material with density rho
    // and Lamé parameters lambda, mu the dilatational wave speed is:
    //   c_d = sqrt((lambda + 2*mu) / rho)
    // The critical time step is dt_crit = h_min / c_d, where h_min is the
    // smallest characteristic element length (shortest edge / sqrt(6) for TET4).
    // Choose dt < dt_crit; a safety factor of 0.5–0.9 is common in practice.
    const Real dt = 1.0e-6;  // 1 µs – must be set below the CFL limit
    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    MOPHI_INFO("LeapfrogSolver created and configured (dt = %.2e s)", dt);

    // -------------------------------------------------------------------------
    // 7. Time-march
    // -------------------------------------------------------------------------
    const int n_steps = 200;
    const int print_interval = 50;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Starting leapfrog time integration for " << n_steps << " steps...\n";

    for (int step = 0; step < n_steps; step++) {
        solver.Solve();

        if ((step + 1) % print_interval == 0) {
            VectorXR x12, y12, z12;
            element_data.RetrievePositionToCPU(x12, y12, z12);
            std::cout << "Step " << step + 1 << " | load node displacement:"
                      << "  dx = " << (x12(load_node) - h_x12(load_node))
                      << "  dy = " << (y12(load_node) - h_y12(load_node))
                      << "  dz = " << (z12(load_node) - h_z12(load_node)) << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // 8. Print final nodal positions
    // -------------------------------------------------------------------------
    VectorXR x12_final, y12_final, z12_final;
    element_data.RetrievePositionToCPU(x12_final, y12_final, z12_final);

    std::cout << std::fixed << std::setprecision(17);
    std::cout << "\nFinal x12:\n";
    for (int i = 0; i < x12_final.size(); i++) {
        std::cout << x12_final(i) << " ";
    }
    std::cout << "\n\nFinal y12:\n";
    for (int i = 0; i < y12_final.size(); i++) {
        std::cout << y12_final(i) << " ";
    }
    std::cout << "\n\nFinal z12:\n";
    for (int i = 0; i < z12_final.size(); i++) {
        std::cout << z12_final(i) << " ";
    }
    std::cout << "\n";

    element_data.Destroy();
    MOPHI_INFO("Done");
    return 0;
}
