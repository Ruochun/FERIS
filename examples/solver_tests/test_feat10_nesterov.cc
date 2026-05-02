/**
 * FEAT10 Cube Nesterov Test
 *
 * Author: Json Zhou
 * Email:  zzhou292@wisc.edu
 *
 * This driver loads a FEAT10 cube mesh, clamps nodes on the z=0 plane,
 * applies a point load, and advances the system using the synchronized
 * Nesterov solver. It is intended to test FEAT10 mass and internal force
 * assembly together with the Nesterov integrator on a simple solid block.
 */

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <MoPhiEssentials.h>

#include "../src/FEASolver.h"
#include "../src/elements/FEAT10Data.cuh"
#include "../src/solvers/SyncedNesterov.cuh"
#include "../src/types.h"
#include "../src/utils/cpu_utils.h"
#include "../src/utils/quadrature_utils.h"

using namespace feris;

const Real E = 7e8;      // Young's modulus
const Real nu = 0.33;    // Poisson's ratio
const Real rho0 = 2700;  // Density

int main() {
    // Initialize MoPhiEssentials logging
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // Read mesh data
    MatrixXR nodes;
    MatrixXi elements;

    int n_nodes = ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/cube.1.node", nodes);
    int n_elems = ANCFCPUUtils::FEAT10_read_elements("data/meshes/T10/cube.1.ele", elements);

    MOPHI_INFO("mesh read nodes: %d", n_nodes);
    MOPHI_INFO("mesh read elements: %d", n_elems);

    // print nodes and elements matrix
    std::cout << "nodes matrix:" << std::endl;
    std::cout << nodes << std::endl;
    std::cout << "elements matrix:" << std::endl;
    std::cout << elements << std::endl;

    GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
    GPU_FEAT10_Data* cube = &gpu_t10_data;

    MOPHI_INFO("gpu_t10_data created");

    cube->Initialize();

    MOPHI_INFO("gpu_t10_data initialized");

    // Extract coordinate vectors from nodes matrix
    VectorXR h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        h_x12(i) = nodes(i, 0);  // X coordinates
        h_y12(i) = nodes(i, 1);  // Y coordinates
        h_z12(i) = nodes(i, 2);  // Z coordinates
    }

    // ==========================================================================

    // Find all nodes with z == 0
    std::vector<int> fixed_node_indices;
    for (int i = 0; i < h_z12.size(); ++i) {
        if (std::abs(h_z12(i)) < 1e-8) {  // tolerance for floating point
            fixed_node_indices.push_back(i);
        }
    }

    // Convert to VectorXi
    VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
        h_fixed_nodes(i) = fixed_node_indices[i];
    }

    // print fixed nodes
    MOPHI_INFO("Fixed nodes (z == 0): %zu nodes", fixed_node_indices.size());
    for (int i = 0; i < h_fixed_nodes.size(); ++i) {
        std::cout << h_fixed_nodes(i) << " ";
    }
    std::cout << std::endl;

    // Set fixed nodes
    cube->SetNodalFixed(h_fixed_nodes);

    // set external force
    VectorXR h_f_ext(cube->get_n_coef() * 3);
    // set external force applied at the end of the beam to be 0,0,3100
    h_f_ext.setZero();
    h_f_ext(3 * 6 + 0) = 1000.0;
    cube->SetExternalForce(h_f_ext);

    // Get quadrature data from quadrature_utils.h
    const VectorXR& tet5pt_x_host = Quadrature::tet5pt_x;
    const VectorXR& tet5pt_y_host = Quadrature::tet5pt_y;
    const VectorXR& tet5pt_z_host = Quadrature::tet5pt_z;
    const VectorXR& tet5pt_weights_host = Quadrature::tet5pt_weights;

    // Call Setup with all required parameters
    cube->Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host, tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

    cube->SetDensity(rho0);
    cube->SetDamping(0.0, 0.0);

    cube->SetSVK(E, nu);

    // =========================================================================

    cube->CalcDnDuPre();

    MOPHI_INFO("gpu_t10_data dndu pre complete");

    // 2. Retrieve results
    std::vector<std::vector<MatrixXR>> ref_grads;
    cube->RetrieveDnDuPreToCPU(ref_grads);

    std::cout << "ref_grads:" << std::endl;
    for (size_t i = 0; i < ref_grads.size(); i++) {
        for (size_t j = 0; j < ref_grads[i].size(); j++) {
            std::cout << ref_grads[i][j] << std::endl;
        }
    }
    std::cout << "done retrieving ref_grads" << std::endl;

    std::vector<std::vector<Real>> detJ;
    cube->RetrieveDetJToCPU(detJ);

    std::cout << "detJ:" << std::endl;
    for (size_t i = 0; i < detJ.size(); i++) {
        for (size_t j = 0; j < detJ[i].size(); j++) {
            std::cout << detJ[i][j] << std::endl;
        }
    }
    MOPHI_INFO("done retrieving detJ");

    cube->CalcMassMatrix();

    cube->CalcConstraintData();

    MOPHI_INFO("done CalcConstraintData");

    cube->ConvertToCSR_ConstraintJacT();

    MOPHI_INFO("done ConvertToCSR_ConstraintJacT");

    // calculate p
    cube->CalcP();

    MOPHI_INFO("done CalcP");

    // retrieve p
    std::vector<std::vector<MatrixXR>> p_from_F;
    cube->RetrievePFromFToCPU(p_from_F);

    std::cout << "P matrices (First Piola-Kirchhoff stress):" << std::endl;
    for (size_t elem = 0; elem < p_from_F.size(); elem++) {
        std::cout << "Element " << elem << ":" << std::endl;
        for (size_t qp = 0; qp < p_from_F[elem].size(); qp++) {
            std::cout << "  Quadrature Point " << qp << ":" << std::endl;
            std::cout << p_from_F[elem][qp] << std::endl;
        }
    }
    MOPHI_INFO("done retrieving P matrices");

    // calculate internal force
    cube->CalcInternalForce();
    MOPHI_INFO("done CalcInternalForce");

    // retrieve internal force
    VectorXR f_int;
    cube->RetrieveInternalForceToCPU(f_int);
    std::cout << "Internal force vector (size: " << f_int.size() << "):" << std::endl;
    std::cout << f_int.transpose() << std::endl;
    std::cout << "done retrieving internal force vector" << std::endl;

    SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300, 1.0e-3};
    SyncedNesterovSolver solver(cube, cube->get_n_constraint());

    solver.Setup();
    solver.SetParameters(&params);
    for (int i = 0; i < 50; i++) {
        solver.Solve();
    }

    // // Set highest precision for cout
    std::cout << std::fixed << std::setprecision(17);

    VectorXR x12, y12, z12;
    cube->RetrievePositionToCPU(x12, y12, z12);

    std::cout << "x12:" << std::endl;
    for (int i = 0; i < x12.size(); i++) {
        std::cout << x12(i) << " ";
    }

    std::cout << std::endl;

    std::cout << "y12:" << std::endl;
    for (int i = 0; i < y12.size(); i++) {
        std::cout << y12(i) << " ";
    }

    std::cout << std::endl;

    std::cout << "z12:" << std::endl;
    for (int i = 0; i < z12.size(); i++) {
        std::cout << z12(i) << " ";
    }

    std::cout << std::endl;

    cube->Destroy();

    std::cout << "gpu_t10_data destroyed" << std::endl;

    return 0;
}
