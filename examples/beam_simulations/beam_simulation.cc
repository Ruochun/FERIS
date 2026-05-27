/**
 * Beam Simulation
 *
 * Author: GitHub Copilot Agent
 *
 * This example demonstrates loading a mesh from a VTU file and performing
 * a realistic engineering analysis:
 * 1. Load a tetrahedral mesh (TET10 elements) from beam.vtu using MoPhiEssentials
 * 2. Apply boundary conditions (fix one end at x≈0, apply load at x≈10)
 * 3. Run a dynamic simulation using FERIS's solvers
 * 4. Export the results to VTK files for visualization
 *
 * This simulates a cantilever beam with one end fixed and a concentrated load
 * applied to the free end, similar to a structural engineering test setup.
 */

#include <cuda_runtime.h>

#include <MoPhiEssentials.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../src/FEASolver.h"
#include "../src/elements/FEAT10Data.cuh"
#include "../src/solvers/SyncedAdamW.cuh"
#include "../src/types.h"
#include "../src/utils/cpu_utils.h"
#include "../src/utils/mesh_utils.h"
#include "../src/utils/quadrature_utils.h"

using namespace feris;

// Material properties for aluminum
const Real E = 7e10;     // Young's modulus: 7e10 Pa (70 GPa)
const Real nu = 0.33;    // Poisson's ratio
const Real rho0 = 2700;  // Density (kg/m³)

// Simulation parameters
const Real TOTAL_TIME = 1.;                                // Total simulation time
const Real STEP_SIZE = 1e-3;                               // Time step size
const int N_TIMESTEPS = (int)(TOTAL_TIME / STEP_SIZE);     // Number of timesteps to simulate
const int FPS = 100;                                       // Output frame per sec
const int OUTPUT_FREQUENCY = (int)(1. / STEP_SIZE) / 100;  // Output every N timesteps
const int TOTAL_FRAME = (int)(TOTAL_TIME * FPS);           // Total number of frames output

int main() {
    const std::string output_dir = "beam_simulation";
    std::filesystem::remove_all(output_dir);
    if (!std::filesystem::create_directories(output_dir)) {
        std::cerr << "Failed to create output directory: " << output_dir << std::endl;
        return 1;
    }

    std::cout << "=======================================================" << std::endl;
    std::cout << "  Beam Simulation" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // ==========================================================================
    // Load mesh from VTU file using MoPhiEssentials
    // ==========================================================================

    std::string vtu_filename = "data/meshes/T10/beam.vtu";
    std::cout << "Loading mesh from: " << vtu_filename << std::endl;

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading VTU file: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Mesh loaded successfully:" << std::endl;
    std::cout << "  Nodes: " << mesh.NumLocalNodes() << std::endl;
    std::cout << "  TET4 elements: " << mesh.NumOwnedTets() << std::endl;
    std::cout << "  TET10 elements: " << mesh.NumOwnedTet10s() << std::endl;
    std::cout << "  HEX8 elements: " << mesh.NumOwnedHexes() << std::endl;

    // ==========================================================================
    // Extract mesh data for FERIS
    // ==========================================================================

    // Check that we have tet10 elements (required for FEAT10)
    if (mesh.NumOwnedTet10s() == 0) {
        std::cerr << "Error: No TET10 elements found in '" << vtu_filename << "'." << std::endl;
        std::cerr << "Found: " << mesh.NumOwnedTets() << " TET4, " << mesh.NumOwnedTet10s() << " TET10, "
                  << mesh.NumOwnedHexes() << " HEX8 elements." << std::endl;
        std::cerr << "FEAT10 requires 10-node tetrahedral elements (VTK cell type 24)." << std::endl;
        std::cerr << "Please verify the mesh format. See examples/README_BEAM.md for requirements." << std::endl;
        return 1;
    }

    int n_nodes = mesh.NumLocalNodes();
    int n_elems = mesh.NumOwnedTet10s();

    // Convert mophi::Mesh geometry to Eigen format
    MatrixXR nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        const auto& node = mesh.geom.nodes[i];
        nodes(i, 0) = node.x();
        nodes(i, 1) = node.y();
        nodes(i, 2) = node.z();
    }

    // Convert mophi::Mesh connectivity to Eigen format
    MatrixXi elements(n_elems, 10);
    for (int i = 0; i < n_elems; i++) {
        const auto& elem = mesh.topo.tet10s[i];
        for (int j = 0; j < 10; j++) {
            elements(i, j) = elem[j];
        }
    }

    std::cout << "Converted mesh data to FERIS format" << std::endl;

    // ==========================================================================
    // Initialize GPU data structure
    // ==========================================================================

    GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
    GPU_FEAT10_Data* beam = &gpu_t10_data;

    beam->Initialize();

    // Extract coordinate vectors from nodes matrix
    VectorXR h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        h_x12(i) = nodes(i, 0);  // X coordinates
        h_y12(i) = nodes(i, 1);  // Y coordinates
        h_z12(i) = nodes(i, 2);  // Z coordinates
    }

    // ==========================================================================
    // Apply boundary conditions
    // ==========================================================================

    // Find the x-coordinate range to determine beam ends
    Real x_min = h_x12.minCoeff();
    Real x_max = h_x12.maxCoeff();
    Real x_range = x_max - x_min;

    std::cout << "Beam x-coordinate range: [" << x_min << ", " << x_max << "]" << std::endl;

    // Fix nodes at x ≈ 0 (the left end of the beam)
    std::vector<int> fixed_node_indices;
    Real fixed_tolerance = 0.1 * x_range;  // 10% of beam length
    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_min) < fixed_tolerance) {
            fixed_node_indices.push_back(i);
        }
    }

    VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
        h_fixed_nodes(i) = fixed_node_indices[i];
    }

    std::cout << "Boundary conditions: Fixed " << fixed_node_indices.size() << " nodes at x ≈ " << x_min << std::endl;
    beam->SetNodalFixed(h_fixed_nodes);

    // ==========================================================================
    // Apply external forces
    // ==========================================================================

    VectorReal3 h_f_ext(static_cast<size_t>(beam->get_n_coef()));
    for (auto& f : h_f_ext)
        f = Real3::Zero();

    // Apply a concentrated load at x ≈ x_max (the right end of the beam)
    // This simulates a load applied to the free end of a cantilever beam
    std::vector<int> loaded_node_indices;
    Real load_tolerance = 0.1 * x_range;  // 10% of beam length
    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_max) < load_tolerance) {
            loaded_node_indices.push_back(i);
        }
    }

    // Apply a downward force in the -z direction
    // Coordinate system: x=axial (beam length), y=width, z=height (positive up)
    Real total_load = -1000.0;  // Total load in Newtons (negative = downward in z)
    Real load_per_node = total_load / loaded_node_indices.size();

    for (int i : loaded_node_indices) {
        h_f_ext[static_cast<size_t>(i)](2) = load_per_node;  // Apply force in z-direction (index 2)
    }

    beam->SetExternalForce(h_f_ext);
    std::cout << "External forces: Applied " << total_load << " N load to " << loaded_node_indices.size()
              << " nodes at x ≈ " << x_max << std::endl;

    // ==========================================================================
    // Setup material and element properties
    // ==========================================================================

    // Get quadrature data
    const VectorXR& tet5pt_x_host = Quadrature::tet5pt_x;
    const VectorXR& tet5pt_y_host = Quadrature::tet5pt_y;
    const VectorXR& tet5pt_z_host = Quadrature::tet5pt_z;
    const VectorXR& tet5pt_weights_host = Quadrature::tet5pt_weights;

    // Setup element data
    beam->Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host, tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

    beam->SetDensity(rho0);
    beam->SetDamping(0.0, 0.1);  // Add some damping for stability
    beam->SetSVK(E, nu);         // St. Venant-Kirchhoff material model

    std::cout << "Material properties: E=" << (E / 1e9) << " GPa, nu=" << nu << ", rho=" << rho0 << " kg/m³"
              << std::endl;

    // ==========================================================================
    // Compute reference configuration data
    // ==========================================================================

    beam->CalcDnDuPre();
    beam->CalcMassMatrix();
    beam->CalcConstraintData();
    beam->ConvertToCSR_ConstraintJacT();

    std::cout << "Reference configuration computed" << std::endl;

    // ==========================================================================
    // Setup solver
    // ==========================================================================

    // lr, beta1, beta2, eps, weight_decay, lr_decay, inner_tol, outer_tol, rho,
    // max_outer, max_inner, time_step, convergence_check_interval, inner_rtol
    SyncedAdamWParams params = {2e-4, 0.9, 0.999, 1e-8, 1e-4, 0.995, 1e-1, 1e-6, 1e14, 5, 300, STEP_SIZE, 10, 0.0};

    SyncedAdamWSolver solver(beam, beam->get_n_constraint());

    solver.Setup();
    solver.SetParameters(&params);

    std::cout << "Solver initialized: SyncedAdamW with h=" << std::scientific << params.time_step << std::defaultfloat
              << std::endl;

    // ==========================================================================
    // Run simulation and output results
    // ==========================================================================

    std::cout << "Starting simulation: " << N_TIMESTEPS << " timesteps, output every " << OUTPUT_FREQUENCY << " steps"
              << std::endl;
    int curr_frame = 0;

    // Retrieve mass CSR once — it is constant throughout the simulation
    std::vector<int> moff, mcols;
    std::vector<Real> mvals;
    beam->RetrieveMassCSRToCPU(moff, mcols, mvals);

    // Print energy table header
    std::cout << "\n"
              << std::setw(10) << "Time [s]" << std::setw(18) << "KE [J]" << std::setw(18) << "SE [J]" << std::setw(18)
              << "Total E [J]" << "\n"
              << std::string(64, '-') << "\n";

    // Output initial configuration
    {
        VectorXR x12, y12, z12;
        beam->RetrievePositionToCPU(x12, y12, z12);
        std::stringstream ss;
        ss << output_dir << "/output_beam_" << std::setfill('0') << std::setw(5) << 0 << ".vtk";
        ANCFCPUUtils::WriteFEAT10ToVTK(ss.str(), nodes, elements, x12, y12, z12);
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) << 0.0 << std::setw(18) << std::scientific
                  << std::setprecision(6) << 0.0 << std::setw(18) << 0.0 << std::setw(18) << 0.0 << "  [frame 0 → "
                  << ss.str() << "]\n";
    }

    // Buffers reused across output frames
    VectorXR x_before, y_before, z_before;

    // Run simulation loop
    for (int step = 1; step <= N_TIMESTEPS; step++) {
        bool output_this_step = (step % OUTPUT_FREQUENCY == 0);

        // Save positions just before solving so we can estimate velocity at output frames
        if (output_this_step) {
            beam->RetrievePositionToCPU(x_before, y_before, z_before);
        }

        solver.Solve();

        // Output at specified frequency
        if (output_this_step) {
            curr_frame++;
            Real t = step * STEP_SIZE;

            VectorXR x12, y12, z12;
            beam->RetrievePositionToCPU(x12, y12, z12);

            // Write VTK
            std::stringstream filename;
            filename << output_dir << "/output_beam_" << std::setfill('0') << std::setw(5) << curr_frame << ".vtk";
            ANCFCPUUtils::WriteFEAT10ToVTK(filename.str(), nodes, elements, x12, y12, z12);

            // Estimate velocity: v ≈ (x_{n+1} − x_n) / h
            int n_dof = n_nodes * 3;
            VectorXR v_cpu(n_dof);
            for (int i = 0; i < n_nodes; ++i) {
                v_cpu(3 * i + 0) = (x12(i) - x_before(i)) / STEP_SIZE;
                v_cpu(3 * i + 1) = (y12(i) - y_before(i)) / STEP_SIZE;
                v_cpu(3 * i + 2) = (z12(i) - z_before(i)) / STEP_SIZE;
            }

            // KE = (1/2) v^T M v  (exact, using mass CSR)
            Real KE = 0.0;
            for (int ni = 0; ni < n_nodes; ++ni) {
                for (int idx = moff[ni]; idx < moff[ni + 1]; ++idx) {
                    int nj = mcols[idx];
                    Real M_ij = mvals[idx];
                    for (int d = 0; d < 3; ++d)
                        KE += 0.5 * M_ij * v_cpu(3 * ni + d) * v_cpu(3 * nj + d);
                }
            }

            // SE = (1/2) f_int · (h·v) = (1/2) f_int · Δx  (linearized work estimate)
            VectorReal3 f_int_cpu;
            beam->RetrieveInternalForceToCPU(f_int_cpu);
            Real SE = 0.0;
            for (int i = 0; i < beam->get_n_coef(); ++i) {
                const Real3& f_i = f_int_cpu[static_cast<size_t>(i)];
                SE += 0.5 * f_i(0) * STEP_SIZE * v_cpu(i * 3 + 0);
                SE += 0.5 * f_i(1) * STEP_SIZE * v_cpu(i * 3 + 1);
                SE += 0.5 * f_i(2) * STEP_SIZE * v_cpu(i * 3 + 2);
            }

            Real TE = KE + SE;

            std::cout << std::setw(10) << std::fixed << std::setprecision(4) << t << std::setw(18) << std::scientific
                      << std::setprecision(6) << KE << std::setw(18) << SE << std::setw(18) << TE << "  [frame "
                      << curr_frame << "/" << TOTAL_FRAME << " → " << filename.str() << "]\n";
        }
    }

    // ==========================================================================
    // Cleanup
    // ==========================================================================

    beam->Destroy();

    std::cout << "\n=======================================================\n"
              << "  Simulation Complete!\n"
              << "  ParaView output: " << output_dir << "/output_beam_*.vtk\n"
              << "=======================================================\n";

    return 0;
}
