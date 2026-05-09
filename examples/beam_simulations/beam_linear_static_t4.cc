/**
 * beam_linear_static_t4.cc
 *
 * Steady-state (linear) FEA analysis of a TET4 cantilever beam.
 *
 * Physics
 * -------
 * Same setup as the TET10 linear static demo (beam_linear_static.cc):
 *   • Beam mesh loaded from data/meshes/T4/beam.vtu
 *   • Fixed end: all nodes with x ≈ x_min (one end clamped)
 *   • Free end:  concentrated downward load P = 1000 N distributed uniformly
 *                among nodes with x ≈ x_max
 *   • Material:  Aluminium — E = 70 GPa, ν = 0.33
 *
 * The LinearStaticSolver<GPU_FEAT4_Data> assembles the tangent stiffness
 * matrix K at the reference configuration and solves K·u = f once on the GPU
 * via conjugate gradient, giving the small-displacement solution directly.
 *
 * Validation
 * ----------
 * For a solid rectangular cantilever beam:
 *   δ_tip = P · L³ / (3 · E · I_y)
 * where I_y = b · h³ / 12 (bending about the y-axis, load in −z).
 * The FEA result is compared with this analytical Euler-Bernoulli value.
 */

#include <cuda_runtime.h>

#include <MoPhiEssentials.h>
#include <iomanip>
#include <iostream>

#include "../src/FEASolver.h"
#include "../src/elements/FEAT4Data.cuh"
#include "../src/solvers/LinearStaticSolver.cuh"
#include "../src/types.h"
#include "../src/utils/cpu_utils.h"
#include "../src/utils/mesh_utils.h"
#include "../src/utils/quadrature_utils.h"

using namespace feris;

// ---------------------------------------------------------------------------
// Material properties (Aluminium)
// ---------------------------------------------------------------------------
static const Real E_mod = 7e10;   // Young's modulus [Pa]
static const Real nu_val = 0.33;  // Poisson's ratio
static const Real rho0 = 2700.0;  // Density [kg/m³] (unused for static, kept for completeness)

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
static const Real TOTAL_LOAD = -1000.0;  // Total transverse force [N] (negative = −z direction)

// Fraction of beam length used as tolerance for detecting end nodes.
static constexpr Real BOUNDARY_TOLERANCE_FRACTION = 0.0005;

int main() {
    std::cout << "=======================================================\n"
              << "  Linear Static FEA – TET4 Cantilever Beam\n"
              << "=======================================================\n";

    // -----------------------------------------------------------------------
    // Load mesh from VTU file
    // -----------------------------------------------------------------------
    const std::string vtu_path = "data/meshes/T4/beam_highres.vtu";
    std::cout << "Loading mesh: " << vtu_path << "\n";

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading mesh: " << e.what() << "\n";
        return 1;
    }

    if (mesh.NumOwnedTets() == 0) {
        std::cerr << "No TET4 elements found in " << vtu_path << "\n";
        return 1;
    }

    const int n_nodes = mesh.NumLocalNodes();
    const int n_elems = static_cast<int>(mesh.NumOwnedTets());
    std::cout << "  Nodes: " << n_nodes << "   TET4 elements: " << n_elems << "\n";

    // -----------------------------------------------------------------------
    // Convert mesh to Eigen format
    // -----------------------------------------------------------------------
    MatrixXR nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i) {
        nodes(i, 0) = mesh.geom.nodes[i].x();
        nodes(i, 1) = mesh.geom.nodes[i].y();
        nodes(i, 2) = mesh.geom.nodes[i].z();
    }

    MatrixXi elements(n_elems, 4);
    for (int i = 0; i < n_elems; ++i)
        for (int j = 0; j < 4; ++j)
            elements(i, j) = mesh.topo.tets[i][j];

    VectorXR h_x(n_nodes), h_y(n_nodes), h_z(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        h_x(i) = nodes(i, 0);
        h_y(i) = nodes(i, 1);
        h_z(i) = nodes(i, 2);
    }

    // Determine beam bounding box.
    const Real x_min = h_x.minCoeff(), x_max = h_x.maxCoeff();
    const Real y_min = h_y.minCoeff(), y_max = h_y.maxCoeff();
    const Real z_min = h_z.minCoeff(), z_max = h_z.maxCoeff();
    const Real L = x_max - x_min;
    const Real b = y_max - y_min;  // cross-section width
    const Real h = z_max - z_min;  // cross-section height

    std::cout << std::fixed << std::setprecision(4) << "  Bounding box:  x∈[" << x_min << ", " << x_max << "]  "
              << "y∈[" << y_min << ", " << y_max << "]  "
              << "z∈[" << z_min << ", " << z_max << "]\n"
              << "  L = " << L << "  b = " << b << "  h = " << h << "\n";

    // -----------------------------------------------------------------------
    // Initialise GPU data structure
    // -----------------------------------------------------------------------
    GPU_FEAT4_Data gpu_data(n_elems, n_nodes);
    GPU_FEAT4_Data* beam = &gpu_data;

    beam->Initialize();

    const auto& qx = Quadrature::tet1pt_x;
    const auto& qy = Quadrature::tet1pt_y;
    const auto& qz = Quadrature::tet1pt_z;
    const auto& qw = Quadrature::tet1pt_weights;

    beam->Setup(qx, qy, qz, qw, h_x, h_y, h_z, elements);
    beam->SetDensity(rho0);
    beam->SetDamping(0.0, 0.0);  // no damping for static analysis
    beam->SetSVK(E_mod, nu_val);

    std::cout << "  Material: E = " << (E_mod / 1e9) << " GPa  ν = " << nu_val << "\n";

    // Pre-compute reference shape-function gradients.
    beam->CalcDnDuPre();

    // -----------------------------------------------------------------------
    // Boundary conditions – fix all nodes at x ≈ x_min
    // -----------------------------------------------------------------------
    const Real tol_bc = BOUNDARY_TOLERANCE_FRACTION * L;
    std::vector<int> fixed_idx;
    for (int i = 0; i < n_nodes; ++i)
        if (std::abs(h_x(i) - x_min) < tol_bc)
            fixed_idx.push_back(i);

    VectorXi h_fixed(static_cast<int>(fixed_idx.size()));
    for (int i = 0; i < static_cast<int>(fixed_idx.size()); ++i)
        h_fixed(i) = fixed_idx[i];

    beam->SetNodalFixed(h_fixed);
    std::cout << "  Fixed " << fixed_idx.size() << " nodes at x ≈ " << x_min << "\n";

    VectorReal3 h_f_ext(static_cast<size_t>(n_nodes));
    for (auto& f : h_f_ext)
        f = Real3::Zero();

    const Real tol_load = BOUNDARY_TOLERANCE_FRACTION * L;
    std::vector<int> load_idx;
    for (int i = 0; i < n_nodes; ++i)
        if (std::abs(h_x(i) - x_max) < tol_load)
            load_idx.push_back(i);

    const Real load_per_node = TOTAL_LOAD / static_cast<Real>(load_idx.size());
    for (int i : load_idx)
        h_f_ext[static_cast<size_t>(i)](2) = load_per_node;  // force in −z direction

    beam->SetExternalForce(h_f_ext);
    std::cout << "  Load: " << TOTAL_LOAD << " N distributed over " << load_idx.size() << " nodes at x ≈ " << x_max
              << "\n";

    // -----------------------------------------------------------------------
    // Run linear static solver
    // -----------------------------------------------------------------------
    std::cout << "\nRunning linear static solver (GPU conjugate gradient)...\n";

    LinearStaticSolver<GPU_FEAT4_Data> solver(beam, /*tol=*/1e-10, /*max_iter=*/100000);

    solver.Solve();

    std::cout << "  CG converged in " << solver.GetLastIterCount() << " iterations"
              << "  (relative residual: " << std::scientific << std::setprecision(3) << solver.GetLastResidual()
              << ")\n";

    // -----------------------------------------------------------------------
    // Retrieve displaced positions and compute nodal displacements
    // -----------------------------------------------------------------------
    VectorXR x12, y12, z12;
    beam->RetrievePositionToCPU(x12, y12, z12);

    // Displacement = deformed − reference positions.
    VectorXR ux = x12 - h_x;
    VectorXR uy = y12 - h_y;
    VectorXR uz = z12 - h_z;

    // Maximum displacement magnitude.
    Real max_disp = 0.0;
    for (int i = 0; i < n_nodes; ++i) {
        Real mag = std::sqrt(ux(i) * ux(i) + uy(i) * uy(i) + uz(i) * uz(i));
        max_disp = std::max(max_disp, mag);
    }

    // Tip deflection in z (maximum at free end).
    Real tip_uz_min = 0.0;
    for (int i : load_idx)
        tip_uz_min = std::min(tip_uz_min, uz(i));

    // -----------------------------------------------------------------------
    // Analytical comparison (Euler-Bernoulli cantilever beam)
    // -----------------------------------------------------------------------
    const Real I_y = b * h * h * h / 12.0;
    const Real delta_ana = std::abs(TOTAL_LOAD) * L * L * L / (3.0 * E_mod * I_y);

    std::cout << std::defaultfloat << std::setprecision(6);
    std::cout << "\n--- Results ---\n"
              << "  Max tip z-displacement (FEA):        " << std::scientific << std::setprecision(6)
              << std::abs(tip_uz_min) << " m\n"
              << "  Analytical Euler-Bernoulli δ_tip:    " << std::scientific << delta_ana << " m\n";

    const Real rel_err = std::abs(std::abs(tip_uz_min) - delta_ana) / delta_ana;
    std::cout << "  Relative error vs analytical:        " << std::fixed << std::setprecision(2) << rel_err * 100.0
              << " %\n";

    std::cout << "  Max nodal displacement magnitude:    " << std::scientific << std::setprecision(6) << max_disp
              << " m\n";

    // -----------------------------------------------------------------------
    // Write output VTK
    // -----------------------------------------------------------------------
    ANCFCPUUtils::WriteFEAT4ToVTK("output_beam_linear_static_t4.vtk", nodes, elements, x12, y12, z12);
    std::cout << "\n  Deformed mesh written to: output_beam_linear_static_t4.vtk\n";

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    beam->Destroy();

    std::cout << "=======================================================\n"
              << "  Linear Static Analysis (TET4) Complete\n"
              << "=======================================================\n";
    return 0;
}
