/**
 * ldpm_singletet_case1_velbc.cc
 *
 * LDPM single regular tetrahedron benchmark (Case 1 from the uploaded document)
 * using LeapfrogSolver with prescribed translational velocity BC.
 *
 * Mesh:
 *   data/meshes/LDPMTet4/SingleTet/LDPM_debugRegTet-*.dat
 *
 * Required output series:
 *   ldpm_singletet_case1_tet4_NNNNN.vtk
 *   ldpm_singletet_case1_subfacets_NNNNN.vtk
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

using namespace feris;

namespace {

// Table-4 parameters used by current LDPMTet4 implementation (mm-tonne-s).
static constexpr Real RHO_VAL = Real(2.40e-9);  // tonne/mm^3
static constexpr Real E_N_VAL = Real(30000.0);  // MPa = N/mm^2
static constexpr Real ALPHA_T = Real(0.2);      // E_T / E_N
static constexpr Real BETA_K = Real(0.25);      // rotational coupling
static constexpr Real SIGMA_T_VAL = Real(3.0);  // MPa
static constexpr Real G_T_VAL = Real(0.0375);   // N/mm

// Case-1 displacement history (Table 5), node index 1 in +x.
static constexpr std::array<Real, 5> kTimePts = {Real(0.0), Real(0.2), Real(2.4), Real(4.6), Real(4.8)};
static constexpr std::array<Real, 5> kDispPts = {Real(0.0), Real(0.02), Real(-0.2), Real(0.02), Real(0.0)};

static constexpr Real T_SIM = Real(4.8);
static constexpr Real T_VTK = Real(0.1);

Real Case1VelocityX(Real t) {
    if (t >= kTimePts.back())
        return Real(0);

    for (size_t i = 0; i + 1 < kTimePts.size(); ++i) {
        if (t >= kTimePts[i] && t < kTimePts[i + 1]) {
            const Real dt = kTimePts[i + 1] - kTimePts[i];
            return (kDispPts[i + 1] - kDispPts[i]) / dt;
        }
    }
    return Real(0);
}

std::string TetVtkName(int frame) {
    std::ostringstream s;
    s << "ldpm_singletet_case1_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
    return s.str();
}

std::string SubfacetVtkName(int frame) {
    std::ostringstream s;
    s << "ldpm_singletet_case1_subfacets_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
    return s.str();
}

}  // namespace

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    const std::string prefix = "data/meshes/LDPMTet4/SingleTet/LDPM_debugRegTet";
    LDPMTet4Mesh mesh;
    std::string err;
    if (!ReadLDPMTet4MeshFromFiles(prefix, mesh, &err)) {
        std::cerr << "Error loading single-tet mesh: " << err << "\n";
        return 1;
    }

    if (mesh.n_particles < 4 || mesh.n_tets < 1) {
        std::cerr << "Unexpected single-tet mesh size: particles=" << mesh.n_particles << ", tets=" << mesh.n_tets
                  << "\n";
        return 1;
    }

    // Characteristic length from minimum tet edge.
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

    const Real E_T_VAL = ALPHA_T * E_N_VAL;
    const Real E_kT_VAL = BETA_K * E_N_VAL * l_min * l_min;
    const Real E_kM_VAL = E_kT_VAL;
    const Real E_kL_VAL = E_kT_VAL;
    const Real H_T_VAL = l_min * SIGMA_T_VAL / G_T_VAL;

    GPU_LDPMTet4_Data element_data;
    element_data.SetupFromMesh(mesh);
    element_data.SetMaterial(E_N_VAL, E_T_VAL, E_kT_VAL, E_kM_VAL, E_kL_VAL);
    element_data.SetDamageParams(SIGMA_T_VAL, H_T_VAL);
    element_data.SetDensity(RHO_VAL);

    VectorReal3 h_f_ext(static_cast<size_t>(mesh.n_particles), Real3::Zero());
    element_data.SetExternalForce(h_f_ext);

    // Case 1 constraints:
    // - node 1 is driven in x only
    // - all other translational and rotational DOFs are fixed to zero
    // To also constrain rotation of node 1, mark all nodes fixed and re-impose
    // node-1 translational prescribed velocity [vx,0,0] in the solver.
    VectorXi h_fixed(4);
    h_fixed << 0, 1, 2, 3;
    element_data.SetNodalFixed(h_fixed);

    element_data.CalcMassMatrix();

    const Real c_N = std::sqrt(E_N_VAL / RHO_VAL);
    const Real dt = Real(0.5) * (l_min / c_N);
    const int n_steps = static_cast<int>(std::ceil(T_SIM / dt));
    const int vtk_interval = std::max(1, static_cast<int>(std::round(T_VTK / dt)));

    // Ensure outputs at key tabulated times in addition to regular cadence.
    std::vector<int> key_steps;
    key_steps.reserve(kTimePts.size());
    for (Real t_key : kTimePts) {
        if (t_key <= Real(0))
            continue;
        const int s = std::max(1, std::min(n_steps, static_cast<int>(std::round(t_key / dt))));
        key_steps.push_back(s);
    }
    std::sort(key_steps.begin(), key_steps.end());
    key_steps.erase(std::unique(key_steps.begin(), key_steps.end()), key_steps.end());

    LeapfrogSolver solver(&element_data);
    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    VectorXi h_driven(1);
    h_driven(0) = 1;
    VectorReal3 h_driven_vel(1, Real3::Zero());
    h_driven_vel[0](0) = Case1VelocityX(Real(0));

    solver.SetPrescribedVelocityBC(h_driven, h_driven_vel);
    solver.SetNodalVelocity(h_driven, h_driven_vel);
    solver.InitialHalfKick();

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Single-tet Case 1: dt=" << dt << " s, steps=" << n_steps << "\n";
    std::cout << "  E_N=" << E_N_VAL << ", E_T=" << E_T_VAL << ", H_t=" << H_T_VAL << ", l_min=" << l_min << "\n";

    int frame = 0;
    {
        if (!WriteLDPMTet4TetMeshToVTK(TetVtkName(frame), mesh, mesh.particle_x, mesh.particle_y, mesh.particle_z)) {
            std::cerr << "Failed writing initial tet VTK\n";
            return 1;
        }
        VectorXR crack0 = VectorXR::Zero(mesh.n_subfacets);
        if (!WriteLDPMTet4SubfacetMeshToVTK(SubfacetVtkName(frame), mesh, "crack_distance", crack0)) {
            std::cerr << "Failed writing initial subfacet VTK\n";
            return 1;
        }
        ++frame;
    }

    Real last_vx = h_driven_vel[0](0);

    for (int step = 0; step < n_steps; ++step) {
        const Real t_n = static_cast<Real>(step) * dt;
        const Real v_now = Case1VelocityX(t_n);
        if (std::abs(v_now - last_vx) > Real(1e-15)) {
            h_driven_vel[0](0) = v_now;
            solver.SetPrescribedVelocityBC(h_driven, h_driven_vel);
            last_vx = v_now;
        }

        solver.Solve();

        const int step_1 = step + 1;
        const bool periodic_output = (step_1 % vtk_interval == 0);
        const bool key_output = std::binary_search(key_steps.begin(), key_steps.end(), step_1);
        const bool last_output = (step_1 == n_steps);

        if (periodic_output || key_output || last_output) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            if (!WriteLDPMTet4TetMeshToVTK(TetVtkName(frame), mesh, x_cur, y_cur, z_cur)) {
                std::cerr << "Failed writing tet VTK at step " << step_1 << "\n";
                return 1;
            }

            VectorXR sf_crack;
            element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
            if (!WriteLDPMTet4SubfacetMeshToVTK(SubfacetVtkName(frame), mesh, "crack_distance", sf_crack)) {
                std::cerr << "Failed writing subfacet VTK at step " << step_1 << "\n";
                return 1;
            }

            const Real t_out = static_cast<Real>(step_1) * dt;
            const Real ux_node1 = x_cur(1) - mesh.particle_x(1);
            std::cout << "frame=" << std::setw(4) << frame << " step=" << std::setw(8) << step_1 << " t=" << t_out
                      << " ux(node1)=" << ux_node1 << " vx_target=" << last_vx << "\n";
            ++frame;
        }
    }

    element_data.Destroy();

    std::cout << "Done. Wrote " << frame << " frame(s).\n";
    std::cout << "  ldpm_singletet_case1_tet4_NNNNN.vtk\n";
    std::cout << "  ldpm_singletet_case1_subfacets_NNNNN.vtk\n";
    return 0;
}
