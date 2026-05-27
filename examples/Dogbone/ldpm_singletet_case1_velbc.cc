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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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
static constexpr Real T_CSV = Real(1.0e-3);

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

const int kLocalEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

std::string JoinPath(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir) / file).string();
}

}  // namespace

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);
    const std::string output_dir = "ldpm_singletet_case1_velbc";
    std::filesystem::remove_all(output_dir);
    if (!std::filesystem::create_directories(output_dir)) {
        std::cerr << "Failed to create output directory: " << output_dir << "\n";
        return 1;
    }

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
    const int csv_interval = std::max(1, static_cast<int>(std::round(T_CSV / dt)));

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
    std::cout << "  Output directory: " << output_dir << "\n";

    const std::string node_force_csv_name = JoinPath(output_dir, "ldpm_singletet_case1_node_forces.csv");
    const std::string subfacet_csv_name = JoinPath(output_dir, "ldpm_singletet_case1_subfacet_stress_strain.csv");
    std::ofstream node_force_csv(node_force_csv_name);
    std::ofstream subfacet_csv(subfacet_csv_name);
    if (!node_force_csv.is_open() || !subfacet_csv.is_open()) {
        std::cerr << "Failed to open CSV outputs in " << output_dir << "\n";
        return 1;
    }
    node_force_csv << "# time_s";
    for (int node = 0; node < mesh.n_particles; ++node) {
        const int node_label = node + 1;
        node_force_csv << ",node_" << node_label << "_f_mag" << ",node_" << node_label << "_f_x" << ",node_"
                       << node_label << "_f_y" << ",node_" << node_label << "_f_z";
    }
    node_force_csv << "\n";

    subfacet_csv << "# time_s";
    for (int sf = 0; sf < mesh.n_subfacets; ++sf) {
        const int sf_label = sf + 1;
        subfacet_csv << ",facet_" << sf_label << "_stress_n" << ",facet_" << sf_label << "_stress_m" << ",facet_"
                     << sf_label << "_stress_l" << ",facet_" << sf_label << "_strain_n" << ",facet_" << sf_label
                     << "_strain_m" << ",facet_" << sf_label << "_strain_l";
    }
    subfacet_csv << "\n";

    const std::vector<int>& edge_nodes = element_data.GetEdgeNodes();
    std::map<std::pair<int, int>, int> edge_index_map;
    for (int e = 0; e < element_data.n_edge; ++e) {
        const int ni = edge_nodes[2 * e + 0];
        const int nj = edge_nodes[2 * e + 1];
        edge_index_map[{std::min(ni, nj), std::max(ni, nj)}] = e;
    }

    std::vector<int> subfacet_edge_idx(static_cast<size_t>(mesh.n_subfacets), -1);
    std::vector<int> subfacet_node_i(static_cast<size_t>(mesh.n_subfacets), -1);
    std::vector<int> subfacet_node_j(static_cast<size_t>(mesh.n_subfacets), -1);
    for (int sf = 0; sf < mesh.n_subfacets; ++sf) {
        const int t = mesh.subfacet_tet(sf);
        const int local_sf = sf % 12;
        const int edge_slot = local_sf / 2;
        const int a = kLocalEdges[edge_slot][0];
        const int b = kLocalEdges[edge_slot][1];
        const int ni = mesh.tet_connectivity(t, a);
        const int nj = mesh.tet_connectivity(t, b);
        const std::pair<int, int> key = {std::min(ni, nj), std::max(ni, nj)};
        auto it = edge_index_map.find(key);
        if (it != edge_index_map.end())
            subfacet_edge_idx[static_cast<size_t>(sf)] = it->second;
        subfacet_node_i[static_cast<size_t>(sf)] = ni;
        subfacet_node_j[static_cast<size_t>(sf)] = nj;
    }

    auto write_csv_metrics = [&](Real time_s) {
        VectorReal3 f_int;
        element_data.RetrieveInternalForceToCPU(f_int);
        node_force_csv << time_s;
        for (int node = 0; node < mesh.n_particles; ++node) {
            const Real fx = f_int[static_cast<size_t>(node)](0);
            const Real fy = f_int[static_cast<size_t>(node)](1);
            const Real fz = f_int[static_cast<size_t>(node)](2);
            const Real fmag = std::sqrt(fx * fx + fy * fy + fz * fz);
            node_force_csv << "," << fmag << "," << fx << "," << fy << "," << fz;
        }
        node_force_csv << "\n";

        VectorXR x_cur, y_cur, z_cur, rx_cur, ry_cur, rz_cur;
        element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);
        element_data.RetrieveRotationToCPU(rx_cur, ry_cur, rz_cur);

        VectorXR edge_traction;
        element_data.RetrieveFacetTractionToCPU(edge_traction);

        std::vector<Real> stress_n_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));
        std::vector<Real> stress_m_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));
        std::vector<Real> stress_l_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));
        std::vector<Real> strain_n_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));
        std::vector<Real> strain_m_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));
        std::vector<Real> strain_l_vals(static_cast<size_t>(mesh.n_subfacets), Real(0));

        for (int sf = 0; sf < mesh.n_subfacets; ++sf) {
            const int ni = subfacet_node_i[static_cast<size_t>(sf)];
            const int nj = subfacet_node_j[static_cast<size_t>(sf)];
            if (ni < 0 || nj < 0)
                continue;
            const Real dx0 = mesh.particle_x(nj) - mesh.particle_x(ni);
            const Real dy0 = mesh.particle_y(nj) - mesh.particle_y(ni);
            const Real dz0 = mesh.particle_z(nj) - mesh.particle_z(ni);
            const Real dx = x_cur(nj) - x_cur(ni);
            const Real dy = y_cur(nj) - y_cur(ni);
            const Real dz = z_cur(nj) - z_cur(ni);

            const Real rix = rx_cur(ni), riy = ry_cur(ni), riz = rz_cur(ni);
            const Real rjx = rx_cur(nj), rjy = ry_cur(nj), rjz = rz_cur(nj);
            const Real rax = Real(0.5) * (rix + rjx);
            const Real ray = Real(0.5) * (riy + rjy);
            const Real raz = Real(0.5) * (riz + rjz);
            const Real rotx = ray * dz0 - raz * dy0;
            const Real roty = raz * dx0 - rax * dz0;
            const Real rotz = rax * dy0 - ray * dx0;

            const Real dux = dx - dx0 - rotx;
            const Real duy = dy - dy0 - roty;
            const Real duz = dz - dz0 - rotz;
            const Real l0 = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
            if (l0 <= Real(0))
                continue;
            const Real inv_l0 = Real(1) / l0;

            const Real n0 = mesh.subfacet_normal(sf * 3 + 0);
            const Real n1 = mesh.subfacet_normal(sf * 3 + 1);
            const Real n2 = mesh.subfacet_normal(sf * 3 + 2);
            const Real m0 = mesh.subfacet_tangent_q(sf * 3 + 0);
            const Real m1 = mesh.subfacet_tangent_q(sf * 3 + 1);
            const Real m2 = mesh.subfacet_tangent_q(sf * 3 + 2);
            const Real l0v = mesh.subfacet_tangent_s(sf * 3 + 0);
            const Real l1v = mesh.subfacet_tangent_s(sf * 3 + 1);
            const Real l2v = mesh.subfacet_tangent_s(sf * 3 + 2);

            const Real strain_n = (dux * n0 + duy * n1 + duz * n2) * inv_l0;
            const Real strain_m = (dux * m0 + duy * m1 + duz * m2) * inv_l0;
            const Real strain_l = (dux * l0v + duy * l1v + duz * l2v) * inv_l0;

            Real stress_n = Real(0), stress_m = Real(0), stress_l = Real(0);
            const int eidx = subfacet_edge_idx[static_cast<size_t>(sf)];
            if (eidx >= 0) {
                stress_n = edge_traction(eidx * 6 + 0);
                stress_m = edge_traction(eidx * 6 + 1);
                stress_l = edge_traction(eidx * 6 + 2);
            }
            stress_n_vals[static_cast<size_t>(sf)] = stress_n;
            stress_m_vals[static_cast<size_t>(sf)] = stress_m;
            stress_l_vals[static_cast<size_t>(sf)] = stress_l;
            strain_n_vals[static_cast<size_t>(sf)] = strain_n;
            strain_m_vals[static_cast<size_t>(sf)] = strain_m;
            strain_l_vals[static_cast<size_t>(sf)] = strain_l;
        }

        subfacet_csv << time_s;
        for (int sf = 0; sf < mesh.n_subfacets; ++sf) {
            const size_t idx = static_cast<size_t>(sf);
            subfacet_csv << "," << stress_n_vals[idx] << "," << stress_m_vals[idx] << "," << stress_l_vals[idx] << ","
                         << strain_n_vals[idx] << "," << strain_m_vals[idx] << "," << strain_l_vals[idx];
        }
        subfacet_csv << "\n";
    };

    write_csv_metrics(Real(0));

    int frame = 0;
    {
        if (!WriteLDPMTet4TetMeshToVTK(JoinPath(output_dir, TetVtkName(frame)), mesh, mesh.particle_x, mesh.particle_y,
                                       mesh.particle_z)) {
            std::cerr << "Failed writing initial tet VTK\n";
            return 1;
        }
        VectorXR crack0 = VectorXR::Zero(mesh.n_subfacets);
        if (!WriteLDPMTet4SubfacetMeshToVTK(JoinPath(output_dir, SubfacetVtkName(frame)), mesh, "crack_distance",
                                            crack0)) {
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
        const bool csv_output = (step_1 % csv_interval == 0);
        const bool key_output = std::binary_search(key_steps.begin(), key_steps.end(), step_1);
        const bool last_output = (step_1 == n_steps);

        if (csv_output || last_output) {
            write_csv_metrics(static_cast<Real>(step_1) * dt);
        }

        if (periodic_output || key_output || last_output) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            if (!WriteLDPMTet4TetMeshToVTK(JoinPath(output_dir, TetVtkName(frame)), mesh, x_cur, y_cur, z_cur)) {
                std::cerr << "Failed writing tet VTK at step " << step_1 << "\n";
                return 1;
            }

            VectorXR sf_crack;
            element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
            if (!WriteLDPMTet4SubfacetMeshToVTK(JoinPath(output_dir, SubfacetVtkName(frame)), mesh, "crack_distance",
                                                sf_crack)) {
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
    std::cout << "  Output directory: " << output_dir << "\n";
    std::cout << "  ldpm_singletet_case1_tet4_NNNNN.vtk\n";
    std::cout << "  ldpm_singletet_case1_subfacets_NNNNN.vtk\n";
    std::cout << "  ldpm_singletet_case1_node_forces.csv\n";
    std::cout << "  ldpm_singletet_case1_subfacet_stress_strain.csv\n";
    return 0;
}
