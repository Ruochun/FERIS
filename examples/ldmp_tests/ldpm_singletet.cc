/**
 * ldpm_singletet.cc
 *
 * LDPM single regular tetrahedron benchmark (Cases 1–7 from the uploaded
 * document) using LeapfrogSolver with prescribed translational/rotational
 * velocity BC selected from a command-line case id.
 *
 * Mesh:
 *   data/meshes/LDPMTet4/SingleTet/LDPM_debugRegTet-*.dat
 *
 * Required output series:
 *   ldpm_singletet_case{N}_tet4_NNNNN.vtk
 *   ldpm_singletet_case{N}_subfacets_NNNNN.vtk
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
#include <stdexcept>
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

// Table-4 parameters for full LDPM model (mm-tonne-s unit system).
static constexpr Real RHO_VAL = Real(2.40e-9);     // tonne/mm^3
static constexpr Real E0_VAL = Real(30000.0);      // MPa (normal modulus)
static constexpr Real ALPHA_VAL = Real(0.2);       // E_T / E_N
static constexpr Real SIGMA_T_VAL = Real(3.0);     // MPa (tensile strength)
static constexpr Real LT_VAL = Real(250.0);        // mm (tensile characteristic length)
static constexpr Real GT_VAL = Real(0.0375);       // N/mm (fracture energy = lt*sigma_t^2/(2*E0))
static constexpr Real NT_VAL = Real(0.2);          // softening exponent
static constexpr Real SIGMA_C0_VAL = Real(120.0);  // MPa (compressive yielding strength)
static constexpr Real HC0_OVER_E0 = Real(0.33);    // H_c0/E_0 = initial hardening modulus ratio
static constexpr Real KC0_VAL = Real(3.0);         // transitional strain ratio
static constexpr Real KC1_VAL = Real(0.5);         // deviatoric strain threshold ratio
static constexpr Real KC2_VAL = Real(5.0);         // deviatoric damage parameter
static constexpr Real MU0_VAL = Real(0.4);         // initial friction
static constexpr Real MU_INF_VAL = Real(0.0);      // asymptotic friction
static constexpr Real SIGMA_N0_VAL = Real(600.0);  // MPa (transitional stress)
static constexpr Real ED_OVER_E0 = Real(1.0);      // densification modulus ratio (E_d/E_0)
static constexpr Real BETA_VAL = Real(0.0);        // volumetric deviatoric coupling
static constexpr Real KT_VAL = Real(0.0);          // tensile unloading parameter
static constexpr Real KS_VAL = Real(0.0);          // shear unloading
static constexpr Real KC_VAL = Real(0.0);          // compressive unloading
static constexpr Real RS_VAL = Real(0.0);          // shear softening modulus ratio
static constexpr Real HC1_OVER_E0 = Real(0.1);     // H_c1/E_0 = final hardening modulus ratio
static constexpr Real KC3_VAL = Real(0.1);         // volumetric strain parameter (from Table)
static constexpr Real BETA_K = Real(0.25);         // rotational coupling

static constexpr std::array<Real, 5> kDispTimePts = {Real(0.0), Real(0.2), Real(2.4), Real(4.6), Real(4.8)};
static constexpr std::array<Real, 5> kRotTimePts = {Real(0.0), Real(0.5), Real(1.5), Real(2.5), Real(3.0)};

static constexpr std::array<Real, 5> kDisp1 = {Real(0.0), Real(0.02), Real(-0.2), Real(0.02), Real(0.0)};
static constexpr std::array<Real, 5> kDisp2 = {Real(0.0), Real(0.01632993162), Real(-0.1632993162), Real(0.01632993162),
                                               Real(0.0)};
static constexpr std::array<Real, 5> kDisp3 = {Real(0.0), Real(-0.00942809042), Real(0.0942809042),
                                               Real(-0.00942809042), Real(0.0)};
static constexpr std::array<Real, 5> kDisp4 = {Real(0.0), Real(-0.006666666667), Real(0.06666666667),
                                               Real(-0.006666666667), Real(0.0)};
static constexpr std::array<Real, 5> kDisp5 = {Real(0.0), Real(0.01885618083), Real(-0.1885618083), Real(0.01885618083),
                                               Real(0.0)};
static constexpr std::array<Real, 5> kRot1 = {Real(0.0), Real(0.2), Real(-0.2), Real(0.2), Real(0.0)};

static constexpr Real T_VTK = Real(0.1);
static constexpr Real T_CSV = Real(1.0e-3);

enum class HistoryType { DISP_1, DISP_2, DISP_3, DISP_4, DISP_5, ROT_1 };

struct DrivenBC {
    int node = 0;
    bool rotational = false;
    int component = 0;
    HistoryType history = HistoryType::DISP_1;
    Real scale = Real(1);
};

struct CaseDefinition {
    int id = 1;
    std::string description;
    std::vector<DrivenBC> bc;
    const std::array<Real, 5>* key_times = &kDispTimePts;
};

const std::array<Real, 5>& HistoryValues(HistoryType h) {
    switch (h) {
        case HistoryType::DISP_1:
            return kDisp1;
        case HistoryType::DISP_2:
            return kDisp2;
        case HistoryType::DISP_3:
            return kDisp3;
        case HistoryType::DISP_4:
            return kDisp4;
        case HistoryType::DISP_5:
            return kDisp5;
        case HistoryType::ROT_1:
            return kRot1;
    }
    return kDisp1;
}

const std::array<Real, 5>& HistoryTimes(HistoryType h) {
    return (h == HistoryType::ROT_1) ? kRotTimePts : kDispTimePts;
}

Real HistoryRate(HistoryType h, Real t) {
    const auto& times = HistoryTimes(h);
    const auto& values = HistoryValues(h);
    if (t >= times.back())
        return Real(0);

    for (size_t i = 0; i + 1 < times.size(); ++i) {
        if (t >= times[i] && t < times[i + 1]) {
            const Real dt = times[i + 1] - times[i];
            return (values[i + 1] - values[i]) / dt;
        }
    }
    return Real(0);
}

CaseDefinition BuildCaseDefinition(int case_id) {
    CaseDefinition c;
    c.id = case_id;
    switch (case_id) {
        case 1:
            c.description = "Node 2: UX = Disp1";
            c.bc.push_back({1, false, 0, HistoryType::DISP_1, Real(1)});
            c.key_times = &kDispTimePts;
            break;
        case 2:
            c.description = "Node 3: UY = Disp1";
            c.bc.push_back({2, false, 1, HistoryType::DISP_1, Real(1)});
            c.key_times = &kDispTimePts;
            break;
        case 3:
            c.description = "Node 4: UZ = Disp1";
            c.bc.push_back({3, false, 2, HistoryType::DISP_1, Real(1)});
            c.key_times = &kDispTimePts;
            break;
        case 4:
            c.description = "Volumetric loading (Disp1..Disp5)";
            c.bc.push_back({0, false, 0, HistoryType::DISP_2, Real(-1)});
            c.bc.push_back({0, false, 1, HistoryType::DISP_3, Real(1)});
            c.bc.push_back({0, false, 2, HistoryType::DISP_4, Real(1)});
            c.bc.push_back({1, false, 0, HistoryType::DISP_2, Real(1)});
            c.bc.push_back({1, false, 1, HistoryType::DISP_3, Real(1)});
            c.bc.push_back({1, false, 2, HistoryType::DISP_4, Real(1)});
            c.bc.push_back({2, false, 1, HistoryType::DISP_5, Real(1)});
            c.bc.push_back({2, false, 2, HistoryType::DISP_4, Real(1)});
            c.bc.push_back({3, false, 2, HistoryType::DISP_1, Real(1)});
            c.key_times = &kDispTimePts;
            break;
        case 5:
            c.description = "Node 2: RX = Rot1";
            c.bc.push_back({1, true, 0, HistoryType::ROT_1, Real(1)});
            c.key_times = &kRotTimePts;
            break;
        case 6:
            c.description = "Node 3: RY = Rot1";
            c.bc.push_back({2, true, 1, HistoryType::ROT_1, Real(1)});
            c.key_times = &kRotTimePts;
            break;
        case 7:
            c.description = "Node 3: RZ = Rot1";
            c.bc.push_back({2, true, 2, HistoryType::ROT_1, Real(1)});
            c.key_times = &kRotTimePts;
            break;
        default:
            break;
    }
    return c;
}

std::string TetVtkName(int case_id, int frame) {
    std::ostringstream s;
    s << "ldpm_singletet_case" << case_id << "_tet4_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
    return s.str();
}

std::string SubfacetVtkName(int case_id, int frame) {
    std::ostringstream s;
    s << "ldpm_singletet_case" << case_id << "_subfacets_" << std::setfill('0') << std::setw(5) << frame << ".vtk";
    return s.str();
}

const int kLocalEdges[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

std::string JoinPath(const std::string& dir, const std::string& file) {
    return (std::filesystem::path(dir) / file).string();
}

// Forward declaration — implementation follows main().
void write_csv_metrics(Real time_s,
                       GPU_LDPMTet4_Data& element_data,
                       const LDPMTet4Mesh& mesh,
                       const std::vector<int>& subfacet_edge_idx,
                       const std::vector<int>& subfacet_node_i,
                       const std::vector<int>& subfacet_node_j,
                       std::ofstream& node_force_csv,
                       std::ofstream& node_rm_csv,
                       std::ofstream& node_u_csv,
                       std::ofstream& subfacet_csv);

}  // namespace

int main(int argc, char* argv[]) {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);
    int case_id = 1;
    if (argc > 2) {
        std::cerr << "Usage: ldpm_singletet [case_id]\n";
        std::cerr << "  case_id must be in [1, 7]. Default is 1.\n";
        return 1;
    }
    if (argc == 2) {
        try {
            size_t parsed = 0;
            case_id = std::stoi(argv[1], &parsed);
            if (parsed != std::string(argv[1]).size())
                throw std::invalid_argument("Trailing characters");
        } catch (const std::exception&) {
            std::cerr << "Invalid case_id '" << argv[1] << "'. Valid values are integers in [1, 7].\n";
            return 1;
        }
    }
    if (case_id < 1 || case_id > 7) {
        std::cerr << "Invalid case_id " << case_id << ". Valid values are integers in [1, 7].\n";
        return 1;
    }

    const CaseDefinition case_def = BuildCaseDefinition(case_id);
    const std::string case_tag = "case" + std::to_string(case_id);
    const std::string output_dir = "ldpm_singletet_" + case_tag;
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

    const Real E_kT_VAL = BETA_K * E0_VAL * l_min * l_min;
    const Real E_kM_VAL = E_kT_VAL;
    const Real E_kL_VAL = E_kT_VAL;

    // Compute derived parameters for the full LDPM model
    const Real sigma_s_val = Real(2.5) * SIGMA_T_VAL;  // r_st = sigma_s / sigma_t (shear strength ratio)

    GPU_LDPMTet4_Data element_data;
    element_data.SetupFromMesh(mesh);

    // Set up full LDPM parameters
    LDPMParams ldpm_params{};
    ldpm_params.E0 = E0_VAL;
    ldpm_params.alpha = ALPHA_VAL;
    ldpm_params.sigma_t = SIGMA_T_VAL;
    ldpm_params.sigma_s = sigma_s_val;
    ldpm_params.n_t = NT_VAL;
    ldpm_params.l_t = LT_VAL;
    ldpm_params.r_s = RS_VAL;
    ldpm_params.k_t = KT_VAL;
    ldpm_params.E_d = ED_OVER_E0 * E0_VAL;
    ldpm_params.sigma_c0 = SIGMA_C0_VAL;
    ldpm_params.H_c0 = HC0_OVER_E0 * E0_VAL;
    ldpm_params.H_c1 = HC1_OVER_E0 * E0_VAL;
    ldpm_params.kc0 = KC0_VAL;
    ldpm_params.kc1 = KC1_VAL;
    ldpm_params.kc2 = KC2_VAL;
    ldpm_params.kc3 = KC3_VAL;
    ldpm_params.beta = BETA_VAL;
    ldpm_params.mu_0 = MU0_VAL;
    ldpm_params.mu_inf = MU_INF_VAL;
    ldpm_params.sigma_N0 = SIGMA_N0_VAL;
    ldpm_params.E_kT = E_kT_VAL;
    ldpm_params.E_kM = E_kM_VAL;
    ldpm_params.E_kL = E_kL_VAL;
    ldpm_params.rho = RHO_VAL;
    ldpm_params.elastic_flag = false;

    element_data.SetLDPMParams(ldpm_params);

    VectorReal3 h_f_ext(static_cast<size_t>(mesh.n_particles), Real3::Zero());
    element_data.SetExternalForce(h_f_ext);

    // Constrain all nodes by default and re-impose the selected prescribed
    // translational/rotational velocity components for the active case.
    VectorXi h_fixed(4);
    h_fixed << 0, 1, 2, 3;
    element_data.SetNodalFixed(h_fixed);

    element_data.CalcMassMatrix();

    const Real c_N = std::sqrt(E0_VAL / RHO_VAL);
    const Real dt = Real(0.5) * (l_min / c_N);
    const Real T_SIM = case_def.key_times->back();
    const int n_steps = static_cast<int>(std::ceil(T_SIM / dt));
    const int vtk_interval = std::max(1, static_cast<int>(std::round(T_VTK / dt)));
    const int csv_interval = std::max(1, static_cast<int>(std::round(T_CSV / dt)));

    // Ensure outputs at key tabulated times in addition to regular cadence.
    std::vector<int> key_steps;
    key_steps.reserve(case_def.key_times->size());
    for (Real t_key : *case_def.key_times) {
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

    std::vector<int> driven_trans_nodes;
    std::vector<int> driven_rot_nodes;
    for (const auto& bc : case_def.bc) {
        auto& nodes = bc.rotational ? driven_rot_nodes : driven_trans_nodes;
        if (std::find(nodes.begin(), nodes.end(), bc.node) == nodes.end())
            nodes.push_back(bc.node);
    }
    std::sort(driven_trans_nodes.begin(), driven_trans_nodes.end());
    std::sort(driven_rot_nodes.begin(), driven_rot_nodes.end());

    VectorXi h_driven_trans(static_cast<int>(driven_trans_nodes.size()));
    for (int i = 0; i < static_cast<int>(driven_trans_nodes.size()); ++i)
        h_driven_trans(i) = driven_trans_nodes[static_cast<size_t>(i)];
    VectorXi h_driven_rot(static_cast<int>(driven_rot_nodes.size()));
    for (int i = 0; i < static_cast<int>(driven_rot_nodes.size()); ++i)
        h_driven_rot(i) = driven_rot_nodes[static_cast<size_t>(i)];

    VectorReal3 h_driven_vel(static_cast<size_t>(h_driven_trans.size()), Real3::Zero());
    VectorReal3 h_driven_rot_vel(static_cast<size_t>(h_driven_rot.size()), Real3::Zero());

    auto fill_driven_velocities = [&](Real t_now) {
        std::fill(h_driven_vel.begin(), h_driven_vel.end(), Real3::Zero());
        std::fill(h_driven_rot_vel.begin(), h_driven_rot_vel.end(), Real3::Zero());
        for (const auto& bc : case_def.bc) {
            const Real rate = bc.scale * HistoryRate(bc.history, t_now);
            if (bc.rotational) {
                for (int i = 0; i < h_driven_rot.size(); ++i) {
                    if (h_driven_rot(i) == bc.node) {
                        h_driven_rot_vel[static_cast<size_t>(i)](bc.component) = rate;
                        break;
                    }
                }
            } else {
                for (int i = 0; i < h_driven_trans.size(); ++i) {
                    if (h_driven_trans(i) == bc.node) {
                        h_driven_vel[static_cast<size_t>(i)](bc.component) = rate;
                        break;
                    }
                }
            }
        }
    };

    fill_driven_velocities(Real(0));
    if (h_driven_trans.size() > 0) {
        solver.SetPrescribedVelocityBC(h_driven_trans, h_driven_vel);
        solver.SetNodalVelocity(h_driven_trans, h_driven_vel);
    }
    if (h_driven_rot.size() > 0) {
        solver.SetPrescribedAngularVelocityBC(h_driven_rot, h_driven_rot_vel);
        solver.SetNodalAngularVelocity(h_driven_rot, h_driven_rot_vel);
    }
    VectorReal3 last_driven_vel = h_driven_vel;
    VectorReal3 last_driven_rot_vel = h_driven_rot_vel;
    auto velocities_changed = [](const VectorReal3& a, const VectorReal3& b) {
        if (a.size() != b.size())
            return true;
        for (size_t i = 0; i < a.size(); ++i) {
            if ((a[i] - b[i]).norm() > Real(1e-15))
                return true;
        }
        return false;
    };
    solver.InitialHalfKick();

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Single-tet Case " << case_id << ": " << case_def.description << "\n";
    std::cout << "  dt=" << dt << " s, steps=" << n_steps << "\n";
    std::cout << "  E0=" << E0_VAL << ", alpha=" << ALPHA_VAL << ", sigma_t=" << SIGMA_T_VAL
              << ", sigma_c0=" << SIGMA_C0_VAL << ", l_min=" << l_min << "\n";
    std::cout << "  Output directory: " << output_dir << "\n";

    const std::string node_force_csv_name =
        JoinPath(output_dir, "ldpm_singletet_case" + std::to_string(case_id) + "_node_forces.csv");
    const std::string node_rm_csv_name =
        JoinPath(output_dir, "ldpm_singletet_case" + std::to_string(case_id) + "_node_rotational_moments.csv");
    const std::string node_u_csv_name =
        JoinPath(output_dir, "ldpm_singletet_case" + std::to_string(case_id) + "_node_displacements.csv");
    const std::string subfacet_csv_name =
        JoinPath(output_dir, "ldpm_singletet_case" + std::to_string(case_id) + "_subfacet_stress_strain.csv");
    std::ofstream node_force_csv(node_force_csv_name);
    std::ofstream node_rm_csv(node_rm_csv_name);
    std::ofstream node_u_csv(node_u_csv_name);
    std::ofstream subfacet_csv(subfacet_csv_name);
    if (!node_force_csv.is_open() || !node_rm_csv.is_open() || !node_u_csv.is_open() || !subfacet_csv.is_open()) {
        std::cerr << "Failed to open CSV outputs in " << output_dir << "\n";
        return 1;
    }
    node_force_csv << "# time_s";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << ",node_" << (node + 1) << "_f_mag";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << ",node_" << (node + 1) << "_f_x";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << ",node_" << (node + 1) << "_f_y";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << ",node_" << (node + 1) << "_f_z";
    node_force_csv << "\n";

    node_rm_csv << "# time_s";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << ",Rm_mag_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << ",Rm_x_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << ",Rm_y_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << ",Rm_z_node_" << (node + 1);
    node_rm_csv << "\n";

    node_u_csv << "# time_s";
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << ",U_mag_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << ",U_x_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << ",U_y_node_" << (node + 1);
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << ",U_z_node_" << (node + 1);
    node_u_csv << "\n";

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

    auto write_csv = [&](Real time_s) {
        write_csv_metrics(time_s, element_data, mesh, subfacet_edge_idx, subfacet_node_i, subfacet_node_j,
                          node_force_csv, node_rm_csv, node_u_csv, subfacet_csv);
    };

    write_csv(Real(0));

    int frame = 0;
    {
        if (!WriteLDPMTet4TetMeshToVTK(JoinPath(output_dir, TetVtkName(case_id, frame)), mesh, mesh.particle_x,
                                       mesh.particle_y, mesh.particle_z)) {
            std::cerr << "Failed writing initial tet VTK\n";
            return 1;
        }
        VectorXR crack0 = VectorXR::Zero(mesh.n_subfacets);
        if (!WriteLDPMTet4SubfacetMeshToVTK(JoinPath(output_dir, SubfacetVtkName(case_id, frame)), mesh,
                                            "crack_distance", crack0)) {
            std::cerr << "Failed writing initial subfacet VTK\n";
            return 1;
        }
        ++frame;
    }

    for (int step = 0; step < n_steps; ++step) {
        const Real t_n = static_cast<Real>(step) * dt;
        fill_driven_velocities(t_n);
        if (h_driven_trans.size() > 0 && velocities_changed(h_driven_vel, last_driven_vel)) {
            solver.SetPrescribedVelocityBC(h_driven_trans, h_driven_vel);
            last_driven_vel = h_driven_vel;
        }
        if (h_driven_rot.size() > 0 && velocities_changed(h_driven_rot_vel, last_driven_rot_vel)) {
            solver.SetPrescribedAngularVelocityBC(h_driven_rot, h_driven_rot_vel);
            last_driven_rot_vel = h_driven_rot_vel;
        }

        solver.Solve();

        const int step_1 = step + 1;
        const bool periodic_output = (step_1 % vtk_interval == 0);
        const bool csv_output = (step_1 % csv_interval == 0);
        const bool key_output = std::binary_search(key_steps.begin(), key_steps.end(), step_1);
        const bool last_output = (step_1 == n_steps);

        if (csv_output || last_output) {
            write_csv(static_cast<Real>(step_1) * dt);
        }

        if (periodic_output || key_output || last_output) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            if (!WriteLDPMTet4TetMeshToVTK(JoinPath(output_dir, TetVtkName(case_id, frame)), mesh, x_cur, y_cur,
                                           z_cur)) {
                std::cerr << "Failed writing tet VTK at step " << step_1 << "\n";
                return 1;
            }

            VectorXR sf_crack;
            element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
            if (!WriteLDPMTet4SubfacetMeshToVTK(JoinPath(output_dir, SubfacetVtkName(case_id, frame)), mesh,
                                                "crack_distance", sf_crack)) {
                std::cerr << "Failed writing subfacet VTK at step " << step_1 << "\n";
                return 1;
            }

            const Real t_out = static_cast<Real>(step_1) * dt;
            Real monitor_target = Real(0);
            if (!case_def.bc.empty())
                monitor_target = case_def.bc.front().scale * HistoryRate(case_def.bc.front().history, t_n);
            Real monitor_state = Real(0);
            if (!case_def.bc.empty()) {
                const auto& bc0 = case_def.bc.front();
                if (bc0.rotational) {
                    VectorXR rx_cur, ry_cur, rz_cur;
                    element_data.RetrieveRotationToCPU(rx_cur, ry_cur, rz_cur);
                    const VectorXR* rot_comp =
                        (bc0.component == 0) ? &rx_cur : ((bc0.component == 1) ? &ry_cur : &rz_cur);
                    monitor_state = (*rot_comp)(bc0.node);
                } else {
                    const VectorXR* pos_comp = (bc0.component == 0) ? &x_cur : ((bc0.component == 1) ? &y_cur : &z_cur);
                    const VectorXR* pos_ref = (bc0.component == 0)
                                                  ? &mesh.particle_x
                                                  : ((bc0.component == 1) ? &mesh.particle_y : &mesh.particle_z);
                    monitor_state = (*pos_comp)(bc0.node) - (*pos_ref)(bc0.node);
                }
            }
            std::cout << "frame=" << std::setw(4) << frame << " step=" << std::setw(8) << step_1 << " t=" << t_out
                      << " monitor=" << monitor_state << " target_rate=" << monitor_target << "\n";
            ++frame;
        }
    }

    element_data.Destroy();

    std::cout << "Done. Wrote " << frame << " frame(s).\n";
    std::cout << "  Output directory: " << output_dir << "\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_tet4_NNNNN.vtk\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_subfacets_NNNNN.vtk\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_node_forces.csv\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_node_rotational_moments.csv\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_node_displacements.csv\n";
    std::cout << "  ldpm_singletet_case" << case_id << "_subfacet_stress_strain.csv\n";
    return 0;
}

namespace {

void write_csv_metrics(Real time_s,
                       GPU_LDPMTet4_Data& element_data,
                       const LDPMTet4Mesh& mesh,
                       const std::vector<int>& subfacet_edge_idx,
                       const std::vector<int>& subfacet_node_i,
                       const std::vector<int>& subfacet_node_j,
                       std::ofstream& node_force_csv,
                       std::ofstream& node_rm_csv,
                       std::ofstream& node_u_csv,
                       std::ofstream& subfacet_csv) {
    VectorReal3 f_int;
    element_data.RetrieveInternalForceToCPU(f_int);
    std::vector<Real> fmag_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> fx_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> fy_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> fz_vals(static_cast<size_t>(mesh.n_particles));
    for (int node = 0; node < mesh.n_particles; ++node) {
        const size_t idx = static_cast<size_t>(node);
        const Real fx = f_int[idx](0);
        const Real fy = f_int[idx](1);
        const Real fz = f_int[idx](2);
        fmag_vals[idx] = std::sqrt(fx * fx + fy * fy + fz * fz);
        fx_vals[idx] = fx;
        fy_vals[idx] = fy;
        fz_vals[idx] = fz;
    }
    node_force_csv << time_s;
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << "," << fmag_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << "," << fx_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << "," << fy_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_force_csv << "," << fz_vals[static_cast<size_t>(node)];
    node_force_csv << "\n";

    VectorReal3 rm_int;
    element_data.RetrieveInternalMomentToCPU(rm_int);
    std::vector<Real> rmmag_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> rmx_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> rmy_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> rmz_vals(static_cast<size_t>(mesh.n_particles));
    for (int node = 0; node < mesh.n_particles; ++node) {
        const size_t idx = static_cast<size_t>(node);
        const Real rmx = rm_int[idx](0);
        const Real rmy = rm_int[idx](1);
        const Real rmz = rm_int[idx](2);
        rmmag_vals[idx] = std::sqrt(rmx * rmx + rmy * rmy + rmz * rmz);
        rmx_vals[idx] = rmx;
        rmy_vals[idx] = rmy;
        rmz_vals[idx] = rmz;
    }
    node_rm_csv << time_s;
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << "," << rmmag_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << "," << rmx_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << "," << rmy_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_rm_csv << "," << rmz_vals[static_cast<size_t>(node)];
    node_rm_csv << "\n";

    VectorXR x_cur, y_cur, z_cur, rx_cur, ry_cur, rz_cur;
    element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);
    element_data.RetrieveRotationToCPU(rx_cur, ry_cur, rz_cur);

    std::vector<Real> umag_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> ux_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> uy_vals(static_cast<size_t>(mesh.n_particles));
    std::vector<Real> uz_vals(static_cast<size_t>(mesh.n_particles));
    for (int node = 0; node < mesh.n_particles; ++node) {
        const size_t idx = static_cast<size_t>(node);
        const Real ux = x_cur(node) - mesh.particle_x(node);
        const Real uy = y_cur(node) - mesh.particle_y(node);
        const Real uz = z_cur(node) - mesh.particle_z(node);
        umag_vals[idx] = std::sqrt(ux * ux + uy * uy + uz * uz);
        ux_vals[idx] = ux;
        uy_vals[idx] = uy;
        uz_vals[idx] = uz;
    }
    node_u_csv << time_s;
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << "," << umag_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << "," << ux_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << "," << uy_vals[static_cast<size_t>(node)];
    for (int node = 0; node < mesh.n_particles; ++node)
        node_u_csv << "," << uz_vals[static_cast<size_t>(node)];
    node_u_csv << "\n";

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
}

}  // namespace
