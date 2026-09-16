/* Project: FERIS
 * File: ldpm_tpb.cc
 * Brief: Leapfrog three-point bending of the Chrono notched LDPM core with
 *        elastic tetrahedral arms and mass-weighted bilateral constraints.
 *        See README.md for reference fidelity and numerical differences.
 */
#include "tpb_model.h"
#include "elements/LDPMTet4Data.cuh"
#include "elements/FEAT4Data.cuh"
#include "solvers/CoupledLeapfrogSolver.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace feris {
namespace tpb {
namespace {
struct Options {
    std::string mesh_dir = TPB_DEFAULT_MESH_DIR;
    long long steps = 0;
    Real end_time = 0.1, dt_scale = 1;
    bool vtk = true, elastic = false;
};
Options Parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-vtk") {
            options.vtk = false;
            continue;
        }
        if (arg == "--elastic") {
            options.elastic = true;
            continue;
        }
        if (arg == "--help") {
            std::cout << "ldpm_tpb [--mesh-dir DIR] [--end-time SECONDS] [--steps N]\n"
                      << "         [--dt-scale (0,1]] [--no-vtk] [--elastic]\n"
                      << "Default: reference 0.1 s nonlinear run. --steps caps steps for smoke tests.\n"
                      << "Outputs replace ./ldpm_tpb/ on each successful setup.\n";
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::runtime_error("Missing value for " + arg);
        std::string value = argv[++i];
        size_t consumed = 0;
        if (arg == "--mesh-dir") {
            options.mesh_dir = value;
            continue;
        }
        if (arg == "--steps") {
            options.steps = std::stoll(value, &consumed);
            if (options.steps <= 0)
                throw std::runtime_error("--steps must be positive");
        } else if (arg == "--end-time")
            options.end_time = std::stod(value, &consumed);
        else if (arg == "--dt-scale")
            options.dt_scale = std::stod(value, &consumed);
        else
            throw std::runtime_error("Unknown option: " + arg);
        if (consumed != value.size())
            throw std::runtime_error("Invalid value for " + arg);
    }
    if (!std::isfinite(options.end_time) || options.end_time <= 0 || options.end_time > 0.1)
        throw std::runtime_error("--end-time must be in (0,0.1] seconds");
    if (!std::isfinite(options.dt_scale) || options.dt_scale <= 0 || options.dt_scale > 1)
        throw std::runtime_error("--dt-scale must be in (0,1]");
    return options;
}
}  // namespace

int Run(int argc, char** argv) {
    const auto options = Parse(argc, argv);
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);
    LDPMTet4Mesh mesh;
    std::string error;
    if (!ReadLDPMTet4MeshFromFiles(options.mesh_dir + "/LDPMgeo000", mesh, &error))
        throw std::runtime_error(error);
    GPU_LDPMTet4_Data core;
    core.SetupFromMesh(mesh);
    struct CoreCleanup {
        GPU_LDPMTet4_Data& data;
        ~CoreCleanup() { data.Destroy(); }
    } cleanup{core};
    LDPMParams material{};
    material.rho = 2.338e-9;
    material.E0 = 60273;
    material.alpha = 0.25;
    material.sigma_t = 3.44;
    material.sigma_s = 8.944;
    material.n_t = 0.4;
    material.l_t = 500;
    material.E_d = 60273;
    material.sigma_c0 = 150;
    material.H_c0 = 24109;
    material.H_c1 = 6027.3;
    material.beta = 0;
    material.kc0 = 4;
    material.kc1 = 1;
    material.kc2 = 5;
    material.kc3 = 0.1;
    material.mu_0 = 0.4;
    material.mu_inf = 0;
    material.sigma_N0 = 600;
    // Chrono LDPM has facet force couples, but no independent curvature modulus.
    material.E_kT = material.E_kM = material.E_kL = 0;
    material.elastic_flag = options.elastic;
    core.SetLDPMParams(material);
    core.CalcMassMatrix();
    const auto arm_mesh = LoadArms(options.mesh_dir + "/TPBT_ElasticPart.inp");
    GPU_FEAT4_Data arms(arm_mesh.connectivity.rows(), arm_mesh.positions.size());
    arms.SetupFromMesh(arm_mesh.positions, arm_mesh.connectivity);
    struct ArmCleanup {
        GPU_FEAT4_Data& data;
        ~ArmCleanup() { data.Destroy(); }
    } arm_cleanup{arms};
    arms.SetLinearizedSVK(39000, 0.2);
    arms.SetDensity(material.rho);
    arms.SetDamping(0, 0);
    arms.CalcMassMatrix();
    DynamicsSystem system;
    system.AddElement(core);
    int arm_offset = system.AddElement(arms);
    Model model = BuildModel(mesh, arm_mesh, arms.AssembleTangentStiffnessCSR(), system, arm_offset);
    CoupledLeapfrogSolver solver(system);
    const Real dt_limit =
        std::min(Real(0.15) * model.min_edge / std::sqrt(material.E0 / material.rho), Real(0.5) * model.elastic_dt);
    const Real target_dt = options.dt_scale * dt_limit;
    if (!std::isfinite(target_dt) || target_dt <= 0 ||
        options.end_time / target_dt > Real(std::numeric_limits<long long>::max() / 2))
        throw std::runtime_error("Invalid or impractically small time step");
    const long long full_steps = static_cast<long long>(std::ceil(options.end_time / target_dt));
    const Real dt = options.end_time / full_steps;
    const long long steps = options.steps > 0 ? std::min(options.steps, full_steps) : full_steps;
    CoupledLeapfrogParams params{dt};
    solver.SetParameters(&params);
    const int n = model.inv_mass.size(), rows = system.Constraints().Matrix().Rows();
    VectorXR h_u = VectorXR::Zero(n), h_v(n), h_reaction(rows);
    const std::string output = "ldpm_tpb";
    std::filesystem::remove_all(output);
    std::filesystem::create_directories(output);
    std::ofstream csv(output + "/history.csv");
    if (!csv)
        throw std::runtime_error("Cannot create TPB history.csv");
    csv << std::setprecision(15)
        << "time_s,reaction_time_s,deflection_mm,cmod_mm,load_N,left_reaction_N,right_reaction_N,"
        << "left_slide_mm,right_slide_mm,kinetic_energy_Nmm,arm_strain_energy_Nmm,max_damage_proxy,max_crack_opening_"
           "mm,max_constraint_error_mm\n";
    std::ofstream run_info(output + "/run.txt");
    run_info << std::setprecision(16) << "dt_s=" << dt << "\nsteps=" << steps << "\nelastic=" << options.elastic
             << "\ncore_particles=" << mesh.n_particles << "\ncore_tets=" << mesh.n_tets
             << "\narm_nodes=" << model.arm_pos_ref.size() << "\narm_tets=" << model.arm_tets.size() << '\n';
    int frame = 0;
    auto WriteFrame = [&]() {
        if (!options.vtk)
            return;
        std::ostringstream tag;
        tag << std::setw(5) << std::setfill('0') << frame++;
        VectorXR x, y, z, crack;
        core.RetrievePositionToCPU(x, y, z);
        core.ProjectEdgeCrackDistanceToSubfacets(crack);
        if (!WriteLDPMTet4TetMeshToVTK(output + "/core_" + tag.str() + ".vtk", mesh, x, y, z) ||
            !WriteLDPMTet4SubfacetMeshToVTK(output + "/cracks_" + tag.str() + ".vtk", mesh, "crack_distance", crack))
            throw std::runtime_error("Failed writing TPB VTK");
        WriteArms(output + "/arms_" + tag.str() + ".vtk", model, h_u);
    };
    csv << "0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
    WriteFrame();
    const long long csv_interval = std::max(1LL, static_cast<long long>(std::llround(0.0001 / dt)));
    const long long vtk_interval = std::max(1LL, static_cast<long long>(std::llround(0.001 / dt)));
    std::cout << std::setprecision(10) << "LDPM TPB: " << steps << " steps, dt=" << dt
              << " s, final time=" << steps * dt << " s\n";
    for (long long step = 1; step <= steps; ++step) {
        const Real t = step * dt;
        solver.SetMotionIncrement(PlateDisplacement(t) - PlateDisplacement((step - 1) * dt));
        solver.Solve();
        const bool final = step == steps;
        if (step % csv_interval == 0 || step % vtk_interval == 0 || final) {
            solver.RetrieveGeneralizedState(h_u, h_v, h_reaction);
            if (!h_u.allFinite() || !h_v.allFinite() || !h_reaction.allFinite())
                throw std::runtime_error("Non-finite TPB state; reduce --dt-scale");
            Real deflection = 0, load = 0;
            for (int node : model.top_nodes)
                deflection -= h_u(3 * node + 2) / model.top_nodes.size();
            for (int row = 0; row < rows; ++row)
                load -= model.drive[row] * h_reaction(row);
            Real cmod = h_u(3 * model.cmod_nodes[1]) - h_u(3 * model.cmod_nodes[0]);
            Real constraint_error = system.Constraints().PositionError(h_u, PlateDisplacement(t));
            if (constraint_error > 1e-7)
                throw std::runtime_error("TPB constraint drift exceeds 1e-7 mm");
            VectorXR damage, crack;
            core.RetrieveFacetDamageToCPU(damage);
            core.ProjectEdgeCrackDistanceToSubfacets(crack);
            if (!damage.allFinite() || !crack.allFinite())
                throw std::runtime_error("Non-finite LDPM fracture state");
            Real kinetic = solver.KineticEnergy();
            Real strain_energy = model.stiffness.QuadraticEnergy(h_u.segment(model.core_dofs, model.stiffness.Rows()));
            csv << t << ',' << t - dt << ',' << deflection << ',' << cmod << ',' << load << ','
                << h_reaction(model.support_rows[0]) << ',' << h_reaction(model.support_rows[1]) << ','
                << h_u(model.plate_offset) << ',' << h_u(model.plate_offset + 6) << ',' << kinetic << ','
                << strain_energy << ',' << damage.maxCoeff() << ',' << crack.maxCoeff() << ',' << constraint_error
                << '\n';
            if (!csv)
                throw std::runtime_error("Failed writing TPB history");
            if (step % vtk_interval == 0 || final) {
                std::cout << "t=" << t << " s, deflection=" << deflection << " mm, CMOD=" << cmod
                          << " mm, load=" << load << " N, max crack=" << crack.maxCoeff() << " mm\n";
                WriteFrame();
            }
        }
    }
    std::cout << "Outputs: " << output << "/\n";
    // cleanup frees core buffers after solver teardown, including on exceptions.
    return 0;
}
}  // namespace tpb
}  // namespace feris
int main(int argc, char** argv) {
    try {
        return feris::tpb::Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ldpm_tpb: " << error.what() << '\n';
        return 1;
    }
}
