/* Project: FERIS
 * File: ldpm_tpb.cu
 * Brief: Leapfrog three-point bending of the Chrono notched LDPM core with
 *        elastic tetrahedral arms and mass-weighted bilateral constraints.
 *        See README_TPB.md for reference fidelity and numerical differences.
 */
#include "tpb_model.h"
#include "elements/LDPMTet4Data.cuh"
#include "solvers/LeapfrogSolver.cuh"
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
void Check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
template <class T>
struct Buffer {
    T* ptr = nullptr;
    size_t size = 0;
    explicit Buffer(size_t n) : size(n) {
        Check(cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T)));
        Check(cudaMemset(ptr, 0, size * sizeof(T)));
    }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) { Upload(v.data()); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void Upload(const T* data) { Check(cudaMemcpy(ptr, data, size * sizeof(T), cudaMemcpyHostToDevice)); }
    void Download(T* data) const { Check(cudaMemcpy(data, ptr, size * sizeof(T), cudaMemcpyDeviceToHost)); }
};
struct DeviceCsr {
    int rows;
    const int *offsets, *columns;
    const Real* values;
};
struct CsrBuffer {
    Buffer<int> offsets, columns;
    Buffer<Real> values;
    explicit CsrBuffer(const Csr& csr) : offsets(csr.offsets), columns(csr.columns), values(csr.values) {}
    DeviceCsr View() const { return {static_cast<int>(offsets.size) - 1, offsets.ptr, columns.ptr, values.ptr}; }
};
__device__ Real MultiplyRow(DeviceCsr a, int row, const Real* x) {
    Real sum = 0;
    for (int k = a.offsets[row]; k < a.offsets[row + 1]; ++k)
        sum += a.values[k] * x[a.columns[k]];
    return sum;
}
__global__ void ElasticKick(DeviceCsr k, const Real* u, Real* v, const Real* inv_mass, int offset, Real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k.rows)
        v[offset + i] -= dt * inv_mass[offset + i] * MultiplyRow(k, i, u + offset);
}
__global__ void Drift(Real* u, const Real* v, int begin, int end, Real dt) {
    int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < end)
        u[i] += dt * v[i];
}
__global__ void GatherCore(GPU_LDPMTet4_Data* core, const Real* core_v, Real* u, Real* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u[3 * i] = core->x_cur()(i) - core->x_ref()(i);
        u[3 * i + 1] = core->y_cur()(i) - core->y_ref()(i);
        u[3 * i + 2] = core->z_cur()(i) - core->z_ref()(i);
        for (int c = 0; c < 3; ++c)
            v[3 * i + c] = core_v[3 * i + c];
    }
}
__global__ void ConstraintResidual(DeviceCsr c, const Real* v, const Real* drive, Real speed, Real* rhs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c.rows)
        rhs[i] = drive[i] * speed - MultiplyRow(c, i, v);
}
__global__ void ConstraintImpulse(DeviceCsr inverse, const Real* rhs, Real* impulse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inverse.rows)
        impulse[i] = MultiplyRow(inverse, i, rhs);
}
__global__ void ScatterImpulse(DeviceCsr c, const Real* impulse, Real* correction) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c.rows)
        for (int k = c.offsets[i]; k < c.offsets[i + 1]; ++k)
            atomicAdd(correction + c.columns[k], c.values[k] * impulse[i]);
}
__global__ void Correct(Real* u,
                        Real* v,
                        const Real* correction,
                        const Real* inv_mass,
                        GPU_LDPMTet4_Data* core,
                        Real* core_v,
                        int n,
                        int core_dofs,
                        Real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    Real dv = inv_mass[i] * correction[i];
    v[i] += dv;
    u[i] += dt * dv;
    if (i < core_dofs) {
        core_v[i] = v[i];
        int node = i / 3;
        // Reconstruct from displacement to avoid accumulating two rounding errors.
        if (i % 3 == 0)
            core->x_cur()(node) = core->x_ref()(node) + u[i];
        if (i % 3 == 1)
            core->y_cur()(node) = core->y_ref()(node) + u[i];
        if (i % 3 == 2)
            core->z_cur()(node) = core->z_ref()(node) + u[i];
    }
}
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
Real ConstraintError(const Model& model, const VectorXR& u, Real displacement) {
    Real maximum = 0;
    for (int i = 0; i < model.constraints.Rows(); ++i) {
        Real error = -model.drive[i] * displacement;
        for (int k = model.constraints.offsets[i]; k < model.constraints.offsets[i + 1]; ++k)
            error += model.constraints.values[k] * u(model.constraints.columns[k]);
        maximum = std::max(maximum, std::abs(error));
    }
    return maximum;
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
    LeapfrogSolver solver(&core);
    // Setup needs a positive placeholder step; SetParameters below installs the
    // final common step and resets velocities before any integration occurs.
    LeapfrogParams params{1e-8};
    solver.SetParameters(&params);
    solver.Setup();
    VectorXR core_mass(2 * mesh.n_particles);
    Check(cudaMemcpy(core_mass.data(), solver.GetLumpedMassDevicePtr(), core_mass.size() * sizeof(Real),
                     cudaMemcpyDeviceToHost));
    Model model = BuildModel(mesh, options.mesh_dir + "/TPBT_ElasticPart.inp", core_mass);
    const Real dt_limit =
        std::min(Real(0.15) * model.min_edge / std::sqrt(material.E0 / material.rho), Real(0.5) * model.elastic_dt);
    const Real target_dt = options.dt_scale * dt_limit;
    if (!std::isfinite(target_dt) || target_dt <= 0 ||
        options.end_time / target_dt > Real(std::numeric_limits<long long>::max() / 2))
        throw std::runtime_error("Invalid or impractically small time step");
    const long long full_steps = static_cast<long long>(std::ceil(options.end_time / target_dt));
    const Real dt = options.end_time / full_steps;
    const long long steps = options.steps > 0 ? std::min(options.steps, full_steps) : full_steps;
    params.time_step = dt;
    solver.SetParameters(&params);
    // The prescribed quadratic history starts from rest and all initial forces
    // vanish, so the initial backward half-kick is zero.
    const int n = model.inv_mass.size(), rows = model.constraints.Rows();
    CsrBuffer stiffness(model.stiffness), constraints(model.constraints), inverse(model.gram_inverse);
    Buffer<Real> u(n), v(n), inv_mass(n), correction(n), rhs(rows), impulse(rows), drive(model.drive);
    inv_mass.Upload(model.inv_mass.data());
    VectorXR h_u = VectorXR::Zero(n), h_v(n), h_impulse(rows), core_vel(6 * mesh.n_particles);
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
        const Real speed = (PlateDisplacement(t) - PlateDisplacement((step - 1) * dt)) / dt;
        // Both subsystems first take the unconstrained kick and drift. Project
        // v_{n+1/2}, then correct the same drift by dt*delta_v. For these constant
        // linear constraints this equals projecting between kick and drift.
        solver.Solve();
        ElasticKick<<<(model.stiffness.Rows() + 255) / 256, 256>>>(stiffness.View(), u.ptr, v.ptr, inv_mass.ptr,
                                                                   model.core_dofs, dt);
        Drift<<<(n - model.core_dofs + 255) / 256, 256>>>(u.ptr, v.ptr, model.core_dofs, n, dt);
        GatherCore<<<(mesh.n_particles + 255) / 256, 256>>>(static_cast<GPU_LDPMTet4_Data*>(core.GetDevicePtr()),
                                                            solver.GetHalfStepVelocityDevicePtr(), u.ptr, v.ptr,
                                                            mesh.n_particles);
        ConstraintResidual<<<(rows + 255) / 256, 256>>>(constraints.View(), v.ptr, drive.ptr, speed, rhs.ptr);
        ConstraintImpulse<<<(rows + 255) / 256, 256>>>(inverse.View(), rhs.ptr, impulse.ptr);
        Check(cudaMemset(correction.ptr, 0, n * sizeof(Real)));
        ScatterImpulse<<<(rows + 255) / 256, 256>>>(constraints.View(), impulse.ptr, correction.ptr);
        Correct<<<(n + 255) / 256, 256>>>(u.ptr, v.ptr, correction.ptr, inv_mass.ptr,
                                          static_cast<GPU_LDPMTet4_Data*>(core.GetDevicePtr()),
                                          solver.GetHalfStepVelocityDevicePtr(), n, model.core_dofs, dt);
        Check(cudaGetLastError());
        const bool final = step == steps;
        if (step % csv_interval == 0 || step % vtk_interval == 0 || final) {
            u.Download(h_u.data());
            v.Download(h_v.data());
            impulse.Download(h_impulse.data());
            if (!h_u.allFinite() || !h_v.allFinite() || !h_impulse.allFinite())
                throw std::runtime_error("Non-finite TPB state; reduce --dt-scale");
            Real deflection = 0, load = 0;
            for (int node : model.top_nodes)
                deflection -= h_u(3 * node + 2) / model.top_nodes.size();
            for (int row = 0; row < rows; ++row)
                load -= model.drive[row] * h_impulse(row) / dt;
            Real cmod = h_u(3 * model.cmod_nodes[1]) - h_u(3 * model.cmod_nodes[0]);
            Real constraint_error = ConstraintError(model, h_u, PlateDisplacement(t));
            if (constraint_error > 1e-7)
                throw std::runtime_error("TPB constraint drift exceeds 1e-7 mm");
            VectorXR damage, crack;
            core.RetrieveFacetDamageToCPU(damage);
            core.ProjectEdgeCrackDistanceToSubfacets(crack);
            if (!damage.allFinite() || !crack.allFinite())
                throw std::runtime_error("Non-finite LDPM fracture state");
            Real kinetic = Real(0.5) * (h_v.array().square() / model.inv_mass.array()).sum();
            Check(cudaMemcpy(core_vel.data(), solver.GetHalfStepVelocityDevicePtr(), core_vel.size() * sizeof(Real),
                             cudaMemcpyDeviceToHost));
            for (int node = 0; node < mesh.n_particles; ++node)
                kinetic += Real(0.5) * core_mass(mesh.n_particles + node) *
                           core_vel.segment<3>(model.core_dofs + 3 * node).squaredNorm();
            Real strain_energy = 0;
            for (int i = 0; i < model.stiffness.Rows(); ++i)
                for (int k = model.stiffness.offsets[i]; k < model.stiffness.offsets[i + 1]; ++k)
                    strain_energy += Real(0.5) * h_u(model.core_dofs + i) * model.stiffness.values[k] *
                                     h_u(model.core_dofs + model.stiffness.columns[k]);
            csv << t << ',' << t - dt << ',' << deflection << ',' << cmod << ',' << load << ','
                << h_impulse(model.support_rows[0]) / dt << ',' << h_impulse(model.support_rows[1]) / dt << ','
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
    Check(cudaDeviceSynchronize());
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
