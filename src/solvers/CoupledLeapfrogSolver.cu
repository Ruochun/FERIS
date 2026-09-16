/* Project: FERIS
 * File: CoupledLeapfrogSolver.cu
 * Brief: Device projection between shared leapfrog kick and drift stages.
 */
#include "CoupledLeapfrogSolver.h"
#include "../dynamics/DynamicsSystemInternal.cuh"
#include <cmath>
#include <stdexcept>
namespace feris {
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
        if (!size)
            return;
        Check(cudaMalloc(reinterpret_cast<void**>(&ptr), size * sizeof(T)));
        Check(cudaMemset(ptr, 0, size * sizeof(T)));
    }
    explicit Buffer(const std::vector<T>& v) : Buffer(v.size()) { Upload(v.data()); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void Upload(const T* data) {
        if (!size)
            return;
        Check(cudaMemcpy(ptr, data, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void Download(T* data) const {
        if (!size)
            return;
        Check(cudaMemcpy(data, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
struct DeviceCsr {
    int rows;
    const int *offsets, *columns;
    const Real* values;
};
struct CsrBuffer {
    Buffer<int> offsets, columns;
    Buffer<Real> values;
    explicit CsrBuffer(const HostCsrMatrix& csr) : offsets(csr.offsets), columns(csr.columns), values(csr.values) {}
    DeviceCsr View() const { return {static_cast<int>(offsets.size) - 1, offsets.ptr, columns.ptr, values.ptr}; }
};
__device__ Real MultiplyRow(DeviceCsr a, int row, const Real* x) {
    Real sum = 0;
    for (int k = a.offsets[row]; k < a.offsets[row + 1]; ++k)
        sum += a.values[k] * x[a.columns[k]];
    return sum;
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

__global__ void ApplyCorrection(Real* v, const Real* impulse, const Real* inverse_mass, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] += inverse_mass[i] * impulse[i];
}
__global__ void
BodyKickDrift(Real* u, Real* v, const Real* force, const Real* inverse_mass, int offset, Real dt, bool kick) {
    int i = threadIdx.x;
    if (i < 6) {
        if (kick)
            v[offset + i] += dt * inverse_mass[offset + i] * force[offset + i];
        else
            u[offset + i] += dt * v[offset + i];
    }
}
}  // namespace
struct CoupledLeapfrogSolver::Impl {
    DynamicsSystem::Impl& system;
    Buffer<Real> u, v, inverse_mass, force, correction, rhs, impulse, drive;
    CsrBuffer constraints, inverse;
    Real dt = 0, speed = 0;
    bool started = false, independent_motion = false;
    explicit Impl(DynamicsSystem::Impl& s)
        : system(s),
          u(s.inverse_mass.size()),
          v(s.inverse_mass.size()),
          inverse_mass(s.inverse_mass.size()),
          force(s.inverse_mass.size()),
          correction(s.inverse_mass.size()),
          rhs(s.constraints.Matrix().Rows()),
          impulse(s.constraints.Matrix().Rows()),
          drive(s.constraints.TargetScales()),
          constraints(s.constraints.Matrix()),
          inverse(s.constraints.SmallBlockInverse()) {
        inverse_mass.Upload(s.inverse_mass.data());
        VectorXR loads = VectorXR::Zero(s.inverse_mass.size());
        for (const auto& b : s.bodies) {
            loads.segment<3>(b.offset) = b.properties.force;
            loads.segment<3>(b.offset + 3) = b.properties.moment;
        }
        force.Upload(loads.data());
    }
};
CoupledLeapfrogSolver::CoupledLeapfrogSolver(DynamicsSystem& system) {
    if (system.impl_->frozen)
        throw std::logic_error("Dynamics system already has a solver");
    system.impl_->constraints.Prepare(system.impl_->inverse_mass);
    system.impl_->frozen = true;
    impl_ = std::make_unique<Impl>(*system.impl_);
}
CoupledLeapfrogSolver::~CoupledLeapfrogSolver() = default;
void CoupledLeapfrogSolver::SetParameters(void* parameters) {
    if (impl_->started)
        throw std::logic_error("Cannot reset time step during integration");
    auto p = static_cast<CoupledLeapfrogParams*>(parameters);
    if (!p || !std::isfinite(p->time_step) || p->time_step <= 0)
        throw std::invalid_argument("Positive coupled time step required");
    impl_->dt = p->time_step;
    for (auto& g : impl_->system.groups) {
        LeapfrogParams q{p->time_step};
        g.solver->SetParameters(&q);
        g.solver->InitialHalfKick();
    }
    // Rest-start rigid velocities stagger to -dt/2 * acceleration.
    VectorXR velocity = VectorXR::Zero(impl_->system.inverse_mass.size());
    for (const auto& b : impl_->system.bodies) {
        velocity.segment<3>(b.offset) = -Real(0.5) * impl_->dt * b.properties.force / b.properties.mass;
        velocity.segment<3>(b.offset + 3) =
            -Real(0.5) * impl_->dt * b.properties.moment.cwiseQuotient(b.properties.inertia);
    }
    impl_->v.Upload(velocity.data());
}
void CoupledLeapfrogSolver::SetMotionIncrement(Real increment) {
    if (impl_->dt <= 0 || !std::isfinite(increment))
        throw std::invalid_argument("Set time step before motion");
    if (impl_->independent_motion) {
        impl_->drive.Upload(impl_->system.constraints.TargetScales().data());
        impl_->independent_motion = false;
    }
    impl_->speed = increment / impl_->dt;
}
void CoupledLeapfrogSolver::SetConstraintMotionIncrements(const VectorXR& increments) {
    if (impl_->dt <= 0 || increments.size() != static_cast<int>(impl_->drive.size) || !increments.allFinite())
        throw std::invalid_argument("Invalid constraint motion increments");
    VectorXR speed = increments / impl_->dt;
    impl_->drive.Upload(speed.data());
    impl_->speed = 1;
    impl_->independent_motion = true;
}
void CoupledLeapfrogSolver::Solve() {
    auto& p = *impl_;
    if (p.dt <= 0)
        throw std::logic_error("Set coupled parameters before Solve");
    p.started = true;
    int n = p.system.inverse_mass.size(), rows = p.system.constraints.Matrix().Rows();
    for (auto& g : p.system.groups) {
        g.solver->Kick();
        Check(cudaMemcpy(p.v.ptr + g.offset, g.solver->GetHalfStepVelocityDevicePtr(),
                         3 * g.element->get_n_coef() * sizeof(Real), cudaMemcpyDeviceToDevice));
    }
    for (const auto& b : p.system.bodies)
        BodyKickDrift<<<1, 6>>>(p.u.ptr, p.v.ptr, p.force.ptr, p.inverse_mass.ptr, b.offset, p.dt, true);
    if (rows) {
        ConstraintResidual<<<(rows + 255) / 256, 256>>>(p.constraints.View(), p.v.ptr, p.drive.ptr, p.speed, p.rhs.ptr);
        ConstraintImpulse<<<(rows + 255) / 256, 256>>>(p.inverse.View(), p.rhs.ptr, p.impulse.ptr);
        if (p.system.constraints.HasLargeBlocks()) {
            VectorXR rhs(rows), impulse(rows);
            p.rhs.Download(rhs.data());
            p.impulse.Download(impulse.data());
            p.system.constraints.SolveLargeBlocks(rhs, impulse);
            p.impulse.Upload(impulse.data());
        }
        Check(cudaMemset(p.correction.ptr, 0, n * sizeof(Real)));
        ScatterImpulse<<<(rows + 255) / 256, 256>>>(p.constraints.View(), p.impulse.ptr, p.correction.ptr);
        ApplyCorrection<<<(n + 255) / 256, 256>>>(p.v.ptr, p.correction.ptr, p.inverse_mass.ptr, n);
    }
    for (auto& g : p.system.groups) {
        Check(cudaMemcpy(g.solver->GetHalfStepVelocityDevicePtr(), p.v.ptr + g.offset,
                         3 * g.element->get_n_coef() * sizeof(Real), cudaMemcpyDeviceToDevice));
        g.solver->Drift();
    }
    for (const auto& b : p.system.bodies)
        BodyKickDrift<<<1, 6>>>(p.u.ptr, p.v.ptr, p.force.ptr, p.inverse_mass.ptr, b.offset, p.dt, false);
    Check(cudaGetLastError());
}
void CoupledLeapfrogSolver::RetrieveGeneralizedState(VectorXR& u, VectorXR& v, VectorXR& reactions) {
    auto& p = *impl_;
    u.resize(p.u.size);
    v.resize(p.v.size);
    reactions.resize(p.impulse.size);
    p.u.Download(u.data());
    p.v.Download(v.data());
    p.impulse.Download(reactions.data());
    if (p.dt > 0)
        reactions /= p.dt;
    for (const auto& g : p.system.groups) {
        VectorXR x, y, z;
        g.element->RetrievePositionToCPU(x, y, z);
        for (int i = 0; i < x.size(); ++i)
            u.segment<3>(g.offset + 3 * i) = Real3(x(i), y(i), z(i)) - g.reference[i];
    }
}
Real CoupledLeapfrogSolver::KineticEnergy() {
    auto& p = *impl_;
    VectorXR v(p.v.size);
    p.v.Download(v.data());
    Real energy = Real(0.5) * (v.array().square() / p.system.inverse_mass.array()).sum();
    for (const auto& g : p.system.groups)
        if (g.inertia.size()) {
            VectorXR omega(3 * g.inertia.size());
            Check(cudaMemcpy(omega.data(), g.solver->GetHalfStepVelocityDevicePtr() + 3 * g.inertia.size(),
                             omega.size() * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int i = 0; i < g.inertia.size(); ++i)
                energy += Real(0.5) * g.inertia(i) * omega.segment<3>(3 * i).squaredNorm();
        }
    return energy;
}
}  // namespace feris
