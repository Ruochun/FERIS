/*==============================================================
 *==============================================================
 * Project: TLFEA
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 * File:    LeapfrogSolver.cuh
 * Brief:   Declares the LeapfrogSolver, a second-order explicit
 *          central-difference (leapfrog) time integrator for solid
 *          mechanics simulations. Owns GPU buffers for half-step nodal
 *          velocities and a lumped (row-summed) mass vector. Supports
 *          ANCF3243, ANCF3443, FEAT10, FEAT4, and LDPM_TET4 element
 *          types, with primary focus on LDPM_TET4 for
 *          particle-scale simulations with 6 DOFs per particle.
 *
 *          Supported element types: TYPE_3243, TYPE_3443, TYPE_T10,
 *          TYPE_T4, TYPE_LDPM_TET4.
 *          See docs/SOLVER_ELEMENT_COMPATIBILITY.md for the full matrix.
 *==============================================================
 *==============================================================*/

#pragma once

#include <MoPhiEssentials.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/LDPMTet4Data.cuh"
#include "../types.h"
#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "SolverBase.h"

namespace tlfea {

// Parameters for the leapfrog (central-difference) explicit integrator.
struct LeapfrogParams {
    // Physical simulation time step (seconds).
    Real time_step;
};

// LeapfrogSolver implements the explicit central-difference (leapfrog) time
// integration scheme for nonlinear solid mechanics:
//
//   Given x_n (full-step positions) and v_{n-1/2} (half-step velocities):
//     1. Compute P = P(x_n, v_{n-1/2})  [stress, including viscous damping]
//     2. Assemble f_int(x_n) from P
//     3. a_n = M_lump^{-1} * (f_ext - f_int)
//     4. v_{n+1/2} = v_{n-1/2} + dt * a_n
//     5. Enforce BCs: zero velocity at fixed nodes
//     6. x_{n+1} = x_n + dt * v_{n+1/2}
//
// The lumped (row-sum) mass matrix is computed once in Setup() from the
// consistent CSR mass matrix that the element has already assembled.
// Each call to Solve() advances the simulation by one time step.
class LeapfrogSolver : public SolverBase {
  public:
    LeapfrogSolver(ElementBase* data) : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()) {
        // Validate: all element types are supported; guard against unknown future types.
        switch (data->type) {
            case TYPE_3243:
            case TYPE_3443:
            case TYPE_T10:
            case TYPE_T4:
            case TYPE_LDPM_TET4:
                break;
            default:
                MOPHI_ERROR(
                    "LeapfrogSolver: unknown element type %s.",
                    ElementTypeToString(data->type));
                return;  // unreachable; MOPHI_ERROR is fatal
        }

        type_ = data->type;
        data->PrepareSolverData();       // no-op for FEAT/LDPM; calls CalcDsDuPre for ANCF
        d_data_ = data->GetDevicePtr();  // device-side copy pointer
        auto info = GetElementDispatchInfo(type_);
        n_total_qp_ = info.n_total_qp;
        n_shape_ = info.n_shape;

        cudaMalloc(&d_time_step_, sizeof(Real));
        cudaMalloc(&d_leapfrog_solver_, sizeof(LeapfrogSolver));

        // For TYPE_LDPM_TET4 the velocity array holds both translational and
        // rotational half-step velocities:
        //   [v_trans (3*n_coef_) | v_rot (3*n_coef_)]
        // All other types use the standard 3*n_coef_ translational layout.
        const size_t v_size =
            (type_ == TYPE_LDPM_TET4) ? static_cast<size_t>(n_coef_) * 6 : static_cast<size_t>(n_coef_) * 3;
        da_v_.resize(v_size);
        da_v_.BindDevicePointer(&d_v_);

        // For TYPE_LDPM_TET4 the mass array holds both translational lumped
        // mass and rotational inertia:
        //   [m_trans (n_coef_) | I_rot (n_coef_)]
        const size_t m_size =
            (type_ == TYPE_LDPM_TET4) ? static_cast<size_t>(n_coef_) * 2 : static_cast<size_t>(n_coef_);
        da_mass_lump_.resize(m_size);
        da_mass_lump_.BindDevicePointer(&d_mass_lump_);
    }

    ~LeapfrogSolver() {
        da_v_.free();
        da_mass_lump_.free();

        cudaFree(d_time_step_);
        cudaFree(d_leapfrog_solver_);
    }

    // Copy time step to device and zero nodal velocities to start from rest.
    void SetParameters(void* params) override {
        LeapfrogParams* p = static_cast<LeapfrogParams*>(params);
        cudaMemcpy(d_time_step_, &p->time_step, sizeof(Real), cudaMemcpyHostToDevice);

        da_v_.SetVal(Real(0));
        da_v_.MakeReadyDevice();
    }

    // Copy struct mirror to device and compute the lumped mass from the
    // pre-assembled CSR consistent mass matrix stored in the element data.
    // Must be called after the element's CalcMassMatrix() and after
    // SetParameters().
    void Setup();

#if defined(__CUDACC__)
    // Half-step nodal velocity vector (length 3*n_coef_).
    __device__ Map<VectorXR> v() {
        return Map<VectorXR>(d_v_, n_coef_ * 3);
    }

    // Lumped nodal mass vector (length n_coef_).
    __device__ Map<VectorXR> mass_lump() {
        return Map<VectorXR>(d_mass_lump_, n_coef_);
    }

    // Half-step rotational velocity vector for TYPE_LDPM_TET4 (length 3*n_coef_).
    // Stored immediately after the translational velocities in da_v_.
    __device__ Map<VectorXR> v_rot() {
        return Map<VectorXR>(d_v_ + n_coef_ * 3, n_coef_ * 3);
    }

    // Per-node rotational inertia for TYPE_LDPM_TET4 (length n_coef_).
    // Stored immediately after the translational masses in da_mass_lump_.
    __device__ Map<VectorXR> mass_inertia() {
        return Map<VectorXR>(d_mass_lump_ + n_coef_, n_coef_);
    }

    __device__ Real solver_time_step() const {
        return *d_time_step_;
    }
#endif

    __host__ __device__ int get_n_coef() const {
        return n_coef_;
    }
    __host__ __device__ int get_n_beam() const {
        return n_beam_;
    }
    __host__ __device__ int get_n_total_qp() const {
        return n_total_qp_;
    }
    __host__ __device__ int get_n_shape() const {
        return n_shape_;
    }

    // Advance one leapfrog time step.
    void OneStepLeapfrog();

    void Solve() override {
        OneStepLeapfrog();
    }

  private:
    ElementType type_;
    // Device pointer to the element data struct (GPU_FEAT4_Data, etc.).
    // Owned by the ElementBase object; not freed here.
    ElementBase* d_data_;
    // Device-side mirror of this solver struct, copied via cudaMemcpy in Setup().
    // Passed directly to GPU kernels so they can access all device pointers and
    // scalar fields without re-deriving the host address.
    LeapfrogSolver* d_leapfrog_solver_;
    int n_total_qp_, n_shape_;
    int n_coef_, n_beam_;

    // DualArrays for long arrays (manage both pinned host and device memory).
    // d_v_: half-step nodal velocities, length 3*n_coef_
    mophi::DualArray<Real> da_v_;
    // d_mass_lump_: lumped (diagonal) nodal mass, one scalar per node
    mophi::DualArray<Real> da_mass_lump_;

    // Raw device pointers for GPU kernel access (bound to DualArrays above).
    Real* d_v_;
    Real* d_mass_lump_;

    // Physical simulation time step stored on device for kernel access.
    Real* d_time_step_;
};

}  // namespace tlfea
