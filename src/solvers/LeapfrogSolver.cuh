/*==============================================================
 *==============================================================
 * Project: TLFEA
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    LeapfrogSolver.cuh
 * Brief:   Declares the LeapfrogSolver, a second-order explicit
 *          central-difference (leapfrog) time integrator for solid
 *          mechanics simulations. Owns GPU buffers for half-step nodal
 *          velocities and a lumped (row-summed) mass vector. Supports
 *          ANCF3243, ANCF3443, FEAT10, and FEAT4 element types, with
 *          primary focus on TET4 elements for use with the LDPM model.
 *==============================================================
 *==============================================================*/

#pragma once

#include <MoPhiEssentials.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT4Data.cuh"
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
    LeapfrogSolver(ElementBase* data)
        : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()) {
        if (data->type == TYPE_3243) {
            type_ = TYPE_3243;
            auto* typed_data = static_cast<GPU_ANCF3243_Data*>(data);
            d_data_ = typed_data->d_data;
            n_total_qp_ = Quadrature::N_TOTAL_QP_3_2_2;
            n_shape_ = Quadrature::N_SHAPE_3243;
            typed_data->CalcDsDuPre();
        } else if (data->type == TYPE_3443) {
            type_ = TYPE_3443;
            auto* typed_data = static_cast<GPU_ANCF3443_Data*>(data);
            d_data_ = typed_data->d_data;
            n_total_qp_ = Quadrature::N_TOTAL_QP_4_4_3;
            n_shape_ = Quadrature::N_SHAPE_3443;
            typed_data->CalcDsDuPre();
        } else if (data->type == TYPE_T10) {
            type_ = TYPE_T10;
            auto* typed_data = static_cast<GPU_FEAT10_Data*>(data);
            d_data_ = typed_data->d_data;
            n_total_qp_ = Quadrature::N_QP_T10_5;
            n_shape_ = Quadrature::N_NODE_T10_10;
        } else if (data->type == TYPE_T4) {
            type_ = TYPE_T4;
            auto* typed_data = static_cast<GPU_FEAT4_Data*>(data);
            d_data_ = typed_data->d_data;
            n_total_qp_ = Quadrature::N_QP_T4_1;
            n_shape_ = Quadrature::N_NODE_T4_4;
        } else {
            d_data_ = nullptr;
            MOPHI_ERROR("Unknown element type in LeapfrogSolver!");
        }

        cudaMalloc(&d_time_step_, sizeof(Real));
        cudaMalloc(&d_leapfrog_solver_, sizeof(LeapfrogSolver));

        // Half-step nodal velocities: layout [vx0, vy0, vz0, vx1, vy1, vz1, ...]
        da_v_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_.BindDevicePointer(&d_v_);

        // Lumped (row-sum) nodal mass: one scalar per node.
        da_mass_lump_.resize(static_cast<size_t>(n_coef_));
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
