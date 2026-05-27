/*==============================================================
 *==============================================================
 * Project: FERIS
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedAdamWNocoop.cuh
 * Brief:   Declares the non-cooperative GPU-synchronized AdamW solver.
 *          Provides host/device storage and kernels to evaluate residuals and
 *          apply AdamW updates for ANCF3243, ANCF3443, FEAT10, and FEAT4
 *          element data.
 *
 *          Supported element types: TYPE_3243, TYPE_3443, TYPE_T10, TYPE_T4.
 *          See docs/SOLVER_ELEMENT_COMPATIBILITY.md for the full matrix.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cublas_v2.h>

#include <iostream>

#include "SyncedAdamW.cuh"

namespace feris {

using SyncedAdamWNocoopParams = SyncedAdamWParams;

class SyncedAdamWNocoopSolver : public SolverBase {
  public:
    SyncedAdamWNocoopSolver(ElementBase* data, int n_constraints)
        : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()), n_constraints_(n_constraints) {
        // Validate: this solver only supports continuum and ANCF element types.
        switch (data->type) {
            case TYPE_3243:
            case TYPE_3443:
            case TYPE_T10:
            case TYPE_T4:
                break;
            default:
                MOPHI_ERROR(
                    "SyncedAdamWNocoopSolver does not support element type %s. "
                    "Supported types: TYPE_3243, TYPE_3443, TYPE_T10, TYPE_T4. "
                    "For TYPE_LDPM_TET4 use LeapfrogSolver instead.",
                    ElementTypeToString(data->type));
                return;  // unreachable; MOPHI_ERROR is fatal
        }

        type_ = data->type;
        data->PrepareSolverData();       // no-op for FEAT; calls CalcDsDuPre for ANCF
        d_data_ = data->GetDevicePtr();  // device-side copy pointer
        d_constraint_ptr_ = data->IsConstraintSetup() ? data->GetConstraintDevicePtr() : nullptr;
        auto info = GetElementDispatchInfo(type_);
        n_total_qp_ = info.n_total_qp;
        n_shape_ = info.n_shape;

        cudaMalloc(&d_norm_g_, sizeof(Real));
        cudaMalloc(&d_norm_v_, sizeof(Real));
        cudaMalloc(&d_inner_flag_, sizeof(int));
        cudaMalloc(&d_outer_flag_, sizeof(int));
        cudaMalloc(&d_alpha_, sizeof(Real));
        cudaMalloc(&d_inner_tol_, sizeof(Real));
        cudaMalloc(&d_inner_rtol_, sizeof(Real));
        cudaMalloc(&d_outer_tol_, sizeof(Real));
        cudaMalloc(&d_max_outer_, sizeof(int));
        cudaMalloc(&d_max_inner_, sizeof(int));
        cudaMalloc(&d_time_step_, sizeof(Real));
        cudaMalloc(&d_solver_rho_, sizeof(Real));
        cudaMalloc(&d_convergence_check_interval_, sizeof(int));

        cudaMalloc(&d_adamw_solver_, sizeof(SyncedAdamWNocoopSolver));

        cudaMalloc(&d_lr_, sizeof(Real));
        cudaMalloc(&d_beta1_, sizeof(Real));
        cudaMalloc(&d_beta2_, sizeof(Real));
        cudaMalloc(&d_eps_, sizeof(Real));
        cudaMalloc(&d_weight_decay_, sizeof(Real));
        cudaMalloc(&d_lr_decay_, sizeof(Real));

        // Long arrays: use DualArray (manages both pinned host and device memory).
        da_v_guess_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_guess_.BindDevicePointer(&d_v_guess_);
        da_v_prev_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_prev_.BindDevicePointer(&d_v_prev_);
        da_v_k_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_k_.BindDevicePointer(&d_v_k_);
        da_v_next_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_next_.BindDevicePointer(&d_v_next_);
        da_lambda_guess_.resize(static_cast<size_t>(n_constraints_));
        da_lambda_guess_.BindDevicePointer(&d_lambda_guess_);
        da_g_.resize(static_cast<size_t>(n_coef_) * 3);
        da_g_.BindDevicePointer(&d_g_);
        da_x12_prev_.resize(static_cast<size_t>(n_coef_));
        da_x12_prev_.BindDevicePointer(&d_x12_prev);
        da_y12_prev_.resize(static_cast<size_t>(n_coef_));
        da_y12_prev_.BindDevicePointer(&d_y12_prev);
        da_z12_prev_.resize(static_cast<size_t>(n_coef_));
        da_z12_prev_.BindDevicePointer(&d_z12_prev);
        da_m_.resize(static_cast<size_t>(n_coef_) * 3);
        da_m_.BindDevicePointer(&d_m_);
        da_v_adam_.resize(static_cast<size_t>(n_coef_) * 3);
        da_v_adam_.BindDevicePointer(&d_v_adam_);

        cublasCreate(&cublas_handle_);
        cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE);
        cudaMalloc(&d_norm_temp_cublas_, sizeof(Real));
    }

    ~SyncedAdamWNocoopSolver() {
        // Long arrays managed by DualArrays
        da_v_guess_.free();
        da_v_prev_.free();
        da_v_k_.free();
        da_v_next_.free();
        da_lambda_guess_.free();
        da_g_.free();
        da_x12_prev_.free();
        da_y12_prev_.free();
        da_z12_prev_.free();
        da_m_.free();
        da_v_adam_.free();

        cudaFree(d_norm_g_);
        cudaFree(d_norm_v_);
        cudaFree(d_inner_flag_);
        cudaFree(d_outer_flag_);
        cudaFree(d_alpha_);
        cudaFree(d_inner_tol_);
        cudaFree(d_inner_rtol_);
        cudaFree(d_outer_tol_);
        cudaFree(d_max_outer_);
        cudaFree(d_max_inner_);
        cudaFree(d_time_step_);
        cudaFree(d_solver_rho_);
        cudaFree(d_convergence_check_interval_);

        cudaFree(d_lr_);
        cudaFree(d_beta1_);
        cudaFree(d_beta2_);
        cudaFree(d_eps_);
        cudaFree(d_weight_decay_);
        cudaFree(d_lr_decay_);

        cudaFree(d_adamw_solver_);

        if (cublas_handle_)
            cublasDestroy(cublas_handle_);
        cudaFree(d_norm_temp_cublas_);
    }

    void SetParameters(void* params) override {
        SyncedAdamWNocoopParams* p = static_cast<SyncedAdamWNocoopParams*>(params);

        cudaMemcpy(d_lr_, &p->lr, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta1_, &p->beta1, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta2_, &p->beta2, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps_, &p->eps, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight_decay_, &p->weight_decay, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lr_decay_, &p->lr_decay, sizeof(Real), cudaMemcpyHostToDevice);

        cudaMemcpy(d_inner_tol_, &p->inner_tol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inner_rtol_, &p->inner_rtol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_outer_tol_, &p->outer_tol, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_outer_, &p->max_outer, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max_inner_, &p->max_inner, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_time_step_, &p->time_step, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_solver_rho_, &p->rho, sizeof(Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_convergence_check_interval_, &p->convergence_check_interval, sizeof(int), cudaMemcpyHostToDevice);

        da_v_guess_.SetVal(Real(0));
        da_v_guess_.ToDevice();
        da_v_prev_.SetVal(Real(0));
        da_v_prev_.ToDevice();
        da_lambda_guess_.SetVal(Real(0));
        da_lambda_guess_.ToDevice();
    }

    void Setup() {
        da_x12_prev_.SetVal(Real(0));
        da_x12_prev_.ToDevice();
        da_y12_prev_.SetVal(Real(0));
        da_y12_prev_.ToDevice();
        da_z12_prev_.SetVal(Real(0));
        da_z12_prev_.ToDevice();

        da_v_guess_.SetVal(Real(0));
        da_v_guess_.ToDevice();
        da_v_prev_.SetVal(Real(0));
        da_v_prev_.ToDevice();
        da_v_k_.SetVal(Real(0));
        da_v_k_.ToDevice();
        da_v_next_.SetVal(Real(0));
        da_v_next_.ToDevice();
        da_lambda_guess_.SetVal(Real(0));
        da_lambda_guess_.ToDevice();
        da_g_.SetVal(Real(0));
        da_g_.ToDevice();

        da_m_.SetVal(Real(0));
        da_m_.ToDevice();
        da_v_adam_.SetVal(Real(0));
        da_v_adam_.ToDevice();

        MOPHI_GPU_CALL(cudaMemcpy(d_adamw_solver_, this, sizeof(SyncedAdamWNocoopSolver), cudaMemcpyHostToDevice));
    }

#if defined(__CUDACC__)
    __device__ Map<VectorXR> v_guess() { return Map<VectorXR>(d_v_guess_, n_coef_ * 3); }
    __device__ Map<VectorXR> v_prev() { return Map<VectorXR>(d_v_prev_, n_coef_ * 3); }
    __device__ Map<VectorXR> v_k() { return Map<VectorXR>(d_v_k_, n_coef_ * 3); }
    __device__ Map<VectorXR> v_next() { return Map<VectorXR>(d_v_next_, n_coef_ * 3); }
    __device__ Map<VectorXR> lambda_guess() { return Map<VectorXR>(d_lambda_guess_, n_constraints_); }
    __device__ Map<VectorXR> g() { return Map<VectorXR>(d_g_, 3 * n_coef_); }
    __device__ Map<VectorXR> m() { return Map<VectorXR>(d_m_, 3 * n_coef_); }
    __device__ Map<VectorXR> v_adam() { return Map<VectorXR>(d_v_adam_, 3 * n_coef_); }

    __device__ int gpu_n_constraints() { return n_constraints_; }
    __device__ int gpu_n_total_qp() { return n_total_qp_; }
    __device__ int gpu_n_shape() { return n_shape_; }

    __device__ Real* norm_g() { return d_norm_g_; }
    __device__ Real* norm_v() { return d_norm_v_; }
    __device__ int* inner_flag() { return d_inner_flag_; }
    __device__ int* outer_flag() { return d_outer_flag_; }
    __device__ Real* solver_rho() { return d_solver_rho_; }
    __device__ Real solver_alpha() const { return *d_alpha_; }
    __device__ Real solver_inner_tol() const { return *d_inner_tol_; }
    __device__ Real solver_inner_rtol() const { return *d_inner_rtol_; }
    __device__ Real solver_outer_tol() const { return *d_outer_tol_; }
    __device__ int solver_max_outer() const { return *d_max_outer_; }
    __device__ int solver_max_inner() const { return *d_max_inner_; }
    __device__ Real solver_time_step() const { return *d_time_step_; }
    __device__ Real solver_lr() const { return *d_lr_; }
    __device__ Real solver_beta1() const { return *d_beta1_; }
    __device__ Real solver_beta2() const { return *d_beta2_; }
    __device__ Real solver_eps() const { return *d_eps_; }
    __device__ Real solver_weight_decay() const { return *d_weight_decay_; }
    __device__ Real solver_lr_decay() const { return *d_lr_decay_; }

    __device__ int solver_convergence_check_interval() const { return *d_convergence_check_interval_; }

    __device__ Map<VectorXR> x12_prev() { return Map<VectorXR>(d_x12_prev, n_coef_); }
    __device__ Map<VectorXR> y12_prev() { return Map<VectorXR>(d_y12_prev, n_coef_); }
    __device__ Map<VectorXR> z12_prev() { return Map<VectorXR>(d_z12_prev, n_coef_); }
#endif

    __host__ __device__ int get_n_coef() const { return n_coef_; }
    __host__ __device__ int get_n_beam() const { return n_beam_; }

    void OneStepAdamWNocoop();

    void Solve() override { OneStepAdamWNocoop(); }

  private:
    Real compute_l2_norm_cublas(Real* d_vec, int n_dofs);

    ElementType type_;
    // Device pointer to the element data struct (GPU_ANCF3243_Data, GPU_ANCF3443_Data,
    // GPU_FEAT10_Data, or GPU_FEAT4_Data). Owned by the ElementBase object; not freed here.
    ElementBase* d_data_;
    // Device-side mirror of this solver struct, copied via cudaMemcpy in Setup().
    // Passed directly to GPU kernels so they can access all device pointers and
    // scalar fields without re-deriving the host address.
    SyncedAdamWNocoopSolver* d_adamw_solver_;
    int n_total_qp_, n_shape_;
    int n_coef_, n_beam_, n_constraints_;

    // Device pointer to the pre-computed constraint Jacobian/data block.
    // Null if no constraints were set up on the element (IsConstraintSetup() == false).
    Real* d_constraint_ptr_;

    // DualArrays for long arrays (manage both pinned host and device memory).
    mophi::DualArray<Real> da_v_guess_, da_v_prev_, da_v_k_, da_v_next_;
    mophi::DualArray<Real> da_lambda_guess_, da_g_;
    mophi::DualArray<Real> da_x12_prev_, da_y12_prev_, da_z12_prev_;
    mophi::DualArray<Real> da_m_, da_v_adam_;

    // Raw device pointers for GPU kernel access (bound to DualArrays above).
    // Nodal x/y/z positions from the previous half-step (used in the momentum update).
    Real *d_x12_prev, *d_y12_prev, *d_z12_prev;
    // Generalized coordinate (velocity/position DOF) vectors at successive AdamW stages:
    //   d_v_guess_  – initial guess for the current outer iteration
    //   d_v_prev_   – accepted solution from the previous outer iteration
    //   d_v_k_      – iterate at the start of the current inner (gradient) step
    //   d_v_next_   – candidate solution after applying the AdamW update
    // Each array has length 3*n_coef_ (x, y, z DOFs concatenated).
    Real *d_v_guess_, *d_v_prev_, *d_v_k_, *d_v_next_;
    // Lagrange multipliers for holonomic constraints (length n_constraints_).
    Real* d_lambda_guess_;
    // Gradient of the total potential energy w.r.t. the DOFs (length 3*n_coef_).
    Real* d_g_;
    // L2 norms stored on device so convergence comparisons stay GPU-resident:
    //   d_norm_g_ – norm of the gradient d_g_ (residual force)
    //   d_norm_v_ – norm of the velocity/displacement update d_v_next_ - d_v_k_
    Real *d_norm_g_, *d_norm_v_;
    // Convergence flags written by GPU kernels; 0 = converged, 1 = not yet converged.
    //   d_inner_flag_ – checked after each inner (gradient) step
    //   d_outer_flag_ – checked after each outer (time) step
    int *d_inner_flag_, *d_outer_flag_;

    // AdamW optimizer hyperparameters stored on the device so kernels can read
    // them without host-device synchronization:
    //   d_lr_           – base learning rate (step size)
    //   d_beta1_        – exponential decay rate for the first moment (momentum) estimate
    //   d_beta2_        – exponential decay rate for the second moment (RMS) estimate
    //   d_eps_          – small constant added to the denominator for numerical stability
    //   d_weight_decay_ – L2 regularisation coefficient applied to the DOF update
    //   d_lr_decay_     – multiplicative decay applied to d_lr_ each outer step
    Real *d_lr_, *d_beta1_, *d_beta2_, *d_eps_, *d_weight_decay_, *d_lr_decay_;

    // Convergence / time-integration scalar parameters stored on the device:
    //   d_alpha_        – gradient step size override (used when lr schedule is bypassed)
    //   d_inner_tol_    – absolute convergence tolerance for the inner loop
    //   d_inner_rtol_   – relative convergence tolerance for the inner loop
    //   d_outer_tol_    – absolute convergence tolerance for the outer loop
    //   d_time_step_    – physical simulation time step
    //   d_solver_rho_   – penalty parameter ρ used in the augmented Lagrangian
    Real *d_alpha_, *d_inner_tol_, *d_inner_rtol_, *d_outer_tol_, *d_time_step_, *d_solver_rho_;

    // How many inner iterations to perform between successive convergence checks
    // (reduces synchronization overhead when n_coef_ is large).
    int* d_convergence_check_interval_;

    // Maximum iteration counts for the inner (gradient) and outer (time) loops.
    int *d_max_inner_, *d_max_outer_;

    // AdamW biased first and second moment estimate vectors (length 3*n_coef_ each):
    //   d_m_ – first moment (exponentially weighted moving average of the gradient)
    //   d_v_adam_ – second moment (exponentially weighted moving average of gradient^2)
    // Both are bias-corrected inside the update kernel before computing the step.
    Real *d_m_, *d_v_adam_;

    cublasHandle_t cublas_handle_;
    // Temporary single-element device buffer written by cuBLAS cublasNrm2 so that
    // the norm result stays on the device and avoids a device-to-host round-trip.
    Real* d_norm_temp_cublas_;
};

}  // namespace feris
