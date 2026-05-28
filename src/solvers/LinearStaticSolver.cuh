#pragma once
/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LinearStaticSolver.cuh
 * Brief:   Declares the LinearStaticSolver class template, which performs a
 *          single-step steady-state (linear) FEA solve for tetrahedral
 *          elements entirely on the GPU.
 *
 *          The TData template parameter must be GPU_FEAT10_Data (TET10) or
 *          GPU_FEAT4_Data (TET4). Passing any other element type triggers a
 *          static_assert with a clear error message. Both element types must
 *          provide:
 *            - get_n_elem(), get_n_coef(), get_n_constraint()
 *            - static constexpr N_NODES_PER_ELEM, N_QP_PER_ELEM
 *            - GetDevicePtr() (GPU copy of the struct)
 *            - GetXCurDevicePtr(), GetYCurDevicePtr(), GetZCurDevicePtr()
 *            - GetExternalForceDevicePtr()
 *            - SetNodalFixed()
 *
 *          The solver:
 *            1. Builds the linearised (small-strain) tangent stiffness matrix K
 *               in 3N x 3N CSR format using the existing SVK tangent block
 *               evaluated at the reference configuration (F = I).
 *            2. Applies Dirichlet boundary conditions by row/column elimination.
 *            3. Solves K*u = f_ext via a conjugate-gradient loop implemented
 *               with cuBLAS (dot, axpy, scal) and cuSPARSE (SpMV).
 *            4. Updates the nodal positions stored in TData.
 *==============================================================
 *==============================================================*/

#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <MoPhiEssentials.h>

#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../types.h"
#include "SolverBase.h"

namespace feris {

template <typename TData>
class LinearStaticSolver : public SolverBase {
    static_assert(std::is_same<TData, GPU_FEAT10_Data>::value || std::is_same<TData, GPU_FEAT4_Data>::value,
                  "LinearStaticSolver only supports GPU_FEAT10_Data (TYPE_T10) and GPU_FEAT4_Data (TYPE_T4). "
                  "For ANCF elements use SyncedAdamWSolver or SyncedNesterovSolver. "
                  "For LDPM particles use LeapfrogSolver.");

  public:
    /**
     * Constructor.
     *
     * @param data      Fully initialised TData (GPU_FEAT10_Data or GPU_FEAT4_Data).
     *                  The caller must have already invoked Setup(), SetSVK(),
     *                  CalcDnDuPre(), and (optionally) SetNodalFixed() /
     *                  SetExternalForce() before calling Solve().
     * @param tol       CG relative-residual convergence tolerance.
     * @param max_iter  Maximum number of CG iterations.
     */
    LinearStaticSolver(TData* data, Real tol = 1e-10, int max_iter = 10000);
    ~LinearStaticSolver();

    /**
     * Assemble K, apply BCs, solve K*u = f_ext on GPU, then write the
     * computed displacement back into the TData nodal positions.
     */
    void Solve() override;

    /** No-op: parameters are set through the constructor. */
    void SetParameters(void* /*params*/) override {}

    /** Number of CG iterations performed during the last Solve() call. */
    int GetLastIterCount() const { return last_iter_count_; }

    /** Relative residual norm at the end of the last Solve() call. */
    Real GetLastResidual() const { return last_residual_; }

  private:
    /* ---- internal stages ---- */
    void BuildStiffnessCSRPattern();
    void AssembleLinearStiffness();
    void ApplyDirichletBCs();
    void SolveLinearSystemCG();
    void UpdatePositions();

    /* ---- data ---- */
    TData* data_;
    int n_dof_;  // 3 * n_nodes

    /* DualArrays for long arrays (manage both pinned host and device memory). */
    mophi::DualArray<int> da_K_offsets_, da_K_columns_;
    mophi::DualArray<Real> da_K_values_;
    mophi::DualArray<Real> da_u_, da_f_, da_r_, da_p_, da_Kp_;

    /* Stiffness K in CSR format (n_dof x n_dof) -- raw device pointers bound to DualArrays */
    int* d_K_offsets_ = nullptr;
    int* d_K_columns_ = nullptr;
    Real* d_K_values_ = nullptr;
    int K_nnz_ = 0;
    bool pattern_built_ = false;

    /* CG workspace (length n_dof each) -- raw device pointers bound to DualArrays */
    Real* d_u_ = nullptr;   // displacement solution
    Real* d_f_ = nullptr;   // RHS (copy of f_ext, modified in-place for BCs)
    Real* d_r_ = nullptr;   // residual
    Real* d_p_ = nullptr;   // search direction
    Real* d_Kp_ = nullptr;  // K * p

    /* cuBLAS / cuSPARSE handles */
    cusparseHandle_t cusparse_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    /* CG parameters */
    Real cg_tol_;
    int cg_max_iter_;

    /* Diagnostics from last Solve() */
    int last_iter_count_ = 0;
    Real last_residual_ = 0.0;
};

// Convenience type aliases for common element types
using LinearStaticSolverT10 = LinearStaticSolver<GPU_FEAT10_Data>;
using LinearStaticSolverT4 = LinearStaticSolver<GPU_FEAT4_Data>;

}  // namespace feris
