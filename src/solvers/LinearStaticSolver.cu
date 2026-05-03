/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LinearStaticSolver.cu
 * Brief:   Implements the GPU-side steady-state (linear) FEA solver for
 *          tetrahedral elements (TET10 via GPU_FEAT10_Data, TET4 via
 *          GPU_FEAT4_Data). The TData template parameter selects the element
 *          type; explicit instantiations for both types are provided at the
 *          bottom of this file.
 *
 *          Key stages performed inside Solve():
 *            1.  BuildStiffnessCSRPattern  - constructs the 3N x 3N sparsity
 *                pattern once using thrust sort + unique on (row,col) keys
 *                generated from element connectivity.
 *            2.  AssembleLinearStiffness   - fills K values via the existing
 *                compute_hessian_assemble_csr device function (SVK tangent at
 *                the reference configuration where F = I gives the classical
 *                linear-elastic stiffness).
 *            3.  ApplyDirichletBCs         - enforces fixed DOFs by row/column
 *                elimination (identity rows, zeroed symmetric columns, zero
 *                RHS entries).
 *            4.  SolveLinearSystemCG       - CG loop using cuSPARSE SpMV and
 *                cuBLAS dot / axpy / scal.
 *            5.  UpdatePositions           - writes u back into the TData
 *                node-position arrays.
 *==============================================================
 *==============================================================*/

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdio>

#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/FEAT4DataFunc.cuh"
#include "../utils/cuda_utils.h"
#include "LinearStaticSolver.cuh"

// cuBLAS error checking
#ifndef CHECK_CUBLAS
    #define CHECK_CUBLAS(func)                                                                            \
        {                                                                                                 \
            cublasStatus_t status = (func);                                                               \
            if (status != CUBLAS_STATUS_SUCCESS) {                                                        \
                MOPHI_ERROR("cuBLAS API failed with error code %d at %s:%d", status, __FILE__, __LINE__); \
            }                                                                                             \
        }
#endif

namespace feris {

// ---------------------------------------------------------------------------
// Helper: set the last entry of an offset array.
// ---------------------------------------------------------------------------
namespace {
__global__ void ls_set_last_offset_kernel(int* d_offsets, int n_rows, int nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_offsets[n_rows] = nnz;
    }
}
}  // namespace

// ---------------------------------------------------------------------------
// Pattern building step 1: generate raw (row<<32|col) keys.
//
// TData::N_NODES_PER_ELEM is a static constexpr that gives the number of
// nodes per element (10 for TET10, 4 for TET4).
// ---------------------------------------------------------------------------
template <typename TData>
__global__ void build_stiffness_keys_kernel(TData* d_data, unsigned long long* d_keys) {
    constexpr int N_NODES = TData::N_NODES_PER_ELEM;
    constexpr int STIFFNESS_KEYS_PER_ELEM = N_NODES * N_NODES * 9;

    const int total = d_data->gpu_n_elem() * STIFFNESS_KEYS_PER_ELEM;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;

    const int elem = tid / STIFFNESS_KEYS_PER_ELEM;
    const int rem = tid % STIFFNESS_KEYS_PER_ELEM;
    const int node_pair = rem / 9;
    const int dof_pair = rem % 9;

    const int i_local = node_pair / N_NODES;
    const int j_local = node_pair % N_NODES;
    const int dof_d = dof_pair / 3;
    const int dof_e = dof_pair % 3;

    const int global_i = d_data->element_connectivity()(elem, i_local);
    const int global_j = d_data->element_connectivity()(elem, j_local);

    const int row = 3 * global_i + dof_d;
    const int col = 3 * global_j + dof_e;

    d_keys[tid] = (static_cast<unsigned long long>(static_cast<unsigned int>(row)) << 32) |
                  static_cast<unsigned long long>(static_cast<unsigned int>(col));
}

// ---------------------------------------------------------------------------
// Pattern building step 2: decode unique keys into CSR columns + row counts.
// ---------------------------------------------------------------------------
__global__ void decode_stiffness_keys_kernel(const unsigned long long* d_keys,
                                             int nnz,
                                             int* d_K_columns,
                                             int* d_row_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz)
        return;

    const unsigned long long key = d_keys[tid];
    const int row = static_cast<int>(key >> 32);
    const int col = static_cast<int>(key & 0xffffffffULL);

    d_K_columns[tid] = col;
    atomicAdd(d_row_counts + row, 1);
}

// ---------------------------------------------------------------------------
// Stiffness assembly: one thread per (element, quadrature-point) pair.
// ---------------------------------------------------------------------------
template <typename TData>
__global__ void assemble_stiffness_kernel(TData* d_data, int* d_K_offsets, int* d_K_columns, Real* d_K_values) {
    constexpr int N_QP = TData::N_QP_PER_ELEM;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elem = tid / N_QP;
    const int qp = tid % N_QP;

    if (elem >= d_data->gpu_n_elem())
        return;

    // h = 1.0: no time-step scaling for static assembly.
    compute_hessian_assemble_csr<TData>(d_data, static_cast<SyncedNewtonSolver*>(nullptr), elem, qp, d_K_offsets,
                                        d_K_columns, d_K_values, 1.0);
}

// ---------------------------------------------------------------------------
// Boundary condition application - phase 1: mark which DOFs are constrained.
// ---------------------------------------------------------------------------
template <typename TData>
__global__ void mark_fixed_dofs_kernel(TData* d_data, int* d_is_fixed) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_fixed = d_data->gpu_n_constraint() / 3;
    if (tid >= n_fixed)
        return;

    const int node = d_data->fixed_nodes()(tid);
    d_is_fixed[3 * node + 0] = 1;
    d_is_fixed[3 * node + 1] = 1;
    d_is_fixed[3 * node + 2] = 1;
}

// ---------------------------------------------------------------------------
// BC application - phase 2: zero/identity rows and columns for fixed DOFs.
// ---------------------------------------------------------------------------
__global__ void apply_bcs_kernel(const int* d_is_fixed,
                                 const int* K_offsets,
                                 const int* K_columns,
                                 Real* K_values,
                                 Real* f,
                                 int n_dof) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_dof)
        return;

    const int row_start = K_offsets[row];
    const int row_end = K_offsets[row + 1];

    if (d_is_fixed[row]) {
        for (int k = row_start; k < row_end; ++k)
            K_values[k] = (K_columns[k] == row) ? static_cast<Real>(1) : static_cast<Real>(0);
        f[row] = static_cast<Real>(0);
    } else {
        for (int k = row_start; k < row_end; ++k)
            if (d_is_fixed[K_columns[k]])
                K_values[k] = static_cast<Real>(0);
    }
}

// ---------------------------------------------------------------------------
// Position update: x12_i += u_{3i}, y12_i += u_{3i+1}, z12_i += u_{3i+2}.
// ---------------------------------------------------------------------------
__global__ void update_positions_kernel(Real* d_x12, Real* d_y12, Real* d_z12, const Real* d_u, int n_nodes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes)
        return;
    d_x12[i] += d_u[3 * i + 0];
    d_y12[i] += d_u[3 * i + 1];
    d_z12[i] += d_u[3 * i + 2];
}

// ===========================================================================
// LinearStaticSolver<TData> -- host-side methods
// ===========================================================================

template <typename TData>
LinearStaticSolver<TData>::LinearStaticSolver(TData* data, Real tol, int max_iter)
    : data_(data), n_dof_(3 * data->get_n_coef()), cg_tol_(tol), cg_max_iter_(max_iter) {
    CHECK_CUSPARSE(cusparseCreate(&cusparse_));
    CHECK_CUBLAS(cublasCreate(&cublas_));

    da_u_.resize(static_cast<size_t>(n_dof_));
    da_u_.BindDevicePointer(&d_u_);
    da_f_.resize(static_cast<size_t>(n_dof_));
    da_f_.BindDevicePointer(&d_f_);
    da_r_.resize(static_cast<size_t>(n_dof_));
    da_r_.BindDevicePointer(&d_r_);
    da_p_.resize(static_cast<size_t>(n_dof_));
    da_p_.BindDevicePointer(&d_p_);
    da_Kp_.resize(static_cast<size_t>(n_dof_));
    da_Kp_.BindDevicePointer(&d_Kp_);
}

template <typename TData>
LinearStaticSolver<TData>::~LinearStaticSolver() {
    da_K_offsets_.free();
    da_K_columns_.free();
    da_K_values_.free();
    da_u_.free();
    da_f_.free();
    da_r_.free();
    da_p_.free();
    da_Kp_.free();
    if (cusparse_)
        cusparseDestroy(cusparse_);
    if (cublas_)
        cublasDestroy(cublas_);
}

// ---------------------------------------------------------------------------
// BuildStiffnessCSRPattern
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::BuildStiffnessCSRPattern() {
    if (pattern_built_)
        return;

    constexpr int N_NODES = TData::N_NODES_PER_ELEM;
    constexpr int STIFFNESS_KEYS_PER_ELEM = N_NODES * N_NODES * 9;

    const int n_elem = data_->get_n_elem();
    const int total_raw = n_elem * STIFFNESS_KEYS_PER_ELEM;

    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys, static_cast<size_t>(total_raw) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_raw + threads - 1) / threads;
        build_stiffness_keys_kernel<TData><<<blocks, threads>>>(data_->d_data, d_keys);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<unsigned long long> keys_begin(d_keys);
    thrust::device_ptr<unsigned long long> keys_end = keys_begin + total_raw;
    thrust::sort(thrust::device, keys_begin, keys_end);
    auto keys_unique_end = thrust::unique(thrust::device, keys_begin, keys_end);
    K_nnz_ = static_cast<int>(keys_unique_end - keys_begin);

    da_K_offsets_.resize(static_cast<size_t>(n_dof_ + 1));
    da_K_offsets_.BindDevicePointer(&d_K_offsets_);
    da_K_columns_.resize(static_cast<size_t>(K_nnz_));
    da_K_columns_.BindDevicePointer(&d_K_columns_);
    da_K_values_.resize(static_cast<size_t>(K_nnz_));
    da_K_values_.BindDevicePointer(&d_K_values_);

    int* d_row_counts = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_row_counts, static_cast<size_t>(n_dof_) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_row_counts, 0, static_cast<size_t>(n_dof_) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (K_nnz_ + threads - 1) / threads;
        decode_stiffness_keys_kernel<<<blocks, threads>>>(d_keys, K_nnz_, d_K_columns_, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> row_counts_ptr(d_row_counts);
    thrust::device_ptr<int> offsets_ptr(d_K_offsets_);
    thrust::exclusive_scan(thrust::device, row_counts_ptr, row_counts_ptr + n_dof_, offsets_ptr);
    ls_set_last_offset_kernel<<<1, 1>>>(d_K_offsets_, n_dof_, K_nnz_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    pattern_built_ = true;
}

// ---------------------------------------------------------------------------
// AssembleLinearStiffness
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::AssembleLinearStiffness() {
    da_K_values_.SetVal(Real(0));
    da_K_values_.ToDevice();

    const int total_qp = data_->get_n_elem() * TData::N_QP_PER_ELEM;
    constexpr int threads = 128;
    const int blocks = (total_qp + threads - 1) / threads;

    assemble_stiffness_kernel<TData><<<blocks, threads>>>(data_->d_data, d_K_offsets_, d_K_columns_, d_K_values_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// ApplyDirichletBCs
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::ApplyDirichletBCs() {
    const int n_constraint = data_->get_n_constraint();
    if (n_constraint == 0)
        return;

    const int n_fixed_nodes = n_constraint / 3;

    int* d_is_fixed = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_is_fixed, static_cast<size_t>(n_dof_) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_is_fixed, 0, static_cast<size_t>(n_dof_) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_fixed_nodes + threads - 1) / threads;
        mark_fixed_dofs_kernel<TData><<<blocks, threads>>>(data_->d_data, d_is_fixed);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    {
        constexpr int threads = 256;
        const int blocks = (n_dof_ + threads - 1) / threads;
        apply_bcs_kernel<<<blocks, threads>>>(d_is_fixed, d_K_offsets_, d_K_columns_, d_K_values_, d_f_, n_dof_);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaFree(d_is_fixed));
}

// ---------------------------------------------------------------------------
// SolveLinearSystemCG
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::SolveLinearSystemCG() {
    const int n = n_dof_;

    da_u_.SetVal(Real(0));
    da_u_.ToDevice();
    MOPHI_GPU_CALL(cudaMemcpy(d_r_, d_f_, static_cast<size_t>(n) * sizeof(Real), cudaMemcpyDeviceToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_p_, d_f_, static_cast<size_t>(n) * sizeof(Real), cudaMemcpyDeviceToDevice));

    cusparseSpMatDescr_t K_descr = nullptr;
    cusparseDnVecDescr_t p_descr = nullptr;
    cusparseDnVecDescr_t Kp_descr = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(&K_descr, n, n, K_nnz_, d_K_offsets_, d_K_columns_, d_K_values_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateDnVec(&p_descr, n, d_p_, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&Kp_descr, n, d_Kp_, CUDA_R_64F));

    const Real one = 1.0;
    const Real zero = 0.0;

    size_t spMV_bufsize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, K_descr, p_descr, &zero,
                                           Kp_descr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spMV_bufsize));

    void* spMV_buf = nullptr;
    if (spMV_bufsize > 0) {
        MOPHI_GPU_CALL(cudaMalloc(&spMV_buf, spMV_bufsize));
    }

    Real rr0 = 0.0;
    CHECK_CUBLAS(cublasDdot(cublas_, n, d_r_, 1, d_r_, 1, &rr0));

    if (rr0 == 0.0) {
        last_iter_count_ = 0;
        last_residual_ = 0.0;
        CHECK_CUSPARSE(cusparseDestroySpMat(K_descr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(p_descr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(Kp_descr));
        if (spMV_buf)
            cudaFree(spMV_buf);
        return;
    }

    Real rr = rr0;

    last_iter_count_ = cg_max_iter_;
    last_residual_ = 1.0;

    for (int iter = 0; iter < cg_max_iter_; ++iter) {
        CHECK_CUSPARSE(cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, K_descr, p_descr, &zero,
                                    Kp_descr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spMV_buf));

        Real pKp = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_p_, 1, d_Kp_, 1, &pKp));

        if (pKp <= 0.0) {
            MOPHI_ERROR("LinearStaticSolver CG: non-positive p*Kp detected; aborting.");
            break;
        }
        Real alpha = rr / pKp;

        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &alpha, d_p_, 1, d_u_, 1));

        const Real neg_alpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &neg_alpha, d_Kp_, 1, d_r_, 1));

        Real rr_new = 0.0;
        CHECK_CUBLAS(cublasDdot(cublas_, n, d_r_, 1, d_r_, 1, &rr_new));

        const Real rel_res = (rr0 > 0.0) ? std::sqrt(rr_new / rr0) : std::sqrt(rr_new);
        if (rel_res < cg_tol_) {
            last_iter_count_ = iter + 1;
            last_residual_ = rel_res;
            break;
        }

        const Real beta = rr_new / rr;

        CHECK_CUBLAS(cublasDscal(cublas_, n, &beta, d_p_, 1));
        CHECK_CUBLAS(cublasDaxpy(cublas_, n, &one, d_r_, 1, d_p_, 1));

        rr = rr_new;

        if (iter == cg_max_iter_ - 1) {
            last_iter_count_ = cg_max_iter_;
            last_residual_ = rel_res;
        }
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(K_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(p_descr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(Kp_descr));
    if (spMV_buf)
        cudaFree(spMV_buf);
}

// ---------------------------------------------------------------------------
// UpdatePositions -- add displacement u to TData positions.
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::UpdatePositions() {
    const int n_nodes = data_->get_n_coef();
    constexpr int threads = 256;
    const int blocks = (n_nodes + threads - 1) / threads;

    Real* d_x = const_cast<Real*>(data_->GetX12DevicePtr());
    Real* d_y = const_cast<Real*>(data_->GetY12DevicePtr());
    Real* d_z = const_cast<Real*>(data_->GetZ12DevicePtr());

    update_positions_kernel<<<blocks, threads>>>(d_x, d_y, d_z, d_u_, n_nodes);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    // Sync the host-visible struct copy on GPU.
    MOPHI_GPU_CALL(cudaMemcpy(data_->d_data, data_, sizeof(TData), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Solve -- top-level entry point.
// ---------------------------------------------------------------------------
template <typename TData>
void LinearStaticSolver<TData>::Solve() {
    BuildStiffnessCSRPattern();
    AssembleLinearStiffness();

    {
        const Real* d_f_ext = data_->GetExternalForceDevicePtr();
        MOPHI_GPU_CALL(cudaMemcpy(d_f_, d_f_ext, static_cast<size_t>(n_dof_) * sizeof(Real), cudaMemcpyDeviceToDevice));
    }

    ApplyDirichletBCs();
    SolveLinearSystemCG();
    UpdatePositions();
}

// ---------------------------------------------------------------------------
// Explicit template instantiations for supported element types
// ---------------------------------------------------------------------------
template class LinearStaticSolver<GPU_FEAT10_Data>;
template class LinearStaticSolver<GPU_FEAT4_Data>;

// Explicit instantiations of device kernels (required for the template kernels)
template __global__ void build_stiffness_keys_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, unsigned long long*);
template __global__ void build_stiffness_keys_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, unsigned long long*);
template __global__ void assemble_stiffness_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, int*, int*, Real*);
template __global__ void assemble_stiffness_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, int*, int*, Real*);
template __global__ void mark_fixed_dofs_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, int*);
template __global__ void mark_fixed_dofs_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, int*);

}  // namespace feris
