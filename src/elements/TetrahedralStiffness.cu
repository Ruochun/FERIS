/* Project: FERIS
 * File: TetrahedralStiffness.cu
 * Brief: Shared T4/T10 tangent assembly used by static and dynamic clients.
 */
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "FEAT4Data.cuh"
#include "FEAT10Data.cuh"
#include "FEAT4DataFunc.cuh"
#include "FEAT10DataFunc.cuh"
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

template <class TData>
void AssembleTetrahedralTangent(TData* data_,
                                mophi::DualArray<int>& da_K_offsets_,
                                mophi::DualArray<int>& da_K_columns_,
                                mophi::DualArray<Real>& da_K_values_) {
    int n_dof_ = 3 * data_->get_n_coef(), K_nnz_ = 0;
    int *d_K_offsets_ = nullptr, *d_K_columns_ = nullptr;
    Real* d_K_values_ = nullptr;
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
    d_K_offsets_ = da_K_offsets_.device();
    da_K_columns_.resize(static_cast<size_t>(K_nnz_));
    d_K_columns_ = da_K_columns_.device();
    da_K_values_.resize(static_cast<size_t>(K_nnz_));
    d_K_values_ = da_K_values_.device();

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

    da_K_values_.SetVal(Real(0));
    da_K_values_.ToDevice();

    const int total_qp = data_->get_n_elem() * TData::N_QP_PER_ELEM;
    constexpr int threads = 128;
    const int blocks = (total_qp + threads - 1) / threads;

    assemble_stiffness_kernel<TData><<<blocks, threads>>>(data_->d_data, d_K_offsets_, d_K_columns_, d_K_values_);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}
template <class TData>
HostCsrMatrix DownloadTetrahedralTangent(TData* data_) {
    mophi::DualArray<int> da_K_offsets_, da_K_columns_;
    mophi::DualArray<Real> da_K_values_;
    AssembleTetrahedralTangent(data_, da_K_offsets_, da_K_columns_, da_K_values_);
    const int n_dof_ = 3 * data_->get_n_coef(), K_nnz_ = da_K_values_.size();
    HostCsrMatrix result;
    da_K_offsets_.ToHost();
    da_K_columns_.ToHost();
    da_K_values_.ToHost();
    result.offsets.assign(da_K_offsets_.host(), da_K_offsets_.host() + n_dof_ + 1);
    result.columns.assign(da_K_columns_.host(), da_K_columns_.host() + K_nnz_);
    result.values.assign(da_K_values_.host(), da_K_values_.host() + K_nnz_);
    da_K_offsets_.free();
    da_K_columns_.free();
    da_K_values_.free();
    return result;
}
HostCsrMatrix GPU_FEAT4_Data::AssembleTangentStiffnessCSR() {
    return DownloadTetrahedralTangent(this);
}
HostCsrMatrix GPU_FEAT10_Data::AssembleTangentStiffnessCSR() {
    return DownloadTetrahedralTangent(this);
}
void GPU_FEAT4_Data::AssembleTangentStiffnessCSR(mophi::DualArray<int>& offsets,
                                                 mophi::DualArray<int>& columns,
                                                 mophi::DualArray<Real>& values) {
    AssembleTetrahedralTangent(this, offsets, columns, values);
}
void GPU_FEAT10_Data::AssembleTangentStiffnessCSR(mophi::DualArray<int>& offsets,
                                                  mophi::DualArray<int>& columns,
                                                  mophi::DualArray<Real>& values) {
    AssembleTetrahedralTangent(this, offsets, columns, values);
}
}  // namespace feris
