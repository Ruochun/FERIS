#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    FEAT4Data.cu
 * Brief:   Implements GPU-side data management and element kernels for
 *          4-node linear tetrahedral (TET4/FEAT4) elements. Handles
 *          allocation, initialization, mass and stiffness assembly,
 *          internal/external force evaluation, and constraint coupling.
 *==============================================================
 *==============================================================*/

#include "FEAT4Data.cuh"
#include "FEAT4DataFunc.cuh"

namespace cg = cooperative_groups;

namespace feris {

// ---------------------------------------------------------------------------
// Keys kernel for mass CSR pattern building
// ---------------------------------------------------------------------------
__global__ void build_mass_keys_feat4_kernel(GPU_FEAT4_Data* d_data, unsigned long long* d_keys) {
    const int total = d_data->gpu_n_elem() * Quadrature::N_NODE_T4_4 * Quadrature::N_NODE_T4_4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int elem = tid / (Quadrature::N_NODE_T4_4 * Quadrature::N_NODE_T4_4);
    const int item_local = tid % (Quadrature::N_NODE_T4_4 * Quadrature::N_NODE_T4_4);
    const int i_local = item_local / Quadrature::N_NODE_T4_4;
    const int j_local = item_local % Quadrature::N_NODE_T4_4;

    const int i_global = d_data->element_connectivity()(elem, i_local);
    const int j_global = d_data->element_connectivity()(elem, j_local);

    const unsigned long long key = (static_cast<unsigned long long>(static_cast<unsigned int>(i_global)) << 32) |
                                   static_cast<unsigned long long>(static_cast<unsigned int>(j_global));
    d_keys[tid] = key;
}

__global__ void decode_mass_keys_feat4_kernel(const unsigned long long* d_keys,
                                              int nnz,
                                              int* d_csr_columns,
                                              int* d_row_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) {
        return;
    }

    const unsigned long long key = d_keys[tid];
    const int row = static_cast<int>(key >> 32);
    const int col = static_cast<int>(key & 0xffffffffULL);
    d_csr_columns[tid] = col;
    atomicAdd(d_row_counts + row, 1);
}

__global__ void set_last_offset_feat4_kernel(int* d_offsets, int n_rows, int nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_offsets[n_rows] = nnz;
    }
}

__device__ __forceinline__ int binary_search_column_csr_mass_feat4(const int* cols, int n_cols, int target) {
    int left = 0;
    int right = n_cols - 1;
    while (left <= right) {
        int mid = left + ((right - left) >> 1);
        int v = cols[mid];
        if (v == target)
            return mid;
        if (v < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Shape function gradient precomputation for TET4.
// TET4 has linear shape functions with constant gradients in each element.
// The 1-point centroid quadrature is used (qp_idx always 0).
// ---------------------------------------------------------------------------
__global__ void dn_du_pre_feat4_kernel(GPU_FEAT4_Data* d_data) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // One thread per (elem, qp) pair — for TET4 there is exactly 1 QP per elem
    int elem_idx = global_thread_idx / Quadrature::N_QP_T4_1;
    int qp_idx = global_thread_idx % Quadrature::N_QP_T4_1;

    if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T4_1) {
        return;
    }

    // Reference-coordinate (xi, eta, zeta) derivatives for TET4 shape functions:
    //   N1 = 1 - xi - eta - zeta,  N2 = xi,  N3 = eta,  N4 = zeta
    //   dN_dxi[i] = [dNi/dxi, dNi/deta, dNi/dzeta]
    Real dN_dxi[4][3] = {
        {-1.0, -1.0, -1.0},  // dN1/dxi, dN1/deta, dN1/dzeta
        {1.0, 0.0, 0.0},     // dN2/dxi, dN2/deta, dN2/dzeta
        {0.0, 1.0, 0.0},     // dN3/dxi, dN3/deta, dN3/dzeta
        {0.0, 0.0, 1.0}      // dN4/dxi, dN4/deta, dN4/dzeta
    };

    // Get element node coordinates
    Real X_elem[4][3];
    for (int node = 0; node < 4; node++) {
        int global_node_idx = d_data->element_connectivity()(elem_idx, node);
        X_elem[node][0] = d_data->x12()(global_node_idx);
        X_elem[node][1] = d_data->y12()(global_node_idx);
        X_elem[node][2] = d_data->z12()(global_node_idx);
    }

    // Jacobian J = Σ(X_node ⊗ dN_dxi) for 4-node tet
    Real J[3][3] = {{0.0}};
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                J[i][j] += X_elem[a][i] * dN_dxi[a][j];
            }
        }
    }

    // Determinant
    Real detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    d_data->detJ_ref(elem_idx, qp_idx) = detJ;

    // Transpose of J
    Real JT[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            JT[i][j] = J[j][i];
        }
    }

    // Solve JT * grad_N = dN_dxi for each of the 4 shape functions
    // Using Cramer's rule / Gaussian elimination (solve_3x3_system from FEAT10DataFunc.cuh
    // is not available here, but we implement inline)
    Real detJT = JT[0][0] * (JT[1][1] * JT[2][2] - JT[1][2] * JT[2][1]) -
                 JT[0][1] * (JT[1][0] * JT[2][2] - JT[1][2] * JT[2][0]) +
                 JT[0][2] * (JT[1][0] * JT[2][1] - JT[1][1] * JT[2][0]);

    // Inverse of JT (3x3) using cofactor formula: JT_inv[i][j] = cofactor[j][i] / det
    Real inv_det = (detJT != 0.0) ? 1.0 / detJT : 0.0;

    Real JT_inv[3][3];
    JT_inv[0][0] = (JT[1][1] * JT[2][2] - JT[1][2] * JT[2][1]) * inv_det;
    JT_inv[0][1] = -(JT[0][1] * JT[2][2] - JT[0][2] * JT[2][1]) * inv_det;
    JT_inv[0][2] = (JT[0][1] * JT[1][2] - JT[0][2] * JT[1][1]) * inv_det;
    JT_inv[1][0] = -(JT[1][0] * JT[2][2] - JT[1][2] * JT[2][0]) * inv_det;
    JT_inv[1][1] = (JT[0][0] * JT[2][2] - JT[0][2] * JT[2][0]) * inv_det;
    JT_inv[1][2] = -(JT[0][0] * JT[1][2] - JT[0][2] * JT[1][0]) * inv_det;
    JT_inv[2][0] = (JT[1][0] * JT[2][1] - JT[1][1] * JT[2][0]) * inv_det;
    JT_inv[2][1] = -(JT[0][0] * JT[2][1] - JT[0][1] * JT[2][0]) * inv_det;
    JT_inv[2][2] = (JT[0][0] * JT[1][1] - JT[0][1] * JT[1][0]) * inv_det;

    // Physical gradients: grad_N[a] = JT_inv * dN_dxi[a]
    Real grad_N[4][3];
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i < 3; i++) {
            grad_N[a][i] = 0.0;
            for (int j = 0; j < 3; j++) {
                grad_N[a][i] += JT_inv[i][j] * dN_dxi[a][j];
            }
        }
    }

    // Store physical gradients
    for (int i = 0; i < 4; i++) {
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 0) = grad_N[i][0];
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 1) = grad_N[i][1];
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 2) = grad_N[i][2];
    }
}

// ---------------------------------------------------------------------------
// Mass matrix assembly kernel for TET4
// TET4 shape functions: N1 = L1, N2 = L2, N3 = L3, N4 = L4
// At centroid: all N = 1/4
// ---------------------------------------------------------------------------
__global__ void mass_matrix_qp_feat4_kernel(GPU_FEAT4_Data* d_data) {
    int thread_global = blockIdx.x * blockDim.x + threadIdx.x;

    int elem = thread_global / (4 * 4);
    int item_local = thread_global % (4 * 4);

    if (elem >= d_data->gpu_n_elem())
        return;

    int i_local = item_local / 4;
    int j_local = item_local % 4;

    int i_global = d_data->element_connectivity()(elem, i_local);
    int j_global = d_data->element_connectivity()(elem, j_local);

    Real rho = d_data->rho0();
    Real mass_contribution = 0.0;

    // Only 1 QP for TET4
    for (int qp = 0; qp < Quadrature::N_QP_T4_1; qp++) {
        // Quadrature point barycentric coordinates (centroid: L1=L2=L3=L4=1/4)
        Real xi = d_data->tet1pt_x(qp);    // = 0.25
        Real eta = d_data->tet1pt_y(qp);   // = 0.25
        Real zeta = d_data->tet1pt_z(qp);  // = 0.25
        Real wq = d_data->tet1pt_weights(qp);

        // TET4 shape functions are just the barycentric coordinates:
        // N[0] = 1 - xi - eta - zeta, N[1] = xi, N[2] = eta, N[3] = zeta
        Real L1 = 1.0 - xi - eta - zeta;
        Real L2 = xi;
        Real L3 = eta;
        Real L4 = zeta;
        Real N[4] = {L1, L2, L3, L4};

        Real detJ = d_data->detJ_ref(elem, qp);

        mass_contribution += rho * N[i_local] * N[j_local] * detJ * wq;
    }

    const int row_start = d_data->csr_offsets()[i_global];
    const int row_end = d_data->csr_offsets()[i_global + 1];
    const int n_cols = row_end - row_start;
    const int local_idx = binary_search_column_csr_mass_feat4(d_data->csr_columns() + row_start, n_cols, j_global);
    if (local_idx >= 0) {
        atomicAdd(d_data->csr_values() + row_start + local_idx, mass_contribution);
    }
}

__global__ void calc_constraint_feat4_kernel(GPU_FEAT4_Data* d_data) {
    compute_constraint_data(d_data);
}

void GPU_FEAT4_Data::CalcDnDuPre() {
    int total_threads = n_elem * Quadrature::N_QP_T4_1;

    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    dn_du_pre_feat4_kernel<<<blocks, threads_per_block>>>(d_data);
    cudaDeviceSynchronize();
}

__global__ void calc_p_feat4_kernel(GPU_FEAT4_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_QP_T4_1;
    int qp_idx = thread_idx % Quadrature::N_QP_T4_1;

    if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T4_1)
        return;

    compute_p(elem_idx, qp_idx, d_data, nullptr, 0.0);
}

void GPU_FEAT4_Data::CalcP() {
    int threads = 128;
    int blocks = (n_elem * Quadrature::N_QP_T4_1 + threads - 1) / threads;
    calc_p_feat4_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

__global__ void compute_internal_force_feat4_kernel(GPU_FEAT4_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_NODE_T4_4;
    int node_local = thread_idx % Quadrature::N_NODE_T4_4;

    if (elem_idx >= d_data->gpu_n_elem() || node_local >= Quadrature::N_NODE_T4_4)
        return;

    compute_internal_force(elem_idx, node_local, d_data);
}

void GPU_FEAT4_Data::CalcInternalForce() {
    int threads = 128;
    int blocks = (n_elem * Quadrature::N_NODE_T4_4 + threads - 1) / threads;
    compute_internal_force_feat4_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

void GPU_FEAT4_Data::CalcConstraintData() {
    if (!is_constraints_setup) {
        MOPHI_ERROR("constraint is not set up");
        return;
    }
    if (n_constraint == 0) {
        return;
    }
    int total_threads = n_constraint / 3;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    calc_constraint_feat4_kernel<<<blocks, threads_per_block>>>(d_data);
    cudaDeviceSynchronize();
}

void GPU_FEAT4_Data::CalcMassMatrix() {
    if (!is_csr_setup) {
        BuildMassCSRPattern();
    }

    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_nnz > 0) {
        MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(h_nnz) * sizeof(Real)));
    }

    int total_threads = n_elem * 4 * 4;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    mass_matrix_qp_feat4_kernel<<<blocks, threads_per_block>>>(d_data);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

void GPU_FEAT4_Data::BuildMassCSRPattern() {
    if (is_csr_setup) {
        return;
    }

    const int total_keys = n_elem * Quadrature::N_NODE_T4_4 * Quadrature::N_NODE_T4_4;
    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys, static_cast<size_t>(total_keys) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_keys + threads - 1) / threads;
        build_mass_keys_feat4_kernel<<<blocks, threads>>>(d_data, d_keys);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<unsigned long long> keys_begin(d_keys);
    thrust::device_ptr<unsigned long long> keys_end = keys_begin + total_keys;
    thrust::sort(thrust::device, keys_begin, keys_end);
    thrust::device_ptr<unsigned long long> keys_unique_end = thrust::unique(thrust::device, keys_begin, keys_end);

    const int nnz = static_cast<int>(keys_unique_end - keys_begin);

    MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_offsets, static_cast<size_t>(n_coef + 1) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_nnz, sizeof(int)));

    int* d_row_counts = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_row_counts, static_cast<size_t>(n_coef) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_row_counts, 0, static_cast<size_t>(n_coef) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (nnz + threads - 1) / threads;
        decode_mass_keys_feat4_kernel<<<blocks, threads>>>(d_keys, nnz, d_csr_columns, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> row_counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_csr_offsets);
    thrust::exclusive_scan(thrust::device, row_counts_begin, row_counts_begin + n_coef, offsets_begin);

    {
        set_last_offset_feat4_kernel<<<1, 1>>>(d_csr_offsets, n_coef, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(Real)));

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    is_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));
}

namespace {
__global__ void build_constraint_j_csr_feat4_kernel(int n_constraint,
                                                    const int* fixed_nodes,
                                                    int* j_offsets,
                                                    int* j_columns,
                                                    Real* j_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_constraint) {
        return;
    }

    j_offsets[tid] = tid;

    int fixed_idx = tid / 3;
    int dof = tid % 3;
    int node = fixed_nodes[fixed_idx];
    j_columns[tid] = node * 3 + dof;
    j_values[tid] = 1.0;
}

__global__ void build_constraint_jt_row_counts_feat4_kernel(int n_constraint, const int* fixed_nodes, int* row_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_constraint) {
        return;
    }

    int fixed_idx = tid / 3;
    int dof = tid % 3;
    int node = fixed_nodes[fixed_idx];
    int row = node * 3 + dof;
    atomicAdd(&row_counts[row], 1);
}

__global__ void build_constraint_jt_fill_feat4_kernel(int n_constraint,
                                                      const int* fixed_nodes,
                                                      const int* offsets,
                                                      int* row_positions,
                                                      int* columns,
                                                      Real* values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_constraint) {
        return;
    }

    int fixed_idx = tid / 3;
    int dof = tid % 3;
    int node = fixed_nodes[fixed_idx];
    int row = node * 3 + dof;

    int pos = atomicAdd(&row_positions[row], 1);
    int out = offsets[row] + pos;

    columns[out] = tid;
    values[out] = 1.0;
}
}  // namespace

void GPU_FEAT4_Data::ConvertToCSR_ConstraintJac() {
    if (is_j_csr_setup) {
        return;
    }
    if (!is_constraints_setup || n_constraint == 0) {
        return;
    }

    const int nnz = n_constraint;

    MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_offsets, static_cast<size_t>(n_constraint + 1) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_nnz, sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_constraint + threads - 1) / threads;
        build_constraint_j_csr_feat4_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_j_csr_offsets,
                                                                 d_j_csr_columns, d_j_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
        set_last_offset_feat4_kernel<<<1, 1>>>(d_j_csr_offsets, n_constraint, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    is_j_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT4_Data::ConvertToCSR_ConstraintJacT() {
    if (is_cj_csr_setup) {
        return;
    }
    if (!is_constraints_setup || n_constraint == 0) {
        return;
    }

    const int num_rows = n_coef * 3;
    const int nnz = n_constraint;

    MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_offsets, static_cast<size_t>(num_rows + 1) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_nnz, sizeof(int)));

    int* d_row_counts = nullptr;
    int* d_row_positions = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_row_counts, static_cast<size_t>(num_rows) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMalloc(&d_row_positions, static_cast<size_t>(num_rows) * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemset(d_row_counts, 0, static_cast<size_t>(num_rows) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_constraint + threads - 1) / threads;
        build_constraint_jt_row_counts_feat4_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_cj_csr_offsets);
    thrust::exclusive_scan(thrust::device, counts_begin, counts_begin + num_rows, offsets_begin);

    set_last_offset_feat4_kernel<<<1, 1>>>(d_cj_csr_offsets, num_rows, nnz);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaMemset(d_row_positions, 0, static_cast<size_t>(num_rows) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_constraint + threads - 1) / threads;
        build_constraint_jt_fill_feat4_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_cj_csr_offsets,
                                                                   d_row_positions, d_cj_csr_columns, d_cj_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_row_positions));

    is_cj_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT4_Data::SetNodalFixed(const VectorXi& fixed_nodes_in) {
    n_constraint = fixed_nodes_in.size() * 3;

    da_constraint.resize(static_cast<size_t>(n_constraint));
    da_constraint.BindDevicePointer(&d_constraint);
    da_constraint.SetVal(Real(0));
    da_constraint.ToDevice();

    da_fixed_nodes.resize(static_cast<size_t>(fixed_nodes_in.size()));
    da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);
    std::copy(fixed_nodes_in.data(), fixed_nodes_in.data() + fixed_nodes_in.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    is_constraints_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT4_Data::UpdateNodalFixed(const VectorXi& fixed_nodes_in) {
    const int new_n_constraint = fixed_nodes_in.size() * 3;

    if (!is_constraints_setup || new_n_constraint != n_constraint) {
        if (is_constraints_setup) {
            da_constraint.free();
            da_fixed_nodes.free();
        }
        n_constraint = new_n_constraint;
        da_constraint.resize(static_cast<size_t>(n_constraint));
        da_constraint.BindDevicePointer(&d_constraint);
        da_fixed_nodes.resize(static_cast<size_t>(fixed_nodes_in.size()));
        da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);
        is_constraints_setup = true;
    }

    da_constraint.SetVal(Real(0));
    da_constraint.ToDevice();
    std::copy(fixed_nodes_in.data(), fixed_nodes_in.data() + fixed_nodes_in.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT4_Data::RetrieveInternalForceToCPU(VectorXR& internal_force) {
    internal_force.resize(n_coef * 3);
    da_f_int.ToHost();
    std::copy(da_f_int.host(), da_f_int.host() + n_coef * 3, internal_force.data());
}

void GPU_FEAT4_Data::RetrieveExternalForceToCPU(VectorXR& external_force) {
    external_force.resize(n_coef * 3);
    da_f_ext.ToHost();
    std::copy(da_f_ext.host(), da_f_ext.host() + n_coef * 3, external_force.data());
}

void GPU_FEAT4_Data::RetrievePositionToCPU(VectorXR& x12_out, VectorXR& y12_out, VectorXR& z12_out) {
    x12_out.resize(n_coef);
    y12_out.resize(n_coef);
    z12_out.resize(n_coef);
    da_h_x12.ToHost();
    da_h_y12.ToHost();
    da_h_z12.ToHost();
    std::copy(da_h_x12.host(), da_h_x12.host() + n_coef, x12_out.data());
    std::copy(da_h_y12.host(), da_h_y12.host() + n_coef, y12_out.data());
    std::copy(da_h_z12.host(), da_h_z12.host() + n_coef, z12_out.data());
}

void GPU_FEAT4_Data::RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F) {
    // Not implemented for TET4 (not needed for basic usage)
}

void GPU_FEAT4_Data::RetrieveDnDuPreToCPU(std::vector<std::vector<MatrixXR>>& dn_du_pre) {
    dn_du_pre.resize(n_elem, std::vector<MatrixXR>(Quadrature::N_QP_T4_1, MatrixXR(4, 3)));
    da_grad_N_ref.ToHost();
    for (int e = 0; e < n_elem; e++) {
        for (int qp = 0; qp < Quadrature::N_QP_T4_1; qp++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    int offset = (e * Quadrature::N_QP_T4_1 + qp) * 4 * 3 + i * 3 + j;
                    dn_du_pre[e][qp](i, j) = da_grad_N_ref.host()[offset];
                }
            }
        }
    }
}

void GPU_FEAT4_Data::RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ) {
    detJ.resize(n_elem, std::vector<Real>(Quadrature::N_QP_T4_1));
    da_detJ_ref.ToHost();
    for (int e = 0; e < n_elem; e++) {
        for (int qp = 0; qp < Quadrature::N_QP_T4_1; qp++) {
            detJ[e][qp] = da_detJ_ref.host()[e * Quadrature::N_QP_T4_1 + qp];
        }
    }
}

void GPU_FEAT4_Data::RetrieveConnectivityToCPU(MatrixXi& connectivity) {
    connectivity.resize(n_elem, 4);
    da_element_connectivity.ToHost();
    for (int e = 0; e < n_elem; e++) {
        for (int n = 0; n < 4; n++) {
            connectivity(e, n) = da_element_connectivity.host()[e * 4 + n];
        }
    }
}

void GPU_FEAT4_Data::RetrieveMassCSRToCPU(std::vector<int>& offsets,
                                          std::vector<int>& columns,
                                          std::vector<Real>& values) {
    if (!is_csr_setup) {
        return;
    }
    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

    offsets.resize(n_coef + 1);
    columns.resize(h_nnz);
    values.resize(h_nnz);

    MOPHI_GPU_CALL(cudaMemcpy(offsets.data(), d_csr_offsets, (n_coef + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaMemcpy(columns.data(), d_csr_columns, h_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaMemcpy(values.data(), d_csr_values, h_nnz * sizeof(Real), cudaMemcpyDeviceToHost));
}

}  // namespace feris
