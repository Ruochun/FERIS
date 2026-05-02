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
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    FEAT10Data.cu
 * Brief:   Implements GPU-side data management and element kernels for
 *          10-node tetrahedral FEAT10 elements. Handles allocation,
 *          initialization, mass and stiffness assembly, internal/external
 *          force evaluation, and optional constraint coupling.
 *==============================================================
 *==============================================================*/

#include "FEAT10Data.cuh"
#include "FEAT10DataFunc.cuh"

namespace cg = cooperative_groups;

namespace feris {

__global__ void build_mass_keys_feat10_kernel(GPU_FEAT10_Data* d_data, unsigned long long* d_keys) {
    const int total = d_data->gpu_n_elem() * Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int elem = tid / (Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10);
    const int item_local = tid % (Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10);
    const int i_local = item_local / Quadrature::N_NODE_T10_10;
    const int j_local = item_local % Quadrature::N_NODE_T10_10;

    const int i_global = d_data->element_connectivity()(elem, i_local);
    const int j_global = d_data->element_connectivity()(elem, j_local);

    const unsigned long long key = (static_cast<unsigned long long>(static_cast<unsigned int>(i_global)) << 32) |
                                   static_cast<unsigned long long>(static_cast<unsigned int>(j_global));
    d_keys[tid] = key;
}

__global__ void decode_mass_keys_kernel(const unsigned long long* d_keys,
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

__global__ void set_last_offset_kernel(int* d_offsets, int n_rows, int nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_offsets[n_rows] = nnz;
    }
}

__device__ __forceinline__ int binary_search_column_csr_mass(const int* cols, int n_cols, int target) {
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

__global__ void dn_du_pre_kernel(GPU_FEAT10_Data* d_data) {
    // Get global thread index
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate element index and quadrature point index
    int elem_idx = global_thread_idx / Quadrature::N_QP_T10_5;
    int qp_idx = global_thread_idx % Quadrature::N_QP_T10_5;

    // Bounds check
    if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5) {
        return;
    }

    // Get quadrature point coordinates (xi, eta, zeta)
    Real xi = d_data->tet5pt_x(qp_idx);    // L2 in Python code
    Real eta = d_data->tet5pt_y(qp_idx);   // L3 in Python code
    Real zeta = d_data->tet5pt_z(qp_idx);  // L4 in Python code

    // Compute barycentric coordinates
    Real L1 = 1.0 - xi - eta - zeta;
    Real L2 = xi;
    Real L3 = eta;
    Real L4 = zeta;
    Real L[4] = {L1, L2, L3, L4};

    // Derivatives of barycentric coordinates (dL matrix from Python)
    Real dL[4][3] = {
        {-1.0, -1.0, -1.0},  // dL1/dxi, dL1/deta, dL1/dzeta
        {1.0, 0.0, 0.0},     // dL2/dxi, dL2/deta, dL2/dzeta
        {0.0, 1.0, 0.0},     // dL3/dxi, dL3/deta, dL3/dzeta
        {0.0, 0.0, 1.0}      // dL4/dxi, dL4/deta, dL4/dzeta
    };

    // Compute shape function derivatives dN_dxi for all 10 nodes
    Real dN_dxi[10][3];

    // Corner nodes (0-3): dN_dxi[i, :] = (4*L[i]-1)*dL[i, :]
    for (int i = 0; i < 4; i++) {
        Real factor = 4.0 * L[i] - 1.0;
        for (int j = 0; j < 3; j++) {
            dN_dxi[i][j] = factor * dL[i][j];
        }
    }

    // Edge nodes (4-9): dN_dxi[k, :] = 4*(L[i]*dL[j, :] + L[j]*dL[i, :])
    // Edge connectivity: [(0,1), (1,2), (0,2), (0,3), (1,3), (2,3)]
    int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};

    for (int k = 0; k < 6; k++) {
        int i = edges[k][0];
        int j = edges[k][1];

        for (int d = 0; d < 3; d++) {
            dN_dxi[k + 4][d] = 4.0 * (L[i] * dL[j][d] + L[j] * dL[i][d]);
        }
    }

    // Get element node coordinates for this element
    Real X_elem[10][3];  // 10 nodes × 3 coordinates
    for (int node = 0; node < 10; node++) {
        int global_node_idx = d_data->element_connectivity()(elem_idx, node);
        X_elem[node][0] = d_data->x12()(global_node_idx);  // x coordinate
        X_elem[node][1] = d_data->y12()(global_node_idx);  // y coordinate
        X_elem[node][2] = d_data->z12()(global_node_idx);  // z coordinate
    }

    // Compute Jacobian matrix J = Σ(X_node ⊗ dN_dxi)
    Real J[3][3] = {{0.0}};  // Initialize to zero
    for (int a = 0; a < 10; a++) {
        for (int i = 0; i < 3; i++) {                    // Node coordinate components
            for (int j = 0; j < 3; j++) {                // Natural coordinate derivatives
                J[i][j] += X_elem[a][i] * dN_dxi[a][j];  // Outer product
            }
        }
    }

    // Compute determinant of J (3x3 matrix)
    Real detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    // Store the determinant in d_detJ_ref
    d_data->detJ_ref(elem_idx, qp_idx) = detJ;

    // Compute J^T (transpose)
    Real JT[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            JT[i][j] = J[j][i];
        }
    }

    // Solve JT * grad_N = dN_dxi for each shape function
    Real grad_N[10][3];
    for (int a = 0; a < 10; a++) {
        // Solve 3×3 system: JT * grad_N[a] = dN_dxi[a]
        // You'll need a 3×3 linear solver here (LU decomposition, Gaussian
        // elimination, etc.)
        solve_3x3_system(JT, dN_dxi[a], grad_N[a]);
    }

    // Store the PHYSICAL gradients in grad_N_ref
    for (int i = 0; i < 10; i++) {
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 0) = grad_N[i][0];  // ∂N_i/∂x
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 1) = grad_N[i][1];  // ∂N_i/∂y
        d_data->grad_N_ref(elem_idx, qp_idx)(i, 2) = grad_N[i][2];  // ∂N_i/∂z
    }
}

__global__ void mass_matrix_qp_kernel(GPU_FEAT10_Data* d_data) {
    int n_qp_per_elem = Quadrature::N_QP_T10_5;  // 5 quadrature points
    int thread_global = blockIdx.x * blockDim.x + threadIdx.x;

    // Decode: which element and which (i, j) node pair?
    int elem = thread_global / (10 * 10);  // 10 nodes per element
    int item_local = thread_global % (10 * 10);

    if (elem >= d_data->gpu_n_elem())
        return;

    // Decode item_local into (i_local, j_local) node indices
    int i_local = item_local / 10;  // Local node i (0-9)
    int j_local = item_local % 10;  // Local node j (0-9)

    // Get global node indices
    int i_global = d_data->element_connectivity()(elem, i_local);
    int j_global = d_data->element_connectivity()(elem, j_local);

    // Get material density
    Real rho = d_data->rho0();

    // Accumulator for this (i, j) pair across all QPs
    Real mass_contribution = 0.0;

    // Loop over all quadrature points
    for (int qp = 0; qp < n_qp_per_elem; qp++) {
        // Get quadrature point coordinates
        Real xi = d_data->tet5pt_x(qp);
        Real eta = d_data->tet5pt_y(qp);
        Real zeta = d_data->tet5pt_z(qp);
        Real wq = d_data->tet5pt_weights(qp);
        // printf("xi: %f, eta: %f, zeta: %f, wq: %f\n", xi, eta, zeta, wq);

        // Compute barycentric coordinates
        Real L1 = 1.0 - xi - eta - zeta;
        Real L2 = xi;
        Real L3 = eta;
        Real L4 = zeta;
        Real L[4] = {L1, L2, L3, L4};

        // Compute shape functions
        Real N[10];

        // Corner nodes (0-3)
        for (int k = 0; k < 4; k++) {
            N[k] = L[k] * (2.0 * L[k] - 1.0);
        }

        // Edge nodes (4-9)
        int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
        for (int k = 0; k < 6; k++) {
            int ii = edges[k][0];
            int jj = edges[k][1];
            N[k + 4] = 4.0 * L[ii] * L[jj];
        }

        // Get determinant (pre-computed)
        Real detJ = d_data->detJ_ref(elem, qp);

        // Accumulate: rho * N[i] * N[j] * detJ * wq
        mass_contribution += rho * N[i_local] * N[j_local] * detJ * wq;
    }

    const int row_start = d_data->csr_offsets()[i_global];
    const int row_end = d_data->csr_offsets()[i_global + 1];
    const int n_cols = row_end - row_start;
    const int local_idx = binary_search_column_csr_mass(d_data->csr_columns() + row_start, n_cols, j_global);
    if (local_idx >= 0) {
        atomicAdd(d_data->csr_values() + row_start + local_idx, mass_contribution);
    }
}

__global__ void calc_constraint_kernel(GPU_FEAT10_Data* d_data) {
    compute_constraint_data(d_data);
}

void GPU_FEAT10_Data::CalcDnDuPre() {
    int total_threads = n_elem * Quadrature::N_QP_T10_5;

    int threads_per_block = 128;  // or another suitable block size
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    dn_du_pre_kernel<<<blocks, threads_per_block>>>(d_data);
    cudaDeviceSynchronize();
}

__global__ void calc_p_kernel(GPU_FEAT10_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_QP_T10_5;
    int qp_idx = thread_idx % Quadrature::N_QP_T10_5;

    if (elem_idx >= d_data->gpu_n_elem() || qp_idx >= Quadrature::N_QP_T10_5)
        return;

    // No solver context for standalone CalcP: pass null v_guess (no viscous
    // contribution)
    compute_p(elem_idx, qp_idx, d_data, nullptr, 0.0);
}

void GPU_FEAT10_Data::CalcP() {
    int threads = 128;
    int blocks = (n_elem * Quadrature::N_QP_T10_5 + threads - 1) / threads;
    calc_p_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

__global__ void compute_internal_force_kernel(GPU_FEAT10_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_NODE_T10_10;    // 10 nodes per element
    int node_local = thread_idx % Quadrature::N_NODE_T10_10;  // Local node index (0-9)

    if (elem_idx >= d_data->gpu_n_elem() || node_local >= Quadrature::N_NODE_T10_10)
        return;

    compute_internal_force(elem_idx, node_local, d_data);
}

void GPU_FEAT10_Data::CalcInternalForce() {
    int threads = 128;
    int blocks = (n_elem * Quadrature::N_NODE_T10_10 + threads - 1) / threads;
    compute_internal_force_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

void GPU_FEAT10_Data::CalcConstraintData() {
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

    calc_constraint_kernel<<<blocks, threads_per_block>>>(d_data);
    cudaDeviceSynchronize();
}

void GPU_FEAT10_Data::CalcMassMatrix() {
    if (!is_csr_setup) {
        BuildMassCSRPattern();
    }

    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_nnz > 0) {
        MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(h_nnz) * sizeof(Real)));
    }

    // Launch: n_elem × 10 × 10 threads
    int total_threads = n_elem * 10 * 10;
    int threads_per_block = 128;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    mass_matrix_qp_kernel<<<blocks, threads_per_block>>>(d_data);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

void GPU_FEAT10_Data::BuildMassCSRPattern() {
    if (is_csr_setup) {
        return;
    }

    const int total_keys = n_elem * Quadrature::N_NODE_T10_10 * Quadrature::N_NODE_T10_10;
    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys, static_cast<size_t>(total_keys) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_keys + threads - 1) / threads;
        build_mass_keys_feat10_kernel<<<blocks, threads>>>(d_data, d_keys);
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
        decode_mass_keys_kernel<<<blocks, threads>>>(d_keys, nnz, d_csr_columns, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> row_counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_csr_offsets);
    thrust::exclusive_scan(thrust::device, row_counts_begin, row_counts_begin + n_coef, offsets_begin);

    {
        set_last_offset_kernel<<<1, 1>>>(d_csr_offsets, n_coef, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(Real)));

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    is_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
}

namespace {
__global__ void build_constraint_j_csr_kernel(int n_constraint,
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

__global__ void build_constraint_jt_row_counts_kernel(int n_constraint, const int* fixed_nodes, int* row_counts) {
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

__global__ void build_constraint_jt_fill_kernel(int n_constraint,
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

// This function converts the Constraint Jacobian matrix J to CSR format
// (Rows = Constraints, Cols = DOFs)
void GPU_FEAT10_Data::ConvertToCSR_ConstraintJac() {
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
        build_constraint_j_csr_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_j_csr_offsets,
                                                           d_j_csr_columns, d_j_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
        set_last_offset_kernel<<<1, 1>>>(d_j_csr_offsets, n_constraint, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    is_j_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
}

// This function converts the TRANSPOSE of the constraint Jacobian matrix to CSR
// format
void GPU_FEAT10_Data::ConvertToCSR_ConstraintJacT() {
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
        build_constraint_jt_row_counts_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_cj_csr_offsets);
    thrust::exclusive_scan(thrust::device, counts_begin, counts_begin + num_rows, offsets_begin);

    set_last_offset_kernel<<<1, 1>>>(d_cj_csr_offsets, num_rows, nnz);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaMemset(d_row_positions, 0, static_cast<size_t>(num_rows) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_constraint + threads - 1) / threads;
        build_constraint_jt_fill_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_cj_csr_offsets,
                                                             d_row_positions, d_cj_csr_columns, d_cj_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_row_positions));

    MOPHI_GPU_CALL(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    is_cj_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT10_Data::RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ) {
    da_detJ_ref.ToHost();
    const Real* host_ptr = da_detJ_ref.host();
    detJ.resize(n_elem);
    for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
        detJ[elem_idx].assign(host_ptr + elem_idx * Quadrature::N_QP_T10_5,
                              host_ptr + (elem_idx + 1) * Quadrature::N_QP_T10_5);
    }
}

void GPU_FEAT10_Data::RetrieveDnDuPreToCPU(std::vector<std::vector<MatrixXR>>& dn_du_pre) {
    da_grad_N_ref.ToHost();
    const Real* host_ptr = da_grad_N_ref.host();
    dn_du_pre.resize(n_elem);
    for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
        dn_du_pre[elem_idx].resize(Quadrature::N_QP_T10_5);
        for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
            int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 10 * 3;
            // Map is a no-copy host view; assignment to MatrixXR deep-copies the data.
            dn_du_pre[elem_idx][qp_idx] = Map<const MatrixXR>(host_ptr + offset, 10, 3).eval();
        }
    }
}

void GPU_FEAT10_Data::RetrieveMassCSRToCPU(std::vector<int>& offsets,
                                           std::vector<int>& columns,
                                           std::vector<Real>& values) {
    offsets.assign(static_cast<size_t>(n_coef) + 1, 0);
    columns.clear();
    values.clear();

    if (!is_csr_setup) {
        return;
    }

    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

    columns.resize(static_cast<size_t>(h_nnz));
    values.resize(static_cast<size_t>(h_nnz));

    MOPHI_GPU_CALL(cudaMemcpy(offsets.data(), d_csr_offsets, static_cast<size_t>(n_coef + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(
        cudaMemcpy(columns.data(), d_csr_columns, static_cast<size_t>(h_nnz) * sizeof(int), cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(
        cudaMemcpy(values.data(), d_csr_values, static_cast<size_t>(h_nnz) * sizeof(Real), cudaMemcpyDeviceToHost));
}

void GPU_FEAT10_Data::RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F) {
    da_P.ToHost();
    const Real* host_ptr = da_P.host();
    p_from_F.resize(n_elem);
    for (int elem_idx = 0; elem_idx < n_elem; elem_idx++) {
        p_from_F[elem_idx].resize(Quadrature::N_QP_T10_5);
        for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T10_5; qp_idx++) {
            int offset = (elem_idx * Quadrature::N_QP_T10_5 + qp_idx) * 3 * 3;
            // Map is a no-copy host view; assignment to MatrixXR deep-copies the data.
            p_from_F[elem_idx][qp_idx] = Map<const MatrixXR>(host_ptr + offset, 3, 3).eval();
        }
    }
}

void GPU_FEAT10_Data::RetrieveInternalForceToCPU(VectorXR& internal_force) {
    int total_dofs = 3 * n_coef;
    da_f_int.ToHost();
    // Map is a no-copy host view; the assignment to VectorXR deep-copies the data.
    internal_force = Map<VectorXR>(da_f_int.host(), total_dofs).eval();
}

void GPU_FEAT10_Data::RetrieveExternalForceToCPU(VectorXR& external_force) {
    int total_dofs = 3 * n_coef;
    da_f_ext.ToHost();
    external_force = Map<VectorXR>(da_f_ext.host(), total_dofs).eval();
}

void GPU_FEAT10_Data::RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12) {
    int total_nodes = n_coef;
    da_h_x12.ToHost();
    da_h_y12.ToHost();
    da_h_z12.ToHost();
    x12 = Map<VectorXR>(da_h_x12.host(), total_nodes).eval();
    y12 = Map<VectorXR>(da_h_y12.host(), total_nodes).eval();
    z12 = Map<VectorXR>(da_h_z12.host(), total_nodes).eval();
}

void GPU_FEAT10_Data::SetNodalFixed(const VectorXi& fixed_nodes) {
    if (is_constraints_setup) {
        MOPHI_ERROR("GPU_FEAT10_Data CONSTRAINT is already set up.");
        return;
    }

    n_constraint = fixed_nodes.size() * 3;

    da_constraint.resize(n_constraint);
    da_constraint.BindDevicePointer(&d_constraint);
    da_fixed_nodes.resize(fixed_nodes.size());
    da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);

    da_constraint.SetVal(Real(0));
    da_constraint.MakeReadyDevice();
    std::copy(fixed_nodes.data(), fixed_nodes.data() + fixed_nodes.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    is_constraints_setup = true;
    if (d_data) {
        MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
    }
}

void GPU_FEAT10_Data::UpdateNodalFixed(const VectorXi& fixed_nodes) {
    int new_n_constraint = fixed_nodes.size() * 3;

    // If constraints not set up yet, just call SetNodalFixed
    if (!is_constraints_setup) {
        SetNodalFixed(fixed_nodes);
        return;
    }

    // If number of constraints changed, resize DualArrays (realloc device if needed)
    if (new_n_constraint != n_constraint) {
        // Free old CSR buffers if they exist
        if (is_cj_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_cj_nnz));
            d_cj_csr_offsets = nullptr;
            d_cj_csr_columns = nullptr;
            d_cj_csr_values = nullptr;
            d_cj_nnz = nullptr;
            is_cj_csr_setup = false;
        }

        if (is_j_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_j_nnz));
            d_j_csr_offsets = nullptr;
            d_j_csr_columns = nullptr;
            d_j_csr_values = nullptr;
            d_j_nnz = nullptr;
            is_j_csr_setup = false;
        }

        n_constraint = new_n_constraint;

        // DualArray handles reallocation; BindDevicePointer was already called
        // in SetNodalFixed so d_constraint and d_fixed_nodes are auto-updated.
        da_constraint.resize(n_constraint);
        da_fixed_nodes.resize(fixed_nodes.size());
    }

    // Clear constraint data and upload new fixed nodes
    da_constraint.SetVal(Real(0));
    da_constraint.MakeReadyDevice();
    std::copy(fixed_nodes.data(), fixed_nodes.data() + fixed_nodes.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    // Invalidate Jacobian CSR caches: fixed nodes may have changed even if the
    // constraint count stayed the same.
    if (is_cj_csr_setup) {
        MOPHI_GPU_CALL(cudaFree(d_cj_csr_offsets));
        MOPHI_GPU_CALL(cudaFree(d_cj_csr_columns));
        MOPHI_GPU_CALL(cudaFree(d_cj_csr_values));
        MOPHI_GPU_CALL(cudaFree(d_cj_nnz));
        d_cj_csr_offsets = nullptr;
        d_cj_csr_columns = nullptr;
        d_cj_csr_values = nullptr;
        d_cj_nnz = nullptr;
        is_cj_csr_setup = false;
    }

    if (is_j_csr_setup) {
        MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
        MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
        MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
        MOPHI_GPU_CALL(cudaFree(d_j_nnz));
        d_j_csr_offsets = nullptr;
        d_j_csr_columns = nullptr;
        d_j_csr_values = nullptr;
        d_j_nnz = nullptr;
        is_j_csr_setup = false;
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT10_Data), cudaMemcpyHostToDevice));
}

void GPU_FEAT10_Data::RetrieveConnectivityToCPU(MatrixXi& connectivity) {
    da_element_connectivity.ToHost();
    // Map is a no-copy host view; assignment to MatrixXi deep-copies the data.
    connectivity = Map<MatrixXi>(da_element_connectivity.host(), n_elem, Quadrature::N_NODE_T10_10).eval();
}

void GPU_FEAT10_Data::WriteOutputVTK(const std::string& filename) {
    VectorXR x12, y12, z12;
    this->RetrievePositionToCPU(x12, y12, z12);

    // Retrieve connectivity
    MatrixXi connectivity;
    this->RetrieveConnectivityToCPU(connectivity);

    std::ofstream out(filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "T10 mesh output\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    // Write points
    out << "POINTS " << x12.size() << " float\n";
    for (int i = 0; i < x12.size(); ++i) {
        out << x12(i) << " " << y12(i) << " " << z12(i) << "\n";
    }

    // Write cells (elements)
    out << "CELLS " << connectivity.rows() << " " << connectivity.rows() * 11 << "\n";
    for (int i = 0; i < connectivity.rows(); ++i) {
        out << "10 ";
        for (int j = 0; j < 10; ++j)
            out << connectivity(i, j) << " ";
        out << "\n";
    }

    // Write cell types (24 = VTK_QUADRATIC_TETRA)
    out << "CELL_TYPES " << connectivity.rows() << "\n";
    for (int i = 0; i < connectivity.rows(); ++i)
        out << "24\n";

    out.close();
}

}  // namespace feris
