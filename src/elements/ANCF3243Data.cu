#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <iomanip>
#include <vector>

/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3243Data.cu
 * Brief:   Implements GPU-side data management and element-level kernels for
 *          ANCF 3243 beam elements. Provides initialization, mass and
 *          stiffness assembly, internal force evaluation, and constraint
 *          handling routines used by the synchronized solvers.
 *==============================================================
 *==============================================================*/

#include "ANCF3243Data.cuh"
#include "ANCF3243DataFunc.cuh"
namespace cg = cooperative_groups;

namespace feris {

__device__ __forceinline__ int binary_search_column_csr_mass_3243(const int* cols, int n_cols, int target) {
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

__global__ void build_mass_keys_3243_kernel(GPU_ANCF3243_Data* d_data, unsigned long long* d_keys) {
    const int total = d_data->gpu_n_beam() * Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int elem = tid / (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);
    const int item_local = tid % (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);
    const int i_local = item_local / Quadrature::N_SHAPE_3243;
    const int j_local = item_local % Quadrature::N_SHAPE_3243;

    const int node_i_local = (i_local < 4) ? 0 : 1;
    const int node_j_local = (j_local < 4) ? 0 : 1;
    const int dof_i_local = i_local % 4;
    const int dof_j_local = j_local % 4;

    const int node_i_global = d_data->element_node(elem, node_i_local);
    const int node_j_global = d_data->element_node(elem, node_j_local);

    const int i_global = node_i_global * 4 + dof_i_local;
    const int j_global = node_j_global * 4 + dof_j_local;

    const unsigned long long key = (static_cast<unsigned long long>(static_cast<unsigned int>(i_global)) << 32) |
                                   static_cast<unsigned long long>(static_cast<unsigned int>(j_global));
    d_keys[tid] = key;
}

__global__ void decode_mass_keys_3243_kernel(const unsigned long long* d_keys,
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

__global__ void set_last_offset_3243_kernel(int* d_offsets, int n_rows, int nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_offsets[n_rows] = nnz;
    }
}

// Precompute reference Jacobian determinant and physical shape gradients
// (per element, per quadrature point).
__global__ void precompute_reference_kernel(GPU_ANCF3243_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_qp = Quadrature::N_TOTAL_QP_3_2_2;
    const int total = d_data->gpu_n_beam() * n_qp;
    if (tid >= total) {
        return;
    }

    const int elem_idx = tid / n_qp;
    const int qp_idx = tid - elem_idx * n_qp;

    const int ixi = qp_idx / (Quadrature::N_QP_2 * Quadrature::N_QP_2);
    const int ieta = (qp_idx / Quadrature::N_QP_2) % Quadrature::N_QP_2;
    const int izeta = qp_idx % Quadrature::N_QP_2;

    const Real xi = d_data->gauss_xi()(ixi);
    const Real eta = d_data->gauss_eta()(ieta);
    const Real zeta = d_data->gauss_zeta()(izeta);

    const Real L = d_data->L(elem_idx);
    const Real W = d_data->W(elem_idx);
    const Real H = d_data->H(elem_idx);

    // Build ∂b/∂xi, ∂b/∂eta, ∂b/∂zeta directly in normalized coordinates.
    const Real db_dxi[Quadrature::N_SHAPE_3243] = {
        0.0, L / 2, 0.0, 0.0, (L * W / 4) * eta, (L * H / 4) * zeta, (L * L / 2) * xi, (3 * L * L * L / 8) * xi * xi};
    const Real db_deta[Quadrature::N_SHAPE_3243] = {0.0, 0.0, W / 2, 0.0, (L * W / 4) * xi, 0.0, 0.0, 0.0};
    const Real db_dzeta[Quadrature::N_SHAPE_3243] = {0.0, 0.0, 0.0, H / 2, 0.0, (L * H / 4) * xi, 0.0, 0.0};

    Real ds_dxi[Quadrature::N_SHAPE_3243];
    Real ds_deta[Quadrature::N_SHAPE_3243];
    Real ds_dzeta[Quadrature::N_SHAPE_3243];
    ancf3243_mat_vec_mul8(d_data->B_inv(elem_idx), db_dxi, ds_dxi);
    ancf3243_mat_vec_mul8(d_data->B_inv(elem_idx), db_deta, ds_deta);
    ancf3243_mat_vec_mul8(d_data->B_inv(elem_idx), db_dzeta, ds_dzeta);

    Real x_local_arr[Quadrature::N_SHAPE_3243];
    Real y_local_arr[Quadrature::N_SHAPE_3243];
    Real z_local_arr[Quadrature::N_SHAPE_3243];
    d_data->x_cur_jac_elem(elem_idx, x_local_arr);
    d_data->y_cur_jac_elem(elem_idx, y_local_arr);
    d_data->z_cur_jac_elem(elem_idx, z_local_arr);
    Map<VectorXR> x_loc(x_local_arr, Quadrature::N_SHAPE_3243);
    Map<VectorXR> y_loc(y_local_arr, Quadrature::N_SHAPE_3243);
    Map<VectorXR> z_loc(z_local_arr, Quadrature::N_SHAPE_3243);

    // Reference Jacobian J = sum_a X_a ⊗ ∂s_a/∂xi.
    Real J[3][3] = {{0.0}};
#pragma unroll
    for (int a = 0; a < Quadrature::N_SHAPE_3243; ++a) {
        J[0][0] += x_loc(a) * ds_dxi[a];
        J[1][0] += y_loc(a) * ds_dxi[a];
        J[2][0] += z_loc(a) * ds_dxi[a];

        J[0][1] += x_loc(a) * ds_deta[a];
        J[1][1] += y_loc(a) * ds_deta[a];
        J[2][1] += z_loc(a) * ds_deta[a];

        J[0][2] += x_loc(a) * ds_dzeta[a];
        J[1][2] += y_loc(a) * ds_dzeta[a];
        J[2][2] += z_loc(a) * ds_dzeta[a];
    }

    const Real detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                      J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                      J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    d_data->detJ_ref(elem_idx, qp_idx) = detJ;

    // Solve J^T * grad_s = [ds_dxi, ds_deta, ds_dzeta] for each shape function.
    Real JT[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            JT[i][j] = J[j][i];
        }
    }

#pragma unroll
    for (int a = 0; a < Quadrature::N_SHAPE_3243; ++a) {
        Real rhs[3] = {ds_dxi[a], ds_deta[a], ds_dzeta[a]};
        Real grad[3] = {0.0, 0.0, 0.0};
        ancf3243_solve_3x3_system(JT, rhs, grad);
        d_data->grad_N_ref(elem_idx, qp_idx)(a, 0) = grad[0];
        d_data->grad_N_ref(elem_idx, qp_idx)(a, 1) = grad[1];
        d_data->grad_N_ref(elem_idx, qp_idx)(a, 2) = grad[2];
    }
}

__global__ void mass_matrix_qp_kernel(GPU_ANCF3243_Data* d_data) {
    int n_qp_per_elem = Quadrature::N_QP_6 * Quadrature::N_QP_2 * Quadrature::N_QP_2;
    int thread_global = blockIdx.x * blockDim.x + threadIdx.x;
    int elem = thread_global / (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);
    int item_local = thread_global % (Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243);

    if (elem >= d_data->gpu_n_beam())
        return;

    // local indices
    int i_local = item_local / Quadrature::N_SHAPE_3243;  // 0–7 (row)
    int j_local = item_local % Quadrature::N_SHAPE_3243;  // 0–7 (col)

    // ===============================================================
    // Local→global DOF mapping using element_connectivity
    // Each element has 2 nodes × 4 DOFs = 8 shape functions
    // local index 0–3 → node 0, dof 0–3
    // local index 4–7 → node 1, dof 0–3
    // ===============================================================

    const int node_i_local = (i_local < 4) ? 0 : 1;
    const int node_j_local = (j_local < 4) ? 0 : 1;
    const int dof_i_local = i_local % 4;
    const int dof_j_local = j_local % 4;

    const int node_i_global = d_data->element_node(elem, node_i_local);
    const int node_j_global = d_data->element_node(elem, node_j_local);

    // Flattened global DOF indices (4 DOFs per node)
    const int i_global = node_i_global * 4 + dof_i_local;
    const int j_global = node_j_global * 4 + dof_j_local;

    // Precompute constants outside the loop
    Real rho = d_data->rho0();
    Real L = d_data->L(elem);
    Real W = d_data->W(elem);
    Real H = d_data->H(elem);

    Real x_local_arr[Quadrature::N_SHAPE_3243];
    Real y_local_arr[Quadrature::N_SHAPE_3243];
    Real z_local_arr[Quadrature::N_SHAPE_3243];
    d_data->x_cur_jac_elem(elem, x_local_arr);
    d_data->y_cur_jac_elem(elem, y_local_arr);
    d_data->z_cur_jac_elem(elem, z_local_arr);
    Map<VectorXR> x_loc(x_local_arr, Quadrature::N_SHAPE_3243);
    Map<VectorXR> y_loc(y_local_arr, Quadrature::N_SHAPE_3243);
    Map<VectorXR> z_loc(z_local_arr, Quadrature::N_SHAPE_3243);

    for (int qp_local = 0; qp_local < n_qp_per_elem; qp_local++) {
        // Decode qp_local into (ixi, ieta, izeta)
        int ixi = qp_local / (Quadrature::N_QP_2 * Quadrature::N_QP_2);
        int ieta = (qp_local / Quadrature::N_QP_2) % Quadrature::N_QP_2;
        int izeta = qp_local % Quadrature::N_QP_2;

        Real xi = d_data->gauss_xi_m()(ixi);
        Real eta = d_data->gauss_eta()(ieta);
        Real zeta = d_data->gauss_zeta()(izeta);
        Real weight = d_data->weight_xi_m()(ixi) * d_data->weight_eta()(ieta) * d_data->weight_zeta()(izeta);

        // Compute shape function vector (8 entries = 2 nodes × 4 DOFs)
        Real b[8];
        ancf3243_b_vec_xi(xi, eta, zeta, L, W, H, b);

        // Compute s = B_inv * b
        Real s[8];
        ancf3243_mat_vec_mul8(d_data->B_inv(elem), b, s);

        // Compute Jacobian determinant
        Real J[9];
        ancf3243_calc_det_J_xi(xi, eta, zeta, d_data->B_inv(elem), x_loc, y_loc, z_loc, L, W, H, J);
        Real detJ = ancf3243_det3x3(J);

        // Accumulate contribution into CSR
        const Real mass_contrib = rho * s[i_local] * s[j_local] * weight * detJ;
        const int row_start = d_data->csr_offsets()[i_global];
        const int row_end = d_data->csr_offsets()[i_global + 1];
        const int n_cols = row_end - row_start;
        const int local_idx = binary_search_column_csr_mass_3243(d_data->csr_columns() + row_start, n_cols, j_global);
        if (local_idx >= 0) {
            atomicAdd(d_data->csr_values() + row_start + local_idx, mass_contrib);
        }
    }
}

__global__ void calc_p_kernel(GPU_ANCF3243_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_TOTAL_QP_3_2_2;
    int qp_idx = thread_idx % Quadrature::N_TOTAL_QP_3_2_2;

    if (elem_idx >= d_data->gpu_n_beam() || qp_idx >= Quadrature::N_TOTAL_QP_3_2_2)
        return;
    // Call compute_p with nullptr velocity and zero dt for standalone usage
    compute_p(elem_idx, qp_idx, d_data, nullptr, 0.0);
}

void GPU_ANCF3243_Data::CalcP() {
    int threads = 128;
    int blocks = (n_beam * Quadrature::N_TOTAL_QP_3_2_2 + threads - 1) / threads;
    calc_p_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::CalcDsDuPre() {
    if (!is_setup) {
        MOPHI_ERROR("GPU_ANCF3243_Data::CalcDsDuPre: call Setup() first.");
        return;
    }
    if (is_reference_precomputed) {
        return;
    }
    const int threads = 128;
    const int total = n_beam * Quadrature::N_TOTAL_QP_3_2_2;
    const int blocks = (total + threads - 1) / threads;
    precompute_reference_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
    is_reference_precomputed = true;
}

void GPU_ANCF3243_Data::PrintDsDuPre() {
    const int n_qp = Quadrature::N_TOTAL_QP_3_2_2;
    const int mat_stride = Quadrature::N_SHAPE_3243 * 3;
    const int total_size = n_beam * n_qp * mat_stride;

    std::vector<Real> h_grad(static_cast<size_t>(total_size));
    std::vector<Real> h_detJ(static_cast<size_t>(n_beam * n_qp));

    MOPHI_GPU_CALL(cudaMemcpy(h_grad.data(), d_grad_N_ref, static_cast<size_t>(total_size) * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaMemcpy(h_detJ.data(), d_detJ_ref, static_cast<size_t>(n_beam * n_qp) * sizeof(Real),
                              cudaMemcpyDeviceToHost));

    for (int e = 0; e < n_beam; ++e) {
        for (int qp = 0; qp < n_qp; ++qp) {
            std::cout << "\n=== Elem " << e << " Quadrature Point " << qp << " detJ_ref=" << h_detJ[e * n_qp + qp]
                      << " ===" << std::endl;

            Real* qp_data = h_grad.data() + (e * n_qp + qp) * mat_stride;
            Map<MatrixXR> grad_matrix(qp_data, Quadrature::N_SHAPE_3243, 3);
            std::cout << "        dN/dx       dN/dy       dN/dz" << std::endl;
            for (int i = 0; i < Quadrature::N_SHAPE_3243; ++i) {
                std::cout << "Shape " << i << ": ";
                for (int j = 0; j < 3; ++j) {
                    std::cout << std::setw(10) << std::fixed << std::setprecision(6) << grad_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void GPU_ANCF3243_Data::RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ) {
    const int n_qp = Quadrature::N_TOTAL_QP_3_2_2;
    detJ.assign(static_cast<size_t>(n_beam), std::vector<Real>(n_qp));
    std::vector<Real> flat(static_cast<size_t>(n_beam * n_qp));
    MOPHI_GPU_CALL(
        cudaMemcpy(flat.data(), d_detJ_ref, static_cast<size_t>(n_beam * n_qp) * sizeof(Real), cudaMemcpyDeviceToHost));
    for (int e = 0; e < n_beam; ++e) {
        for (int qp = 0; qp < n_qp; ++qp) {
            detJ[e][qp] = flat[e * n_qp + qp];
        }
    }
}

void GPU_ANCF3243_Data::CalcMassMatrix() {
    if (!is_csr_setup) {
        BuildMassCSRPattern();
    }

    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_nnz > 0) {
        MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(h_nnz) * sizeof(Real)));
    }

    // Mass terms computation
    const int N_OUT = n_beam * Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243;

    // Launch kernel
    int threads = 128;
    int blocks = (N_OUT + threads - 1) / threads;
    mass_matrix_qp_kernel<<<blocks, threads>>>(d_data);

    cudaDeviceSynchronize();
}

void GPU_ANCF3243_Data::BuildMassCSRPattern() {
    if (is_csr_setup) {
        return;
    }

    const int total_keys = n_beam * Quadrature::N_SHAPE_3243 * Quadrature::N_SHAPE_3243;
    unsigned long long* d_keys = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_keys, static_cast<size_t>(total_keys) * sizeof(unsigned long long)));

    {
        constexpr int threads = 256;
        const int blocks = (total_keys + threads - 1) / threads;
        build_mass_keys_3243_kernel<<<blocks, threads>>>(d_data, d_keys);
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
        decode_mass_keys_3243_kernel<<<blocks, threads>>>(d_keys, nnz, d_csr_columns, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> row_counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_csr_offsets);
    thrust::exclusive_scan(thrust::device, row_counts_begin, row_counts_begin + n_coef, offsets_begin);

    {
        set_last_offset_3243_kernel<<<1, 1>>>(d_csr_offsets, n_coef, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemset(d_csr_values, 0, static_cast<size_t>(nnz) * sizeof(Real)));

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_keys));

    is_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));
}

namespace {
__global__ void build_constraint_j_csr_3243_kernel(int n_constraint,
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

__global__ void build_constraint_jt_row_counts_3243_kernel(int n_constraint, const int* fixed_nodes, int* row_counts) {
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

__global__ void build_constraint_jt_fill_3243_kernel(int n_constraint,
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

// This function converts the TRANSPOSE of the constraint Jacobian matrix to CSR
// format
void GPU_ANCF3243_Data::ConvertToCSR_ConstraintJacT() {
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
        build_constraint_jt_row_counts_3243_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_row_counts);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    thrust::device_ptr<int> counts_begin(d_row_counts);
    thrust::device_ptr<int> offsets_begin(d_cj_csr_offsets);
    thrust::exclusive_scan(thrust::device, counts_begin, counts_begin + num_rows, offsets_begin);

    set_last_offset_3243_kernel<<<1, 1>>>(d_cj_csr_offsets, num_rows, nnz);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    MOPHI_GPU_CALL(cudaMemset(d_row_positions, 0, static_cast<size_t>(num_rows) * sizeof(int)));

    {
        constexpr int threads = 256;
        const int blocks = (n_constraint + threads - 1) / threads;
        build_constraint_jt_fill_3243_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_cj_csr_offsets,
                                                                  d_row_positions, d_cj_csr_columns, d_cj_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaFree(d_row_counts));
    MOPHI_GPU_CALL(cudaFree(d_row_positions));

    MOPHI_GPU_CALL(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    is_cj_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));
}

// This function converts the Constraint Jacobian matrix J to CSR format
// (Rows = Constraints, Cols = DOFs)
void GPU_ANCF3243_Data::ConvertToCSR_ConstraintJac() {
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
        build_constraint_j_csr_3243_kernel<<<blocks, threads>>>(n_constraint, d_fixed_nodes, d_j_csr_offsets,
                                                                d_j_csr_columns, d_j_csr_values);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
        set_last_offset_3243_kernel<<<1, 1>>>(d_j_csr_offsets, n_constraint, nnz);
        MOPHI_GPU_CALL(cudaDeviceSynchronize());
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    is_j_csr_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3243_Data), cudaMemcpyHostToDevice));
}

void GPU_ANCF3243_Data::RetrieveConnectivityToCPU(MatrixXi& connectivity) {
    // `d_element_connectivity` is stored as a flat row-major (n_beam × 2) array.
    // Use DualArray ToHost + Map into a row-major staging matrix, then assign.
    da_element_connectivity.ToHost();
    // Map is a no-copy host view; the assignment to MatrixXi deep-copies the data.
    Map<Matrix<int, DynamicMatrix, 2, RowMajorMatrix>> connectivity_row_major(da_element_connectivity.host(), n_beam,
                                                                              2);
    connectivity = connectivity_row_major.eval();
}

void GPU_ANCF3243_Data::RetrieveMassCSRToCPU(std::vector<int>& offsets,
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

void GPU_ANCF3243_Data::RetrieveInternalForceToCPU(VectorXR& internal_force) {
    int expected_size = n_coef * 3;
    da_f_int.ToHost();
    // Map is a no-copy host view; the assignment to VectorXR deep-copies the data.
    internal_force = Map<VectorXR>(da_f_int.host(), expected_size).eval();
}

void GPU_ANCF3243_Data::RetrieveConstraintJacobianCSRToCPU(std::vector<int>& offsets,
                                                           std::vector<int>& columns,
                                                           std::vector<Real>& values) {
    offsets.assign(static_cast<size_t>(n_constraint) + 1, 0);
    columns.clear();
    values.clear();

    if (!is_constraints_setup || n_constraint == 0) {
        return;
    }
    if (!is_j_csr_setup) {
        return;
    }

    int h_nnz = 0;
    MOPHI_GPU_CALL(cudaMemcpy(&h_nnz, d_j_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_nnz < 0) {
        return;
    }

    columns.resize(static_cast<size_t>(h_nnz));
    values.resize(static_cast<size_t>(h_nnz));

    MOPHI_GPU_CALL(cudaMemcpy(offsets.data(), d_j_csr_offsets, (static_cast<size_t>(n_constraint) + 1) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    if (h_nnz > 0) {
        MOPHI_GPU_CALL(cudaMemcpy(columns.data(), d_j_csr_columns, static_cast<size_t>(h_nnz) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        MOPHI_GPU_CALL(cudaMemcpy(values.data(), d_j_csr_values, static_cast<size_t>(h_nnz) * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
    }
}

void GPU_ANCF3243_Data::RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>& deformation_gradient) {
    da_F.ToHost();
    const Real* host_ptr = da_F.host();
    deformation_gradient.resize(n_beam);
    for (int i = 0; i < n_beam; i++) {
        deformation_gradient[i].resize(Quadrature::N_TOTAL_QP_3_2_2);
        for (int j = 0; j < Quadrature::N_TOTAL_QP_3_2_2; j++) {
            int offset = i * Quadrature::N_TOTAL_QP_3_2_2 * 9 + j * 9;
            // Map is a no-copy host view; assignment to MatrixXR deep-copies the data.
            deformation_gradient[i][j] = Map<const MatrixXR>(host_ptr + offset, 3, 3).eval();
        }
    }
}

void GPU_ANCF3243_Data::RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F) {
    da_P.ToHost();
    const Real* host_ptr = da_P.host();
    p_from_F.resize(n_beam);
    for (int i = 0; i < n_beam; i++) {
        p_from_F[i].resize(Quadrature::N_TOTAL_QP_3_2_2);
        for (int j = 0; j < Quadrature::N_TOTAL_QP_3_2_2; j++) {
            int offset = i * Quadrature::N_TOTAL_QP_3_2_2 * 9 + j * 9;
            // Map is a no-copy host view; assignment to MatrixXR deep-copies the data.
            p_from_F[i][j] = Map<const MatrixXR>(host_ptr + offset, 3, 3).eval();
        }
    }
}

void GPU_ANCF3243_Data::RetrieveConstraintDataToCPU(VectorXR& constraint) {
    int expected_size = n_constraint;
    da_constraint.ToHost();
    // Map is a no-copy host view; the assignment to VectorXR deep-copies the data.
    constraint = Map<VectorXR>(da_constraint.host(), expected_size).eval();
}

void GPU_ANCF3243_Data::RetrieveConstraintJacobianToCPU(MatrixXR& constraint_jac) {
    constraint_jac.resize(n_constraint, n_coef * 3);
    constraint_jac.setZero();

    if (!is_constraints_setup || n_constraint == 0) {
        return;
    }

    std::vector<int> h_fixed(static_cast<size_t>(n_constraint / 3), 0);
    da_fixed_nodes.ToHost();
    std::copy(da_fixed_nodes.host(), da_fixed_nodes.host() + n_constraint / 3, h_fixed.begin());

    for (int i = 0; i < n_constraint / 3; ++i) {
        int node = h_fixed[static_cast<size_t>(i)];
        constraint_jac(i * 3 + 0, node * 3 + 0) = 1.0;
        constraint_jac(i * 3 + 1, node * 3 + 1) = 1.0;
        constraint_jac(i * 3 + 2, node * 3 + 2) = 1.0;
    }
}

void GPU_ANCF3243_Data::RetrievePositionToCPU(VectorXR& x_cur, VectorXR& y_cur, VectorXR& z_cur) {
    int expected_size = n_coef;
    da_x_cur.ToHost();
    da_y_cur.ToHost();
    da_z_cur.ToHost();
    // Map is a no-copy host view; the assignment to VectorXR deep-copies the data.
    x_cur = Map<VectorXR>(da_x_cur.host(), expected_size).eval();
    y_cur = Map<VectorXR>(da_y_cur.host(), expected_size).eval();
    z_cur = Map<VectorXR>(da_z_cur.host(), expected_size).eval();
}

__global__ void compute_internal_force_kernel(GPU_ANCF3243_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = thread_idx / Quadrature::N_SHAPE_3243;
    int node_idx = thread_idx % Quadrature::N_SHAPE_3243;

    if (elem_idx >= d_data->gpu_n_beam() || node_idx >= Quadrature::N_SHAPE_3243)
        return;

    compute_internal_force(elem_idx, node_idx, d_data);
}

void GPU_ANCF3243_Data::CalcInternalForce() {
    int threads = 128;
    int blocks = (n_beam * Quadrature::N_SHAPE_3243 + threads - 1) / threads;
    compute_internal_force_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

__global__ void compute_constraint_data_kernel(GPU_ANCF3243_Data* d_data) {
    compute_constraint_data(d_data);
}

void GPU_ANCF3243_Data::CalcConstraintData() {
    if (!is_constraints_setup) {
        MOPHI_ERROR("constraint is not set up");
        return;
    }

    int threads = 128;
    int n_work = 0;
    if (constraint_mode == kConstraintFixedCoefficients) {
        n_work = n_constraint / 3;
    } else if (constraint_mode == kConstraintLinearCSR) {
        n_work = n_constraint;
    } else {
        n_work = 0;
    }
    if (n_work <= 0) {
        return;
    }
    int blocks = (n_work + threads - 1) / threads;
    compute_constraint_data_kernel<<<blocks, threads>>>(d_data);
    cudaDeviceSynchronize();
}

}  // namespace feris
