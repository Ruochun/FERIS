#pragma once
/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3443DataFunc.cuh
 * Brief:   Contains CUDA device functions and kernels for ANCF 3443 shell
 *          elements, including shape function evaluation, quadrature loops,
 *          stress computation, and element stiffness/internal force assembly
 *          into global structures.
 *==============================================================
 *==============================================================*/

#include <cmath>
#include "../types.h"

#include "../materials/MooneyRivlin.cuh"
#include "../materials/SVK.cuh"
#include "ANCF3443Data.cuh"

namespace feris {

// forward-declare solver type (pointer-only used here)
struct SyncedNewtonSolver;

__device__ __forceinline__ void compute_p(int, int, GPU_ANCF3443_Data*, const Real*, Real);
__device__ __forceinline__ void compute_internal_force(int, int, GPU_ANCF3443_Data*);
__device__ __forceinline__ void compute_constraint_data(GPU_ANCF3443_Data*);

// Solve 3x3 linear system: A * x = b (Gaussian elimination with pivoting).
__device__ __forceinline__ void ancf3443_solve_3x3_system(Real A[3][3], Real b[3], Real x[3]) {
    constexpr Real kEpsPivot = 1e-14;
    Real aug[3][4];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            aug[i][j] = A[i][j];
        }
        aug[i][3] = b[i];
    }

    for (int k = 0; k < 3; k++) {
        int pivot_row = k;
        Real max_val = fabs(aug[k][k]);
        for (int i = k + 1; i < 3; i++) {
            Real v = fabs(aug[i][k]);
            if (v > max_val) {
                max_val = v;
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                Real tmp = aug[k][j];
                aug[k][j] = aug[pivot_row][j];
                aug[pivot_row][j] = tmp;
            }
        }

        if (fabs(aug[k][k]) < kEpsPivot) {
            x[0] = x[1] = x[2] = 0.0;
            return;
        }

        for (int i = k + 1; i < 3; i++) {
            Real factor = aug[i][k] / aug[k][k];
            for (int j = k; j < 4; j++) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Extra guard: even with pivoting, near-singular systems can produce tiny
    // diagonal entries after elimination (e.g., due to floating-point effects).
    if (fabs(aug[2][2]) < kEpsPivot || fabs(aug[1][1]) < kEpsPivot || fabs(aug[0][0]) < kEpsPivot) {
        x[0] = x[1] = x[2] = 0.0;
        return;
    }

    x[2] = aug[2][3] / aug[2][2];
    x[1] = (aug[1][3] - aug[1][2] * x[2]) / aug[1][1];
    x[0] = (aug[0][3] - aug[0][2] * x[2] - aug[0][1] * x[1]) / aug[0][0];
}

__device__ __forceinline__ void ancf3443_mat_vec_mul(Map<MatrixXR> A, const Real* x, Real* out) {
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i) {
        out[i] = 0.0;
#pragma unroll
        for (int j = 0; j < Quadrature::N_SHAPE_3443; ++j) {
            out[i] += A(i, j) * x[j];
        }
    }
}

// Device function to compute determinant of 3x3 matrix
__device__ __forceinline__ Real ancf3443_det3x3(const Real* J) {
    return J[0] * (J[4] * J[8] - J[5] * J[7]) - J[1] * (J[3] * J[8] - J[5] * J[6]) + J[2] * (J[3] * J[7] - J[4] * J[6]);
}

__device__ __forceinline__ void ancf3443_b_vec(Real u, Real v, Real w, Real* out) {
    out[0] = 1.0;
    out[1] = u;
    out[2] = v;
    out[3] = w;
    out[4] = u * v;
    out[5] = u * w;
    out[6] = v * w;
    out[7] = u * v * w;
    out[8] = u * u;
    out[9] = v * v;
    out[10] = (u * u) * v;
    out[11] = u * (v * v);
    out[12] = u * u * u;
    out[13] = v * v * v;
    out[14] = (u * u * u) * v;
    out[15] = u * (v * v * v);
}

__device__ __forceinline__ void ancf3443_b_vec_xi(Real xi, Real eta, Real zeta, Real L, Real W, Real H, Real* out) {
    Real u = L * xi / 2.0;
    Real v = W * eta / 2.0;
    Real w = H * zeta / 2.0;
    ancf3443_b_vec(u, v, w, out);
}

__device__ __forceinline__ void ancf3443_db_dxi(Real xi, Real eta, Real zeta, Real L, Real W, Real H, Real* out) {
    // Map to physical coordinates
    Real u = 0.5 * L * xi;
    Real v = 0.5 * W * eta;
    Real w = 0.5 * H * zeta;

    // db_du as in your Python code
    out[0] = 0.0;
    out[1] = 1.0;
    out[2] = 0.0;
    out[3] = 0.0;
    out[4] = v;
    out[5] = w;
    out[6] = 0.0;
    out[7] = v * w;
    out[8] = 2.0 * u;
    out[9] = 0.0;
    out[10] = 2.0 * u * v;
    out[11] = v * v;
    out[12] = 3.0 * u * u;
    out[13] = 0.0;
    out[14] = 3.0 * u * u * v;
    out[15] = v * v * v;

// Chain rule: db/dxi = 0.5 * L * db/du
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
        out[i] *= 0.5 * L;
}

__device__ __forceinline__ void ancf3443_db_deta(Real xi, Real eta, Real zeta, Real L, Real W, Real H, Real* out) {
    Real u = 0.5 * L * xi;
    Real v = 0.5 * W * eta;
    Real w = 0.5 * H * zeta;

    // db_dv as in your Python code
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 1.0;
    out[3] = 0.0;
    out[4] = u;
    out[5] = 0.0;
    out[6] = w;
    out[7] = u * w;
    out[8] = 0.0;
    out[9] = 2.0 * v;
    out[10] = u * u;
    out[11] = 2.0 * u * v;
    out[12] = 0.0;
    out[13] = 3.0 * v * v;
    out[14] = u * u * u;
    out[15] = 3.0 * u * v * v;

// Chain rule: db/deta = 0.5 * W * db/dv
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
        out[i] *= 0.5 * W;
}

__device__ __forceinline__ void ancf3443_db_dzeta(Real xi, Real eta, Real zeta, Real L, Real W, Real H, Real* out) {
    Real u = 0.5 * L * xi;
    Real v = 0.5 * W * eta;

    // db_dw as in your Python code
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 0.0;
    out[3] = 1.0;
    out[4] = 0.0;
    out[5] = u;
    out[6] = v;
    out[7] = u * v;
    out[8] = 0.0;
    out[9] = 0.0;
    out[10] = 0.0;
    out[11] = 0.0;
    out[12] = 0.0;
    out[13] = 0.0;
    out[14] = 0.0;
    out[15] = 0.0;

    // Chain rule: db/dzeta = 0.5 * H * db/dw
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i)
        out[i] *= 0.5 * H;
}

// Device function for Jacobian determinant in normalized coordinates
__device__ __forceinline__ void ancf3443_calc_det_J_xi(Real xi,
                                                       Real eta,
                                                       Real zeta,
                                                       Map<MatrixXR> B_inv,
                                                       Map<VectorXR> x_cur_jac,
                                                       Map<VectorXR> y_cur_jac,
                                                       Map<VectorXR> z_cur_jac,
                                                       Real L,
                                                       Real W,
                                                       Real H,
                                                       Real* J_out) {
    Real db_dxi[Quadrature::N_SHAPE_3443], db_deta[Quadrature::N_SHAPE_3443], db_dzeta[Quadrature::N_SHAPE_3443];
    ancf3443_db_dxi(xi, eta, zeta, L, W, H, db_dxi);
    ancf3443_db_deta(xi, eta, zeta, L, W, H, db_deta);
    ancf3443_db_dzeta(xi, eta, zeta, L, W, H, db_dzeta);

    Real ds_dxi[Quadrature::N_SHAPE_3443], ds_deta[Quadrature::N_SHAPE_3443], ds_dzeta[Quadrature::N_SHAPE_3443];
    ancf3443_mat_vec_mul(B_inv, db_dxi, ds_dxi);
    ancf3443_mat_vec_mul(B_inv, db_deta, ds_deta);
    ancf3443_mat_vec_mul(B_inv, db_dzeta, ds_dzeta);

#pragma unroll
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_out[i * 3 + j] = 0.0;

#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; ++i) {
        J_out[0 * 3 + 0] += x_cur_jac(i) * ds_dxi[i];
        J_out[1 * 3 + 0] += y_cur_jac(i) * ds_dxi[i];
        J_out[2 * 3 + 0] += z_cur_jac(i) * ds_dxi[i];

        J_out[0 * 3 + 1] += x_cur_jac(i) * ds_deta[i];
        J_out[1 * 3 + 1] += y_cur_jac(i) * ds_deta[i];
        J_out[2 * 3 + 1] += z_cur_jac(i) * ds_deta[i];

        J_out[0 * 3 + 2] += x_cur_jac(i) * ds_dzeta[i];
        J_out[1 * 3 + 2] += y_cur_jac(i) * ds_dzeta[i];
        J_out[2 * 3 + 2] += z_cur_jac(i) * ds_dzeta[i];
    }
}

__device__ __forceinline__ void compute_p(int elem_idx,
                                          int qp_idx,
                                          GPU_ANCF3443_Data* d_data,
                                          const Real* __restrict__ v_guess,
                                          Real dt) {
// --- Compute C = F^T * F ---

// Initialize F to zero
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            d_data->F(elem_idx, qp_idx)(i, j) = 0.0;
        }
    }

    // Get local nodal coordinates for this element
    Real x_local_arr[Quadrature::N_SHAPE_3443], y_local_arr[Quadrature::N_SHAPE_3443],
        z_local_arr[Quadrature::N_SHAPE_3443];
    d_data->x_cur_elem(elem_idx, x_local_arr);
    d_data->y_cur_elem(elem_idx, y_local_arr);
    d_data->z_cur_elem(elem_idx, z_local_arr);
    Map<VectorXR> x_local(x_local_arr, Quadrature::N_SHAPE_3443);
    Map<VectorXR> y_local(y_local_arr, Quadrature::N_SHAPE_3443);
    Map<VectorXR> z_local(z_local_arr, Quadrature::N_SHAPE_3443);

    Real e[Quadrature::N_SHAPE_3443][3];
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
        e[i][0] = x_local[i];  // x position
        e[i][1] = y_local[i];  // y position
        e[i][2] = z_local[i];  // z position
        // If you need derivatives, use x_local[i*4+1], etc.
    }
// Compute F = sum_i e_i ⊗ ∇s_i
// F is 3x3 matrix stored in row-major order
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {  // Loop over nodes
        // Get gradient of shape function i (∇s_i) - this needs proper indexing
        // Assuming ds_du_pre is laid out as [qp_total][16][3]
        // You'll need to provide the correct qp_idx for the current quadrature
        // point
        Real grad_s_i[3];
        grad_s_i[0] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 0);
        grad_s_i[1] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 1);
        grad_s_i[2] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 2);

// Compute outer product: e_i ⊗ ∇s_i and add to F
#pragma unroll
        for (int row = 0; row < 3; row++) {  // e_i components
#pragma unroll
            for (int col = 0; col < 3; col++) {  // ∇s_i components
                d_data->F(elem_idx, qp_idx)(row, col) += e[i][row] * grad_s_i[col];
            }
        }
    }

    // Precompute Ft (transpose of F) for reuse in viscous and elastic terms
    Real Ft[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            Ft[i][j] = d_data->F(elem_idx, qp_idx)(j, i);

    Real eta = d_data->eta_damp();
    Real lambda_d = d_data->lambda_damp();
    const bool do_damp = (v_guess != nullptr) && (eta != 0.0 || lambda_d != 0.0);
    Real P_vis[3][3] = {{0.0}};

    if (do_damp) {
        // Compute Fdot = sum_i v_i ⊗ ∇s_i
        Real Fdot[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
            Real v_i[3] = {0.0, 0.0, 0.0};
            int node_local = i / 4;
            int dof_local = i % 4;
            int node_global = d_data->element_connectivity()(elem_idx, node_local);
            int coef_idx = node_global * 4 + dof_local;
            v_i[0] = v_guess[coef_idx * 3 + 0];
            v_i[1] = v_guess[coef_idx * 3 + 1];
            v_i[2] = v_guess[coef_idx * 3 + 2];
#pragma unroll
            for (int row = 0; row < 3; row++) {
#pragma unroll
                for (int col = 0; col < 3; col++) {
                    // grad_s not in scope here; read from precomputed ds_du_pre
                    Real grad_si_col = d_data->grad_N_ref(elem_idx, qp_idx)(i, col);
                    Fdot[row][col] += v_i[row] * grad_si_col;
                }
            }
        }

        // Edot = 0.5*(Fdot^T * F + F^T * Fdot)
        Real FdotT_F[3][3] = {{0.0}};
        Real Ft_Fdot[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    FdotT_F[i][j] += Fdot[k][i] * d_data->F(elem_idx, qp_idx)(k, j);
                    Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
                }
            }
        }
        Real Edot[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);

        Real trEdot = Edot[0][0] + Edot[1][1] + Edot[2][2];
        Real S_vis[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                S_vis[i][j] = 2.0 * eta * Edot[i][j] + lambda_d * trEdot * (i == j ? 1.0 : 0.0);
            }
        }
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    P_vis[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * S_vis[k][j];
                }
            }
        }
// store Fdot and P_vis
#pragma unroll
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
                d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
            }
    } else {
#pragma unroll
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                d_data->Fdot(elem_idx, qp_idx)(i, j) = 0.0;
                d_data->P_vis(elem_idx, qp_idx)(i, j) = 0.0;
            }
    }

    Real FtF[3][3] = {0.0};

#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int k = 0; k < 3; ++k)
                FtF[i][j] += d_data->F(elem_idx, qp_idx)(k, i) * d_data->F(elem_idx, qp_idx)(k, j);

    // --- trace(F^T F) ---
    Real tr_FtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

    // Note: Ft already computed earlier (transpose of F)

    // 2. Compute G = F * Ft

    Real G[3][3] = {0};  // G = F * F^T
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                G[i][j] += d_data->F(elem_idx, qp_idx)(i, k) * Ft[k][j];
            }

    // 3. Compute FFF = G * F = (F * Ft) * F
    Real FFF[3][3] = {0};
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                FFF[i][j] += G[i][k] * d_data->F(elem_idx, qp_idx)(k, j);
            }

    // --- Compute P ---
    Real F_local[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            F_local[i][j] = d_data->F(elem_idx, qp_idx)(i, j);

    Real P_el[3][3];
    if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
        mr_compute_P(F_local, d_data->mu10(), d_data->mu01(), d_data->kappa(), P_el);
    } else {
        svk_compute_P_from_trFtF_and_FFtF(F_local, tr_FtF, FFF, d_data->lambda(), d_data->mu(), P_el);
    }

#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            d_data->P(elem_idx, qp_idx)(i, j) = P_el[i][j] + P_vis[i][j];
}

__device__ __forceinline__ void compute_internal_force(int elem_idx, int node_idx, GPU_ANCF3443_Data* d_data) {
    Real f_i[3] = {0};
    // int node_base = d_data->offset_start()(elem_idx);
    int global_node_idx = d_data->element_connectivity()(elem_idx, node_idx / 4) * 4 + (node_idx % 4);

#pragma unroll
    for (int qp_idx = 0; qp_idx < Quadrature::N_TOTAL_QP_4_4_3; qp_idx++) {
        Real grad_s[3];
        grad_s[0] = d_data->grad_N_ref(elem_idx, qp_idx)(node_idx, 0);
        grad_s[1] = d_data->grad_N_ref(elem_idx, qp_idx)(node_idx, 1);
        grad_s[2] = d_data->grad_N_ref(elem_idx, qp_idx)(node_idx, 2);

        Real scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                     d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                     d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
        const Real dV = d_data->detJ_ref(elem_idx, qp_idx) * scale;
#pragma unroll
        for (int r = 0; r < 3; ++r) {
#pragma unroll
            for (int c = 0; c < 3; ++c) {
                f_i[r] += (d_data->P(elem_idx, qp_idx)(r, c) * grad_s[c]) * dV;
            }
        }
    }

#pragma unroll
    for (int d = 0; d < 3; ++d) {
        atomicAdd(&d_data->f_int(global_node_idx)(d), f_i[d]);
    }
}

__device__ __forceinline__ void compute_constraint_data(GPU_ANCF3443_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int mode = d_data->constraint_mode_device();

    if (mode == GPU_ANCF3443_Data::kConstraintFixedCoefficients) {
        if (thread_idx < d_data->gpu_n_constraint() / 3) {
            d_data->constraint()[thread_idx * 3 + 0] = d_data->x_cur()(d_data->fixed_nodes()[thread_idx]) -
                                                       d_data->x_cur_jac()(d_data->fixed_nodes()[thread_idx]);
            d_data->constraint()[thread_idx * 3 + 1] = d_data->y_cur()(d_data->fixed_nodes()[thread_idx]) -
                                                       d_data->y_cur_jac()(d_data->fixed_nodes()[thread_idx]);
            d_data->constraint()[thread_idx * 3 + 2] = d_data->z_cur()(d_data->fixed_nodes()[thread_idx]) -
                                                       d_data->z_cur_jac()(d_data->fixed_nodes()[thread_idx]);
        }
        return;
    }

    if (mode == GPU_ANCF3443_Data::kConstraintLinearCSR) {
        const int row = thread_idx;
        if (row >= d_data->gpu_n_constraint()) {
            return;
        }

        const int start = d_data->j_csr_offsets()[row];
        const int end = d_data->j_csr_offsets()[row + 1];
        Real sum = 0.0;
        for (int idx = start; idx < end; ++idx) {
            const int col = d_data->j_csr_columns()[idx];
            const Real w = d_data->j_csr_values()[idx];
            const int coef = col / 3;
            const int comp = col - coef * 3;
            Real v = 0.0;
            if (comp == 0)
                v = d_data->x_cur()(coef);
            if (comp == 1)
                v = d_data->y_cur()(coef);
            if (comp == 2)
                v = d_data->z_cur()(coef);
            sum += w * v;
        }
        d_data->constraint()(row) = sum - d_data->constraint_rhs()[row];
        return;
    }
}

__device__ __forceinline__ void vbd_accumulate_residual_and_hessian_diag(int elem_idx,
                                                                         int qp_idx,
                                                                         int local_node,
                                                                         GPU_ANCF3443_Data* d_data,
                                                                         Real dt,
                                                                         Real& r0,
                                                                         Real& r1,
                                                                         Real& r2,
                                                                         Real& h00,
                                                                         Real& h01,
                                                                         Real& h02,
                                                                         Real& h10,
                                                                         Real& h11,
                                                                         Real& h12,
                                                                         Real& h20,
                                                                         Real& h21,
                                                                         Real& h22) {
    const Real ha0 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 0);
    const Real ha1 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 1);
    const Real ha2 = d_data->grad_N_ref(elem_idx, qp_idx)(local_node, 2);
    const Real scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                       d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                       d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
    const Real dV = d_data->detJ_ref(elem_idx, qp_idx) * scale;

    const Real P00 = d_data->P(elem_idx, qp_idx)(0, 0);
    const Real P01 = d_data->P(elem_idx, qp_idx)(0, 1);
    const Real P02 = d_data->P(elem_idx, qp_idx)(0, 2);
    const Real P10 = d_data->P(elem_idx, qp_idx)(1, 0);
    const Real P11 = d_data->P(elem_idx, qp_idx)(1, 1);
    const Real P12 = d_data->P(elem_idx, qp_idx)(1, 2);
    const Real P20 = d_data->P(elem_idx, qp_idx)(2, 0);
    const Real P21 = d_data->P(elem_idx, qp_idx)(2, 1);
    const Real P22 = d_data->P(elem_idx, qp_idx)(2, 2);

    r0 += (P00 * ha0 + P01 * ha1 + P02 * ha2) * dV;
    r1 += (P10 * ha0 + P11 * ha1 + P12 * ha2) * dV;
    r2 += (P20 * ha0 + P21 * ha1 + P22 * ha2) * dV;

    const Real F00 = d_data->F(elem_idx, qp_idx)(0, 0);
    const Real F01 = d_data->F(elem_idx, qp_idx)(0, 1);
    const Real F02 = d_data->F(elem_idx, qp_idx)(0, 2);
    const Real F10 = d_data->F(elem_idx, qp_idx)(1, 0);
    const Real F11 = d_data->F(elem_idx, qp_idx)(1, 1);
    const Real F12 = d_data->F(elem_idx, qp_idx)(1, 2);
    const Real F20 = d_data->F(elem_idx, qp_idx)(2, 0);
    const Real F21 = d_data->F(elem_idx, qp_idx)(2, 1);
    const Real F22 = d_data->F(elem_idx, qp_idx)(2, 2);

    const Real trFtF =
        F00 * F00 + F01 * F01 + F02 * F02 + F10 * F10 + F11 * F11 + F12 * F12 + F20 * F20 + F21 * F21 + F22 * F22;
    const Real trE = 0.5 * (trFtF - 3.0);

    const Real FFT00 = F00 * F00 + F01 * F01 + F02 * F02;
    const Real FFT01 = F00 * F10 + F01 * F11 + F02 * F12;
    const Real FFT02 = F00 * F20 + F01 * F21 + F02 * F22;
    const Real FFT10 = FFT01;
    const Real FFT11 = F10 * F10 + F11 * F11 + F12 * F12;
    const Real FFT12 = F10 * F20 + F11 * F21 + F12 * F22;
    const Real FFT20 = FFT02;
    const Real FFT21 = FFT12;
    const Real FFT22 = F20 * F20 + F21 * F21 + F22 * F22;

    const Real Fh0 = F00 * ha0 + F01 * ha1 + F02 * ha2;
    const Real Fh1 = F10 * ha0 + F11 * ha1 + F12 * ha2;
    const Real Fh2 = F20 * ha0 + F21 * ha1 + F22 * ha2;

    const Real hij = ha0 * ha0 + ha1 * ha1 + ha2 * ha2;
    const Real Fh_dot_Fh = Fh0 * Fh0 + Fh1 * Fh1 + Fh2 * Fh2;
    const Real weight_k = dt * dV;

    Real Kblock[3][3];
    if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
        Real F_local[3][3] = {{F00, F01, F02}, {F10, F11, F12}, {F20, F21, F22}};
        Real A_mr[3][3][3][3];
        mr_compute_tangent_tensor(F_local, d_data->mu10(), d_data->mu01(), d_data->kappa(), A_mr);
#pragma unroll
        for (int d = 0; d < 3; ++d) {
#pragma unroll
            for (int e = 0; e < 3; ++e) {
                Real sum = 0.0;
#pragma unroll
                for (int J = 0; J < 3; ++J) {
#pragma unroll
                    for (int L = 0; L < 3; ++L) {
                        const Real giJ = (J == 0 ? ha0 : (J == 1 ? ha1 : ha2));
                        const Real giL = (L == 0 ? ha0 : (L == 1 ? ha1 : ha2));
                        sum += A_mr[d][J][e][L] * giJ * giL;
                    }
                }
                Kblock[d][e] = sum * weight_k;
            }
        }
    } else {
        const Real Fh_vec[3] = {Fh0, Fh1, Fh2};
        const Real FFT[3][3] = {{FFT00, FFT01, FFT02}, {FFT10, FFT11, FFT12}, {FFT20, FFT21, FFT22}};
        svk_compute_tangent_block(Fh_vec, Fh_vec, hij, trE, Fh_dot_Fh, FFT, d_data->lambda(), d_data->mu(), weight_k,
                                  Kblock);
    }

    h00 += Kblock[0][0];
    h01 += Kblock[0][1];
    h02 += Kblock[0][2];
    h10 += Kblock[1][0];
    h11 += Kblock[1][1];
    h12 += Kblock[1][2];
    h20 += Kblock[2][0];
    h21 += Kblock[2][1];
    h22 += Kblock[2][2];
}

__device__ __forceinline__ void clear_internal_force(GPU_ANCF3443_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < d_data->n_coef * 3) {
        d_data->f_int()[thread_idx] = 0.0;
    }
}

// --- CSR-version Hessian assembly for ANCF3443 ---
static __device__ __forceinline__ int binary_search_column_csr_3443(const int* cols, int n_cols, int target) {
    int left = 0, right = n_cols - 1;
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

template <typename ElementType>
__device__ __forceinline__ void compute_hessian_assemble_csr(ElementType* d_data,
                                                             SyncedNewtonSolver* d_solver,
                                                             int elem_idx,
                                                             int qp_idx,
                                                             int* d_csr_row_offsets,
                                                             int* d_csr_col_indices,
                                                             Real* d_csr_values,
                                                             Real h);

// Explicit specialization for GPU_ANCF3443_Data
template <>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_ANCF3443_Data>(GPU_ANCF3443_Data* d_data,
                                                                                SyncedNewtonSolver* d_solver,
                                                                                int elem_idx,
                                                                                int qp_idx,
                                                                                int* d_csr_row_offsets,
                                                                                int* d_csr_col_indices,
                                                                                Real* d_csr_values,
                                                                                Real h) {
    // Build local element K_elem (48x48) similarly to FEAT10/ANCF3243.
    // Use existing local nodal extraction provided in compute_hessian_assemble
    // (if you need the full constitutive formulas I can copy them in).

    Real e[Quadrature::N_SHAPE_3443][3];
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
        int node_local = i / 4;  // which physical node (0..3)
        int dof_local = i % 4;   // which local DOF
        int node_global = d_data->element_connectivity()(elem_idx, node_local);
        int coef_idx = node_global * 4 + dof_local;
        e[i][0] = d_data->x_cur()(coef_idx);
        e[i][1] = d_data->y_cur()(coef_idx);
        e[i][2] = d_data->z_cur()(coef_idx);
    }

    Real grad_s[Quadrature::N_SHAPE_3443][3];
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
        grad_s[i][0] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 0);
        grad_s[i][1] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 1);
        grad_s[i][2] = d_data->grad_N_ref(elem_idx, qp_idx)(i, 2);
    }

    Real F[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
#pragma unroll
        for (int row = 0; row < 3; row++) {
#pragma unroll
            for (int col = 0; col < 3; col++) {
                F[row][col] += e[i][row] * grad_s[i][col];
            }
        }
    }

    Real C[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
#pragma unroll
            for (int k = 0; k < 3; k++) {
                C[i][j] += F[k][i] * F[k][j];
            }
        }
    }

    Real trC = C[0][0] + C[1][1] + C[2][2];
    Real trE = 0.5 * (trC - 3.0);

    Real FFT[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
#pragma unroll
            for (int k = 0; k < 3; k++) {
                FFT[i][j] += F[i][k] * F[j][k];
            }
        }
    }

    Real Fh[Quadrature::N_SHAPE_3443][3];
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
#pragma unroll
        for (int row = 0; row < 3; row++) {
            Fh[i][row] = 0.0;
#pragma unroll
            for (int col = 0; col < 3; col++) {
                Fh[i][row] += F[row][col] * grad_s[i][col];
            }
        }
    }

    Real lambda = d_data->lambda();
    Real mu = d_data->mu();
    Real scale = d_data->weight_xi()(qp_idx / (Quadrature::N_QP_4 * Quadrature::N_QP_3)) *
                 d_data->weight_eta()((qp_idx / Quadrature::N_QP_3) % Quadrature::N_QP_4) *
                 d_data->weight_zeta()(qp_idx % Quadrature::N_QP_3);
    Real dV = d_data->detJ_ref(elem_idx, qp_idx) * scale;

    const bool use_mr = (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN);
    Real A_mr[3][3][3][3];
    if (use_mr) {
        mr_compute_tangent_tensor(F, d_data->mu10(), d_data->mu01(), d_data->kappa(), A_mr);
    }

    // Local K_elem (48 x 48)
    const int Nloc = Quadrature::N_SHAPE_3443 * 3;  // 16*3 = 48
    // Because large stack arrays can be heavy, but follow pattern from other
    // DataFunc Implement K_elem as a local array
    Real K_elem[48][48];
#pragma unroll
    for (int ii = 0; ii < Nloc; ii++)
        for (int jj = 0; jj < Nloc; jj++)
            K_elem[ii][jj] = 0.0;

// Build K_elem using the same constitutive building block pattern
#pragma unroll
    for (int i = 0; i < Quadrature::N_SHAPE_3443; i++) {
#pragma unroll
        for (int j = 0; j < Quadrature::N_SHAPE_3443; j++) {
            Real h_ij = grad_s[j][0] * grad_s[i][0] + grad_s[j][1] * grad_s[i][1] + grad_s[j][2] * grad_s[i][2];
            Real Fhj_dot_Fhi = Fh[j][0] * Fh[i][0] + Fh[j][1] * Fh[i][1] + Fh[j][2] * Fh[i][2];

            Real Kblock[3][3];
            if (use_mr) {
#pragma unroll
                for (int d = 0; d < 3; d++) {
#pragma unroll
                    for (int eidx = 0; eidx < 3; eidx++) {
                        Real sum = 0.0;
#pragma unroll
                        for (int J = 0; J < 3; J++) {
#pragma unroll
                            for (int L = 0; L < 3; L++) {
                                sum += A_mr[d][J][eidx][L] * grad_s[i][J] * grad_s[j][L];
                            }
                        }
                        Kblock[d][eidx] = sum * dV;
                    }
                }
            } else {
                svk_compute_tangent_block(Fh[i], Fh[j], h_ij, trE, Fhj_dot_Fhi, FFT, lambda, mu, dV, Kblock);
            }

            for (int d = 0; d < 3; d++) {
                for (int eidx = 0; eidx < 3; eidx++) {
                    int row = 3 * i + d;
                    int col = 3 * j + eidx;
                    K_elem[row][col] = Kblock[d][eidx];
                }
            }
        }
    }

    // Scatter to CSR
    for (int local_row = 0; local_row < Quadrature::N_SHAPE_3443; local_row++) {
        int node_local_row = local_row / 4;
        int dof_local_row = local_row % 4;
        int node_global_row = d_data->element_connectivity()(elem_idx, node_local_row);
        int coef_idx_row = node_global_row * 4 + dof_local_row;

        for (int r_dof = 0; r_dof < 3; r_dof++) {
            int global_row = 3 * coef_idx_row + r_dof;
            int local_row_index = 3 * local_row + r_dof;

            int row_begin = d_csr_row_offsets[global_row];
            int row_len = d_csr_row_offsets[global_row + 1] - row_begin;

            for (int local_col = 0; local_col < Quadrature::N_SHAPE_3443; local_col++) {
                int node_local_col = local_col / 4;
                int dof_local_col = local_col % 4;
                int node_global_col = d_data->element_connectivity()(elem_idx, node_local_col);
                int coef_idx_col = node_global_col * 4 + dof_local_col;

                for (int c_dof = 0; c_dof < 3; c_dof++) {
                    int global_col = 3 * coef_idx_col + c_dof;
                    int local_col_index = 3 * local_col + c_dof;

                    int pos = binary_search_column_csr_3443(&d_csr_col_indices[row_begin], row_len, global_col);
                    if (pos >= 0) {
                        atomicAdd(&d_csr_values[row_begin + pos], h * K_elem[local_row_index][local_col_index]);
                    }
                }
            }
        }
    }

    Real eta_d_3443 = d_data->eta_damp();
    Real lambda_d_3443 = d_data->lambda_damp();
    if (eta_d_3443 == 0.0 && lambda_d_3443 == 0.0) {
        return;
    }

    // --- Viscous tangent (Kelvin-Voigt) assembly: C_elem (Nloc*3 x Nloc*3) ---
    const int Nloc3443 = Quadrature::N_SHAPE_3443;  // typically 16
    const int Ndof3443 = Nloc3443 * 3;              // 48
    Real C_elem_3443[48][48];
#pragma unroll
    for (int ii = 0; ii < Ndof3443; ii++)
        for (int jj = 0; jj < Ndof3443; jj++)
            C_elem_3443[ii][jj] = 0.0;

#pragma unroll
    for (int a = 0; a < Nloc3443; a++) {
        Real h_a0 = grad_s[a][0];
        Real h_a1 = grad_s[a][1];
        Real h_a2 = grad_s[a][2];
        Real Fh_a0 = Fh[a][0];
        Real Fh_a1 = Fh[a][1];
        Real Fh_a2 = Fh[a][2];
#pragma unroll
        for (int b = 0; b < Nloc3443; b++) {
            Real h_b0 = grad_s[b][0];
            Real h_b1 = grad_s[b][1];
            Real h_b2 = grad_s[b][2];
            Real Fh_b0 = Fh[b][0];
            Real Fh_b1 = Fh[b][1];
            Real Fh_b2 = Fh[b][2];

            Real hdot = h_a0 * h_b0 + h_a1 * h_b1 + h_a2 * h_b2;

            // build 3x3 block for (a,b)
            Real B00 =
                (eta_d_3443 * (Fh_b0 * Fh_a0) + eta_d_3443 * FFT[0][0] * hdot + lambda_d_3443 * (Fh_a0 * Fh_b0)) * dV;
            Real B01 =
                (eta_d_3443 * (Fh_b0 * Fh_a1) + eta_d_3443 * FFT[0][1] * hdot + lambda_d_3443 * (Fh_a0 * Fh_b1)) * dV;
            Real B02 =
                (eta_d_3443 * (Fh_b0 * Fh_a2) + eta_d_3443 * FFT[0][2] * hdot + lambda_d_3443 * (Fh_a0 * Fh_b2)) * dV;

            Real B10 =
                (eta_d_3443 * (Fh_b1 * Fh_a0) + eta_d_3443 * FFT[1][0] * hdot + lambda_d_3443 * (Fh_a1 * Fh_b0)) * dV;
            Real B11 =
                (eta_d_3443 * (Fh_b1 * Fh_a1) + eta_d_3443 * FFT[1][1] * hdot + lambda_d_3443 * (Fh_a1 * Fh_b1)) * dV;
            Real B12 =
                (eta_d_3443 * (Fh_b1 * Fh_a2) + eta_d_3443 * FFT[1][2] * hdot + lambda_d_3443 * (Fh_a1 * Fh_b2)) * dV;

            Real B20 =
                (eta_d_3443 * (Fh_b2 * Fh_a0) + eta_d_3443 * FFT[2][0] * hdot + lambda_d_3443 * (Fh_a2 * Fh_b0)) * dV;
            Real B21 =
                (eta_d_3443 * (Fh_b2 * Fh_a1) + eta_d_3443 * FFT[2][1] * hdot + lambda_d_3443 * (Fh_a2 * Fh_b1)) * dV;
            Real B22 =
                (eta_d_3443 * (Fh_b2 * Fh_a2) + eta_d_3443 * FFT[2][2] * hdot + lambda_d_3443 * (Fh_a2 * Fh_b2)) * dV;

            int row0 = 3 * a;
            int col0 = 3 * b;
            C_elem_3443[row0 + 0][col0 + 0] = B00;
            C_elem_3443[row0 + 0][col0 + 1] = B01;
            C_elem_3443[row0 + 0][col0 + 2] = B02;
            C_elem_3443[row0 + 1][col0 + 0] = B10;
            C_elem_3443[row0 + 1][col0 + 1] = B11;
            C_elem_3443[row0 + 1][col0 + 2] = B12;
            C_elem_3443[row0 + 2][col0 + 0] = B20;
            C_elem_3443[row0 + 2][col0 + 1] = B21;
            C_elem_3443[row0 + 2][col0 + 2] = B22;
        }
    }

    // Scatter viscous C_elem_3443 to CSR (no h scaling)
    for (int local_row = 0; local_row < Nloc3443; local_row++) {
        int node_local = local_row / 4;
        int dof_local = local_row % 4;
        int node_global = d_data->element_connectivity()(elem_idx, node_local);
        int coef_idx = node_global * 4 + dof_local;

        for (int r_dof = 0; r_dof < 3; r_dof++) {
            int global_row = 3 * coef_idx + r_dof;
            int local_row_index = 3 * local_row + r_dof;

            int row_begin = d_csr_row_offsets[global_row];
            int row_len = d_csr_row_offsets[global_row + 1] - row_begin;

            for (int local_col = 0; local_col < Nloc3443; local_col++) {
                int node_local_col = local_col / 4;
                int dof_local_col = local_col % 4;
                int node_global_col = d_data->element_connectivity()(elem_idx, node_local_col);
                int coef_idx_col = node_global_col * 4 + dof_local_col;

                for (int c_dof = 0; c_dof < 3; c_dof++) {
                    int global_col = 3 * coef_idx_col + c_dof;
                    int local_col_index = 3 * local_col + c_dof;

                    int pos = binary_search_column_csr_3443(&d_csr_col_indices[row_begin], row_len, global_col);
                    if (pos >= 0) {
                        atomicAdd(&d_csr_values[row_begin + pos], C_elem_3443[local_row_index][local_col_index]);
                    }
                }
            }
        }
    }
}

}  // namespace feris
