#pragma once
/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    FEAT4DataFunc.cuh
 * Brief:   Defines CUDA device utilities and kernels for FEAT4 (TET4)
 *          elements, including shape function evaluation, quadrature
 *          integration, internal force and stress computation, and element
 *          contribution assembly into global mass and stiffness structures.
 *
 *          TET4 uses 4-node linear tetrahedral elements with 1-point centroid
 *          quadrature. Shape functions are linear (constant strain elements).
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <cmath>

#include "../materials/MooneyRivlin.cuh"
#include "../materials/SVK.cuh"
#include "FEAT4Data.cuh"

namespace feris {

// Forward declaration for device helper templates.
struct SyncedNewtonSolver;

// ---------------------------------------------------------------------------
// compute_p for TET4: computes the deformation gradient F and first
// Piola-Kirchhoff stress P at each quadrature point.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_p(int elem_idx,
                                          int qp_idx,
                                          GPU_FEAT4_Data* d_data,
                                          const Real* __restrict__ v_guess,
                                          Real dt) {
    // Get current nodal positions for this element (4 nodes)
    Real x_nodes[4][3];

#pragma unroll
    for (int node = 0; node < 4; node++) {
        int global_node_idx = d_data->element_connectivity()(elem_idx, node);
        x_nodes[node][0] = d_data->x_cur()(global_node_idx);
        x_nodes[node][1] = d_data->y_cur()(global_node_idx);
        x_nodes[node][2] = d_data->z_cur()(global_node_idx);
    }

    // Get precomputed shape function gradients (constant for TET4)
    Real grad_N[4][3];
#pragma unroll
    for (int a = 0; a < 4; a++) {
        grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);
        grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);
        grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);
    }

    // Compute deformation gradient F = sum_a (x_nodes[a] ⊗ grad_N[a])
    Real F[3][3] = {{0.0}};
#pragma unroll
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F[i][j] += x_nodes[a][i] * grad_N[a][j];
            }
        }
    }

    // Store deformation gradient
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            d_data->F(elem_idx, qp_idx)(i, j) = F[i][j];
        }
    }

    Real eta = d_data->eta_damp();
    Real lambda_d = d_data->lambda_damp();
    const bool do_damp = (v_guess != nullptr) && (eta != 0.0 || lambda_d != 0.0);

    Real P_vis[3][3] = {{0.0}};
    if (do_damp) {
        // Compute Fdot = sum_a (v_nodes[a] ⊗ grad_N[a])
        Real Fdot[3][3] = {{0.0}};
#pragma unroll
        for (int a = 0; a < 4; a++) {
            Real v_a[3] = {0.0, 0.0, 0.0};
            int global_node_idx = d_data->element_connectivity()(elem_idx, a);
            v_a[0] = v_guess[global_node_idx * 3 + 0];
            v_a[1] = v_guess[global_node_idx * 3 + 1];
            v_a[2] = v_guess[global_node_idx * 3 + 2];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    Fdot[i][j] += v_a[i] * grad_N[a][j];
                }
            }
        }

        Real Edot[3][3] = {{0.0}};
        Real Ft[3][3];
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ft[i][j] = F[j][i];
            }
        }
        Real FdotT_F[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    FdotT_F[i][j] += Fdot[k][i] * F[k][j];
                }
            }
        }
        Real Ft_Fdot[3][3] = {{0.0}};
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    Ft_Fdot[i][j] += Ft[i][k] * Fdot[k][j];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Edot[i][j] = 0.5 * (FdotT_F[i][j] + Ft_Fdot[i][j]);
            }
        }

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
                    P_vis[i][j] += F[i][k] * S_vis[k][j];
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                d_data->Fdot(elem_idx, qp_idx)(i, j) = Fdot[i][j];
                d_data->P_vis(elem_idx, qp_idx)(i, j) = P_vis[i][j];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                d_data->Fdot(elem_idx, qp_idx)(i, j) = 0.0;
                d_data->P_vis(elem_idx, qp_idx)(i, j) = 0.0;
            }
        }
    }

    // Compute F^T * F
    Real FtF[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
#pragma unroll
            for (int k = 0; k < 3; k++) {
                FtF[i][j] += F[k][i] * F[k][j];
            }
        }
    }

    Real trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

    Real FFt[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
#pragma unroll
            for (int k = 0; k < 3; k++) {
                FFt[i][j] += F[i][k] * F[j][k];
            }
        }
    }

    Real FFtF[3][3] = {{0.0}};
#pragma unroll
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                FFtF[i][j] += FFt[i][k] * F[k][j];
            }
        }
    }

    Real P_el[3][3];
    if (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN) {
        mr_compute_P(F, d_data->mu10(), d_data->mu01(), d_data->kappa(), P_el);
    } else {
        Real lambda = d_data->lambda();
        Real mu = d_data->mu();
        svk_compute_P_from_trFtF_and_FFtF(F, trFtF, FFtF, lambda, mu, P_el);
    }

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            d_data->P(elem_idx, qp_idx)(i, j) = P_el[i][j] + P_vis[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// compute_internal_force for TET4: accumulates f_int for one node of one
// element across all quadrature points (only 1 QP for TET4).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_internal_force(int elem_idx, int node_local, GPU_FEAT4_Data* d_data) {
    int global_node_idx = d_data->element_connectivity()(elem_idx, node_local);

    Real f_node[3] = {0.0, 0.0, 0.0};

    // Only 1 QP for TET4
    for (int qp_idx = 0; qp_idx < Quadrature::N_QP_T4_1; qp_idx++) {
        Real P[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                P[i][j] = d_data->P(elem_idx, qp_idx)(i, j);
            }
        }

        Real grad_N[3];
        grad_N[0] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 0);
        grad_N[1] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 1);
        grad_N[2] = d_data->grad_N_ref(elem_idx, qp_idx)(node_local, 2);

        Real detJ = d_data->detJ_ref(elem_idx, qp_idx);
        Real wq = d_data->tet1pt_weights(qp_idx);
        Real dV = detJ * wq;

        Real f_contribution[3];
#pragma unroll
        for (int i = 0; i < 3; i++) {
            f_contribution[i] = 0.0;
            for (int j = 0; j < 3; j++) {
                f_contribution[i] += P[i][j] * grad_N[j];
            }
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            f_node[i] += f_contribution[i] * dV;
        }
    }

#pragma unroll
    for (int i = 0; i < 3; i++) {
        int global_dof_idx = 3 * global_node_idx + i;
        atomicAdd(&(d_data->f_int()(global_dof_idx)), f_node[i]);
    }
}

// ---------------------------------------------------------------------------
// clear_internal_force for TET4
// ---------------------------------------------------------------------------
__device__ __forceinline__ void clear_internal_force(GPU_FEAT4_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < d_data->n_coef * 3) {
        d_data->f_int()[thread_idx] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// compute_constraint_data for TET4
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_constraint_data(GPU_FEAT4_Data* d_data) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < d_data->gpu_n_constraint() / 3) {
        d_data->constraint()[thread_idx * 3 + 0] =
            d_data->x_cur()(d_data->fixed_nodes()[thread_idx]) - d_data->x12_jac()(d_data->fixed_nodes()[thread_idx]);
        d_data->constraint()[thread_idx * 3 + 1] =
            d_data->y_cur()(d_data->fixed_nodes()[thread_idx]) - d_data->y12_jac()(d_data->fixed_nodes()[thread_idx]);
        d_data->constraint()[thread_idx * 3 + 2] =
            d_data->z_cur()(d_data->fixed_nodes()[thread_idx]) - d_data->z12_jac()(d_data->fixed_nodes()[thread_idx]);
    }
}

// ---------------------------------------------------------------------------
// CSR-version Hessian assembly for FEAT4 (TET4 specialization)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int binary_search_column_csr_feat4(const int* cols, int n_cols, int target) {
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

// Template declaration (matches the one in FEAT10DataFunc.cuh)
template <typename ElementType>
__device__ __forceinline__ void compute_hessian_assemble_csr(ElementType* d_data,
                                                             SyncedNewtonSolver* d_solver,
                                                             int elem_idx,
                                                             int qp_idx,
                                                             int* d_csr_row_offsets,
                                                             int* d_csr_col_indices,
                                                             Real* d_csr_values,
                                                             Real h);

// Explicit specialization for GPU_FEAT4_Data
template <>
__device__ __forceinline__ void compute_hessian_assemble_csr<GPU_FEAT4_Data>(GPU_FEAT4_Data* d_data,
                                                                             SyncedNewtonSolver* d_solver,
                                                                             int elem_idx,
                                                                             int qp_idx,
                                                                             int* d_csr_row_offsets,
                                                                             int* d_csr_col_indices,
                                                                             Real* d_csr_values,
                                                                             Real h) {
    // Get element connectivity (4 nodes)
    int global_node_indices[4];
#pragma unroll
    for (int node = 0; node < 4; node++) {
        global_node_indices[node] = d_data->element_connectivity()(elem_idx, node);
    }

    // Read current nodal positions
    Real x_nodes[4][3];
#pragma unroll
    for (int node = 0; node < 4; node++) {
        int gn = global_node_indices[node];
        x_nodes[node][0] = d_data->x_cur()(gn);
        x_nodes[node][1] = d_data->y_cur()(gn);
        x_nodes[node][2] = d_data->z_cur()(gn);
    }

    // grad_N (4 nodes × 3 spatial dims)
    Real grad_N[4][3];
#pragma unroll
    for (int a = 0; a < 4; a++) {
        grad_N[a][0] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 0);
        grad_N[a][1] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 1);
        grad_N[a][2] = d_data->grad_N_ref(elem_idx, qp_idx)(a, 2);
    }

    // Compute F
    Real F[3][3] = {{0.0}};
#pragma unroll
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                F[i][j] += x_nodes[a][i] * grad_N[a][j];
            }
        }
    }

    // Compute C = F^T * F
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

    // F * grad_N for each node
    Real Fh[4][3];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int row = 0; row < 3; row++) {
            Fh[i][row] = 0.0;
#pragma unroll
            for (int col = 0; col < 3; col++) {
                Fh[i][row] += F[row][col] * grad_N[i][col];
            }
        }
    }

    Real lambda = d_data->lambda();
    Real mu = d_data->mu();
    Real detJ = d_data->detJ_ref(elem_idx, qp_idx);
    Real wq = d_data->tet1pt_weights(qp_idx);
    Real dV = detJ * wq;

    const bool use_mr = (d_data->material_model() == MATERIAL_MODEL_MOONEY_RIVLIN);
    Real A_mr[3][3][3][3];
    if (use_mr) {
        mr_compute_tangent_tensor(F, d_data->mu10(), d_data->mu01(), d_data->kappa(), A_mr);
    }

    // Local K_elem: 12×12 (4 nodes × 3 DOFs)
    Real K_elem[12][12];
#pragma unroll
    for (int ii = 0; ii < 12; ii++)
        for (int jj = 0; jj < 12; jj++)
            K_elem[ii][jj] = 0.0;

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            Real hij = grad_N[j][0] * grad_N[i][0] + grad_N[j][1] * grad_N[i][1] + grad_N[j][2] * grad_N[i][2];
            Real Fhj_dot_Fhi = Fh[j][0] * Fh[i][0] + Fh[j][1] * Fh[i][1] + Fh[j][2] * Fh[i][2];

            Real Kblock[3][3];
            if (use_mr) {
#pragma unroll
                for (int d = 0; d < 3; d++) {
#pragma unroll
                    for (int e = 0; e < 3; e++) {
                        Real sum = 0.0;
#pragma unroll
                        for (int J = 0; J < 3; J++) {
#pragma unroll
                            for (int L = 0; L < 3; L++) {
                                sum += A_mr[d][J][e][L] * grad_N[i][J] * grad_N[j][L];
                            }
                        }
                        Kblock[d][e] = sum * dV;
                    }
                }
            } else {
                svk_compute_tangent_block(Fh[i], Fh[j], hij, trE, Fhj_dot_Fhi, FFT, lambda, mu, dV, Kblock);
            }

#pragma unroll
            for (int d = 0; d < 3; d++) {
#pragma unroll
                for (int e = 0; e < 3; e++) {
                    int row = 3 * i + d;
                    int col = 3 * j + e;
                    K_elem[row][col] = Kblock[d][e];
                }
            }
        }
    }

    // Scatter to CSR
    for (int local_row_node = 0; local_row_node < 4; local_row_node++) {
        int global_node_row = global_node_indices[local_row_node];
        for (int r_dof = 0; r_dof < 3; r_dof++) {
            int global_row = 3 * global_node_row + r_dof;
            int local_row = 3 * local_row_node + r_dof;

            int row_begin = d_csr_row_offsets[global_row];
            int row_end = d_csr_row_offsets[global_row + 1];
            int row_len = row_end - row_begin;

            for (int local_col_node = 0; local_col_node < 4; local_col_node++) {
                int global_node_col = global_node_indices[local_col_node];
                for (int c_dof = 0; c_dof < 3; c_dof++) {
                    int global_col = 3 * global_node_col + c_dof;
                    int local_col = 3 * local_col_node + c_dof;

                    int pos = binary_search_column_csr_feat4(&d_csr_col_indices[row_begin], row_len, global_col);
                    if (pos >= 0) {
                        atomicAdd(&d_csr_values[row_begin + pos], h * K_elem[local_row][local_col]);
                    }
                }
            }
        }
    }

    // Viscous tangent (if enabled)
    Real eta_d = d_data->eta_damp();
    Real lambda_d = d_data->lambda_damp();
    if (eta_d == 0.0 && lambda_d == 0.0) {
        return;
    }

    // C_elem: 12×12 viscous damping matrix
    Real C_elem[12][12];
#pragma unroll
    for (int ii = 0; ii < 12; ii++)
        for (int jj = 0; jj < 12; jj++)
            C_elem[ii][jj] = 0.0;

#pragma unroll
    for (int a = 0; a < 4; a++) {
        Real* h_a = grad_N[a];
        Real Fh_a0 = Fh[a][0];
        Real Fh_a1 = Fh[a][1];
        Real Fh_a2 = Fh[a][2];
#pragma unroll
        for (int b = 0; b < 4; b++) {
            Real* h_b = grad_N[b];
            Real Fh_b0 = Fh[b][0];
            Real Fh_b1 = Fh[b][1];
            Real Fh_b2 = Fh[b][2];

            Real hdot = h_a[0] * h_b[0] + h_a[1] * h_b[1] + h_a[2] * h_b[2];

            Real Cblock00 = (eta_d * (Fh_b0 * Fh_a0) + eta_d * FFT[0][0] * hdot + lambda_d * (Fh_a0 * Fh_b0)) * dV;
            Real Cblock01 = (eta_d * (Fh_b0 * Fh_a1) + eta_d * FFT[0][1] * hdot + lambda_d * (Fh_a0 * Fh_b1)) * dV;
            Real Cblock02 = (eta_d * (Fh_b0 * Fh_a2) + eta_d * FFT[0][2] * hdot + lambda_d * (Fh_a0 * Fh_b2)) * dV;

            Real Cblock10 = (eta_d * (Fh_b1 * Fh_a0) + eta_d * FFT[1][0] * hdot + lambda_d * (Fh_a1 * Fh_b0)) * dV;
            Real Cblock11 = (eta_d * (Fh_b1 * Fh_a1) + eta_d * FFT[1][1] * hdot + lambda_d * (Fh_a1 * Fh_b1)) * dV;
            Real Cblock12 = (eta_d * (Fh_b1 * Fh_a2) + eta_d * FFT[1][2] * hdot + lambda_d * (Fh_a1 * Fh_b2)) * dV;

            Real Cblock20 = (eta_d * (Fh_b2 * Fh_a0) + eta_d * FFT[2][0] * hdot + lambda_d * (Fh_a2 * Fh_b0)) * dV;
            Real Cblock21 = (eta_d * (Fh_b2 * Fh_a1) + eta_d * FFT[2][1] * hdot + lambda_d * (Fh_a2 * Fh_b1)) * dV;
            Real Cblock22 = (eta_d * (Fh_b2 * Fh_a2) + eta_d * FFT[2][2] * hdot + lambda_d * (Fh_a2 * Fh_b2)) * dV;

            int row0 = 3 * a;
            int col0 = 3 * b;
            C_elem[row0 + 0][col0 + 0] = Cblock00;
            C_elem[row0 + 0][col0 + 1] = Cblock01;
            C_elem[row0 + 0][col0 + 2] = Cblock02;
            C_elem[row0 + 1][col0 + 0] = Cblock10;
            C_elem[row0 + 1][col0 + 1] = Cblock11;
            C_elem[row0 + 1][col0 + 2] = Cblock12;
            C_elem[row0 + 2][col0 + 0] = Cblock20;
            C_elem[row0 + 2][col0 + 1] = Cblock21;
            C_elem[row0 + 2][col0 + 2] = Cblock22;
        }
    }

    // Scatter viscous C_elem to CSR
    for (int local_row_node = 0; local_row_node < 4; local_row_node++) {
        int global_node_row = global_node_indices[local_row_node];
        for (int r_dof = 0; r_dof < 3; r_dof++) {
            int global_row = 3 * global_node_row + r_dof;
            int local_row = 3 * local_row_node + r_dof;

            int row_begin = d_csr_row_offsets[global_row];
            int row_end = d_csr_row_offsets[global_row + 1];
            int row_len = row_end - row_begin;

            for (int local_col_node = 0; local_col_node < 4; local_col_node++) {
                int global_node_col = global_node_indices[local_col_node];
                for (int c_dof = 0; c_dof < 3; c_dof++) {
                    int global_col = 3 * global_node_col + c_dof;
                    int local_col = 3 * local_col_node + c_dof;

                    int pos = binary_search_column_csr_feat4(&d_csr_col_indices[row_begin], row_len, global_col);
                    if (pos >= 0) {
                        atomicAdd(&d_csr_values[row_begin + pos], C_elem[local_row][local_col]);
                    }
                }
            }
        }
    }
}

}  // namespace feris
