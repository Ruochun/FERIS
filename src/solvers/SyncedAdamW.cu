/*==============================================================
 *==============================================================
 * Project: FERIS
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedAdamW.cu
 * Brief:   Implements the GPU-synchronized AdamW optimizer used to advance
 *          generalized velocities and positions in FERIS. Defines the
 *          cooperative kernel that evaluates element residuals, constraint
 *          contributions, and performs fully-synchronized AdamW updates for
 *          ANCF3243, ANCF3443, and FEAT10 element data.
 *==============================================================
 *==============================================================*/

#include <cooperative_groups.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/FEAT4DataFunc.cuh"
#include "SyncedAdamW.cuh"
#include <MoPhiEssentials.h>

namespace cg = cooperative_groups;

namespace feris {

// Templated solver_grad_L
template <typename ElementType>
__device__ Real solver_grad_L(int tid, ElementType* data, SyncedAdamWSolver* d_solver) {
    Real res = 0.0;

    const int node_i = tid / 3;
    const int dof_i = tid % 3;

    const int n_coef = d_solver->get_n_coef();
    const Real inv_dt = 1.0 / d_solver->solver_time_step();
    const Real dt = d_solver->solver_time_step();

    // Cache pointers once
    const Real* __restrict__ v_g = d_solver->v_guess().data();
    const Real* __restrict__ v_p = d_solver->v_prev().data();
    const int* __restrict__ offsets = data->csr_offsets();
    const int* __restrict__ columns = data->csr_columns();
    const Real* __restrict__ values = data->csr_values();

    // Mass matrix contribution: (M @ (v_loc - v_prev)) / h
    int row_start = offsets[node_i];
    int row_end = offsets[node_i + 1];

    for (int idx = row_start; idx < row_end; idx++) {
        int node_j = columns[idx];
        Real mass_ij = values[idx];
        int tid_j = node_j * 3 + dof_i;
        Real v_diff = v_g[tid_j] - v_p[tid_j];
        res += mass_ij * v_diff * inv_dt;
    }

    // Mechanical force contribution: - (-f_int + f_ext) = f_int - f_ext
    res -= (-data->f_int()(tid));  // Add f_int
    res -= data->f_ext()(tid);     // Subtract f_ext

    const int n_constraints = d_solver->gpu_n_constraints();

    if (n_constraints > 0) {
        // Python: h * (J.T @ (lam_mult + rho_bb * cA))
        const Real rho = *d_solver->solver_rho();

        const Real* __restrict__ lam = d_solver->lambda_guess().data();
        const Real* __restrict__ con = data->constraint().data();

        // CSR format stores J^T (transpose of constraint Jacobian).
        const int* __restrict__ cjT_offsets = data->cj_csr_offsets();
        const int* __restrict__ cjT_columns = data->cj_csr_columns();
        const Real* __restrict__ cjT_values = data->cj_csr_values();

        // Get all constraints that affect this DOF (tid)
        const int col_start = cjT_offsets[tid];
        const int col_end = cjT_offsets[tid + 1];

        for (int idx = col_start; idx < col_end; idx++) {
            const int constraint_idx = cjT_columns[idx];      // Which constraint
            const Real constraint_jac_val = cjT_values[idx];  // J^T[tid, constraint_idx] = J[constraint_idx, tid]
            const Real constraint_val = con[constraint_idx];

            // Add constraint contribution: h * J^T * (lambda + rho*c)
            res += dt * constraint_jac_val * (lam[constraint_idx] + rho * constraint_val);
        }
    }

    return res;
}

// Templated kernel
template <typename ElementType>
__global__ void one_step_adamw_kernel_impl(ElementType* d_data, SyncedAdamWSolver* d_adamw_solver) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Save previous positions
    if (tid < d_adamw_solver->get_n_coef()) {
        d_adamw_solver->x12_prev()(tid) = d_data->x_cur()(tid);
        d_adamw_solver->y12_prev()(tid) = d_data->y_cur()(tid);
        d_adamw_solver->z12_prev()(tid) = d_data->z_cur()(tid);
    }

    grid.sync();

    if (tid == 0) {
        *d_adamw_solver->inner_flag() = 0;
        *d_adamw_solver->outer_flag() = 0;
    }

    grid.sync();

    for (int outer_iter = 0; outer_iter < d_adamw_solver->solver_max_outer(); outer_iter++) {
        if (*d_adamw_solver->outer_flag() == 0) {
            // Initialize per-thread variables
            Real t = 1.0;
            Real m_t = 0.0;
            Real v_t = 0.0;

            Real lr = d_adamw_solver->solver_lr();
            Real beta1 = d_adamw_solver->solver_beta1();
            Real beta2 = d_adamw_solver->solver_beta2();
            Real eps = d_adamw_solver->solver_eps();
            Real weight_decay = d_adamw_solver->solver_weight_decay();
            Real lr_decay = d_adamw_solver->solver_lr_decay();
            int conv_check_interval = d_adamw_solver->solver_convergence_check_interval();

            if (tid == 0) {
                *d_adamw_solver->norm_g() = 0.0;
                *d_adamw_solver->inner_flag() = 0;
            }

            grid.sync();

            if (tid < d_adamw_solver->get_n_coef() * 3) {
                d_adamw_solver->g()(tid) = 0.0;
                t = 1.0;
            }

            Real norm_g0 = -1.0;
            for (int inner_iter = 0; inner_iter < d_adamw_solver->solver_max_inner(); inner_iter++) {
                grid.sync();

                if (*d_adamw_solver->inner_flag() == 0) {
                    // if (tid == 0 && inner_iter % conv_check_interval == 0) {
                    //   printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
                    // }

                    // Step 1: Compute look-ahead velocity
                    Real y = 0.0;
                    if (tid < d_adamw_solver->get_n_coef() * 3) {
                        Real g_tid = d_adamw_solver->g()(tid);
                        Real v_guess_tid = d_adamw_solver->v_guess()(tid);
                        lr = lr * lr_decay;
                        t += 1;

                        m_t = beta1 * m_t + (1 - beta1) * g_tid;
                        v_t = beta2 * v_t + (1 - beta2) * g_tid * g_tid;
                        Real m_hat = m_t / (1 - pow(beta1, t));
                        Real v_hat = v_t / (1 - pow(beta2, t));
                        y = v_guess_tid - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * v_guess_tid);

                        d_adamw_solver->v_guess()(tid) = y;
                    }

                    grid.sync();

                    // Step 2: Update scratch positions
                    if (tid < d_adamw_solver->get_n_coef()) {
                        d_data->x_cur()(tid) =
                            d_adamw_solver->x12_prev()(tid) +
                            d_adamw_solver->solver_time_step() * d_adamw_solver->v_guess()(tid * 3 + 0);
                        d_data->y_cur()(tid) =
                            d_adamw_solver->y12_prev()(tid) +
                            d_adamw_solver->solver_time_step() * d_adamw_solver->v_guess()(tid * 3 + 1);
                        d_data->z_cur()(tid) =
                            d_adamw_solver->z12_prev()(tid) +
                            d_adamw_solver->solver_time_step() * d_adamw_solver->v_guess()(tid * 3 + 2);
                    }

                    grid.sync();

                    // Compute P (stress)
                    if (tid < d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_total_qp()) {
                        for (int idx = tid; idx < d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_total_qp();
                             idx += grid.size()) {
                            int elem_idx = idx / d_adamw_solver->gpu_n_total_qp();
                            int qp_idx = idx % d_adamw_solver->gpu_n_total_qp();
                            compute_p(elem_idx, qp_idx, d_data, d_adamw_solver->v_guess().data(),
                                      d_adamw_solver->solver_time_step());
                        }
                    }

                    grid.sync();

                    // Clear internal force
                    if (tid < d_adamw_solver->get_n_coef() * 3) {
                        clear_internal_force(d_data);
                    }

                    grid.sync();

                    // Compute internal force
                    if (tid < d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_shape()) {
                        for (int idx = tid; idx < d_adamw_solver->get_n_beam() * d_adamw_solver->gpu_n_shape();
                             idx += grid.size()) {
                            int elem_idx = idx / d_adamw_solver->gpu_n_shape();
                            int node_idx = idx % d_adamw_solver->gpu_n_shape();
                            compute_internal_force(elem_idx, node_idx, d_data);
                        }
                    }

                    grid.sync();

                    // Compute constraints
                    if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
                        compute_constraint_data(d_data);
                    }

                    grid.sync();

                    // Compute gradient
                    if (tid < d_adamw_solver->get_n_coef() * 3) {
                        Real g = solver_grad_L(tid, d_data, d_adamw_solver);
                        d_adamw_solver->g()[tid] = g;
                    }

                    grid.sync();

                    // Check convergence
                    if (tid == 0 && inner_iter % conv_check_interval == 0) {
                        Real norm_g = 0.0;
                        for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
                            norm_g += d_adamw_solver->g()(i) * d_adamw_solver->g()(i);
                        }
                        *d_adamw_solver->norm_g() = sqrt(norm_g);

                        if (norm_g0 < 0.0) {
                            norm_g0 = *d_adamw_solver->norm_g();
                        }

                        Real norm_v_curr = 0.0;
                        for (int i = 0; i < 3 * d_adamw_solver->get_n_coef(); i++) {
                            norm_v_curr += d_adamw_solver->v_guess()(i) * d_adamw_solver->v_guess()(i);
                        }
                        norm_v_curr = sqrt(norm_v_curr);

                        const Real inner_tol = d_adamw_solver->solver_inner_tol();
                        const Real rtol = d_adamw_solver->solver_inner_rtol();
                        const Real tol_abs = inner_tol * (1.0 + norm_v_curr);
                        const Real tol_rel = (rtol > 0.0 && norm_g0 > 0.0) ? (rtol * norm_g0) : 0.0;

                        // printf(
                        //     "norm_g: %.17f, norm_v_curr: %.17f, tol_abs: %.17f, rtol*g0: "
                        //     "%.17f\n",
                        //     *d_adamw_solver->norm_g(), norm_v_curr, tol_abs, tol_rel);

                        if (*d_adamw_solver->norm_g() <= tol_abs ||
                            (tol_rel > 0.0 && *d_adamw_solver->norm_g() <= tol_rel)) {
                            // const Real tol = (tol_abs > tol_rel ? tol_abs : tol_rel);
                            // printf("Converged: gnorm=%.17f <= max(tol_abs, rtol*g0)=%.17f\n",
                            //        *d_adamw_solver->norm_g(), tol);
                            *d_adamw_solver->inner_flag() = 1;
                        }
                    }

                    grid.sync();
                }
            }

            // Update v_prev
            if (tid < d_adamw_solver->get_n_coef() * 3) {
                d_adamw_solver->v_prev()[tid] = d_adamw_solver->v_guess()[tid];
            }

            grid.sync();

            // Update positions
            if (tid < d_adamw_solver->get_n_coef()) {
                d_data->x_cur()(tid) = d_adamw_solver->x12_prev()(tid) +
                                     d_adamw_solver->v_guess()(tid * 3 + 0) * d_adamw_solver->solver_time_step();
                d_data->y_cur()(tid) = d_adamw_solver->y12_prev()(tid) +
                                     d_adamw_solver->v_guess()(tid * 3 + 1) * d_adamw_solver->solver_time_step();
                d_data->z_cur()(tid) = d_adamw_solver->z12_prev()(tid) +
                                     d_adamw_solver->v_guess()(tid * 3 + 2) * d_adamw_solver->solver_time_step();
            }

            grid.sync();

            // Compute constraints at new position
            if (tid < d_adamw_solver->gpu_n_constraints() / 3) {
                compute_constraint_data(d_data);
            }

            grid.sync();

            // dual variable update
            int n_constraints = d_adamw_solver->gpu_n_constraints();
            for (int i = tid; i < n_constraints; i += grid.size()) {
                Real constraint_val = d_data->constraint()[i];
                d_adamw_solver->lambda_guess()[i] +=
                    *d_adamw_solver->solver_rho() * d_adamw_solver->solver_time_step() * constraint_val;
            }
            grid.sync();

            if (tid == 0) {
                // check constraint convergence
                Real norm_constraint = 0.0;
                for (int i = 0; i < d_adamw_solver->gpu_n_constraints(); i++) {
                    Real constraint_val = d_data->constraint()[i];
                    norm_constraint += constraint_val * constraint_val;
                }
                norm_constraint = sqrt(norm_constraint);
                // printf("norm_constraint: %.17f\n", norm_constraint);

                if (norm_constraint < d_adamw_solver->solver_outer_tol()) {
                    // printf("Converged constraint: %.17f\n", norm_constraint);
                    *d_adamw_solver->outer_flag() = 1;
                }
            }

            grid.sync();
        }
    }

    // Final position update
    if (tid < d_adamw_solver->get_n_coef()) {
        d_data->x_cur()(tid) = d_adamw_solver->x12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 0) * d_adamw_solver->solver_time_step();
        d_data->y_cur()(tid) = d_adamw_solver->y12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 1) * d_adamw_solver->solver_time_step();
        d_data->z_cur()(tid) = d_adamw_solver->z12_prev()(tid) +
                             d_adamw_solver->v_guess()(tid * 3 + 2) * d_adamw_solver->solver_time_step();
    }

    grid.sync();
}

// Explicit instantiations
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, SyncedAdamWSolver*);
template __global__ void one_step_adamw_kernel_impl<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, SyncedAdamWSolver*);
template __global__ void one_step_adamw_kernel_impl<GPU_FEAT10_Data>(GPU_FEAT10_Data*, SyncedAdamWSolver*);
template __global__ void one_step_adamw_kernel_impl<GPU_FEAT4_Data>(GPU_FEAT4_Data*, SyncedAdamWSolver*);

void SyncedAdamWSolver::OneStepAdamW() {
    cudaEvent_t start, stop;
    MOPHI_GPU_CALL(cudaEventCreate(&start));
    MOPHI_GPU_CALL(cudaEventCreate(&stop));

    int threads = 128;
    cudaDeviceProp props;
    MOPHI_GPU_CALL(cudaGetDeviceProperties(&props, 0));

    int N_dof = 3 * n_coef_;
    int blocksNeededDof = (N_dof + threads - 1) / threads;
    int n_constraint_nodes = n_constraints_ / 3;
    int blocksNeededConstr = (n_constraint_nodes + threads - 1) / threads;
    int blocksNeeded = std::max(blocksNeededDof, blocksNeededConstr);

    int maxBlocksPerSm = 0;

    if (type_ == TYPE_3243) {
        MOPHI_GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, threads, 0));
    } else if (type_ == TYPE_3443) {
        MOPHI_GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, threads, 0));
    } else if (type_ == TYPE_T10) {
        MOPHI_GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_FEAT10_Data>, threads, 0));
    } else if (type_ == TYPE_T4) {
        MOPHI_GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSm, one_step_adamw_kernel_impl<GPU_FEAT4_Data>, threads, 0));
    }

    int maxCoopBlocks = maxBlocksPerSm * props.multiProcessorCount;

    if (blocksNeeded > maxCoopBlocks) {
        MOPHI_ERROR(
            "SyncedAdamW: problem size too large for cooperative launch on "
            "%s. Requested blocks: %d, max cooperative blocks: %d, n_coef: %d, n_constraints: %d",
            props.name, blocksNeeded, maxCoopBlocks, n_coef_, n_constraints_);
    }

    int blocks = blocksNeeded;

    MOPHI_GPU_CALL(cudaEventRecord(start));

    // Launch appropriate templated kernel based on element type
    if (type_ == TYPE_3243) {
        void* args[] = {&d_data_, &d_adamw_solver_};
        MOPHI_GPU_CALL(
            cudaLaunchCooperativeKernel((void*)one_step_adamw_kernel_impl<GPU_ANCF3243_Data>, blocks, threads, args));
    } else if (type_ == TYPE_3443) {
        void* args[] = {&d_data_, &d_adamw_solver_};
        MOPHI_GPU_CALL(
            cudaLaunchCooperativeKernel((void*)one_step_adamw_kernel_impl<GPU_ANCF3443_Data>, blocks, threads, args));
    } else if (type_ == TYPE_T10) {
        void* args[] = {&d_data_, &d_adamw_solver_};
        MOPHI_GPU_CALL(
            cudaLaunchCooperativeKernel((void*)one_step_adamw_kernel_impl<GPU_FEAT10_Data>, blocks, threads, args));
    } else if (type_ == TYPE_T4) {
        void* args[] = {&d_data_, &d_adamw_solver_};
        MOPHI_GPU_CALL(
            cudaLaunchCooperativeKernel((void*)one_step_adamw_kernel_impl<GPU_FEAT4_Data>, blocks, threads, args));
    }

    MOPHI_GPU_CALL(cudaEventRecord(stop));
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    float milliseconds = 0;
    MOPHI_GPU_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    MOPHI_INFO("OneStepAdamW kernel time: %.3f ms", milliseconds);

    MOPHI_GPU_CALL(cudaEventDestroy(start));
    MOPHI_GPU_CALL(cudaEventDestroy(stop));
}

}  // namespace feris
