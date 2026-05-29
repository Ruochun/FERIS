/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SyncedNesterov.cu
 * Brief:   Implements a GPU-synchronized Nesterov accelerated gradient
 *          integrator. Defines the cooperative kernel that performs multiple
 *          inner/outer iterations, computing gradients, applying Nesterov
 *          momentum, updating positions, and enforcing constraints for
 *          ANCF3243, ANCF3443, and FEAT10 elements.
 *==============================================================
 *==============================================================*/

#include <cooperative_groups.h>
#include <MoPhiEssentials.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/FEAT4DataFunc.cuh"
#include "SyncedNesterov.cuh"

namespace cg = cooperative_groups;

namespace feris {

template <typename ElementType>
__device__ Real solver_grad_L(int tid, ElementType* data, SyncedNesterovSolver* d_solver) {
    Real res = 0.0;

    int node_i = tid / 3;
    int dof_i = tid % 3;

    const Real inv_dt = 1.0 / d_solver->solver_time_step();
    const Real dt = d_solver->solver_time_step();

    // Mass matrix contribution using CSR format
    int* offsets = data->csr_offsets();
    int* columns = data->csr_columns();
    Real* values = data->csr_values();

    int row_start = offsets[node_i];
    int row_end = offsets[node_i + 1];

    for (int idx = row_start; idx < row_end; idx++) {
        int node_j = columns[idx];
        Real mass_ij = values[idx];
        int tid_j = node_j * 3 + dof_i;
        Real v_diff = d_solver->v_guess()[tid_j] - d_solver->v_prev()[tid_j];
        res += mass_ij * v_diff * inv_dt;
    }

    // Internal force
    res -= (-data->f_int()(tid));

    // External force
    res -= data->f_ext()(tid);

    // Constraints using CSR format for J^T
    const int n_constraints = d_solver->gpu_n_constraints();

    if (n_constraints > 0) {
        // Python: h * (J.T @ (lam_mult + rho_bb * cA))
        const Real rho = *d_solver->solver_rho();

        const Real* __restrict__ lam = d_solver->lambda_guess().data();
        const Real* __restrict__ con = data->constraint().data();

        // CSR format stores J^T (transpose of constraint Jacobian)
        const int* __restrict__ cjT_offsets = data->cj_csr_offsets();
        const int* __restrict__ cjT_columns = data->cj_csr_columns();
        const Real* __restrict__ cjT_values = data->cj_csr_values();

        // Get all constraints that affect this DOF (tid)
        const int col_start = cjT_offsets[tid];
        const int col_end = cjT_offsets[tid + 1];

        // Loop through all constraints affecting this DOF
        for (int idx = col_start; idx < col_end; idx++) {
            const int constraint_idx = cjT_columns[idx];
            const Real constraint_jac_val = cjT_values[idx];
            const Real constraint_val = con[constraint_idx];

            // Add constraint contribution: h * J^T * (lambda + rho*c)
            res += dt * constraint_jac_val * (lam[constraint_idx] + rho * constraint_val);
        }
    }

    return res;
}

template <typename ElementType>
__global__ void one_step_nesterov_kernel(ElementType* data, SyncedNesterovSolver* d_nesterov_solver) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Assign x_prev, y_prev, z_prev
    if (tid < d_nesterov_solver->get_n_coef()) {
        d_nesterov_solver->x_prev()(tid) = data->x_cur()(tid);
        d_nesterov_solver->y_prev()(tid) = data->y_cur()(tid);
        d_nesterov_solver->z_prev()(tid) = data->z_cur()(tid);
    }

    grid.sync();

    if (tid == 0) {
        *d_nesterov_solver->inner_flag() = 0;
        *d_nesterov_solver->outer_flag() = 0;
    }

    grid.sync();

    for (int outer_iter = 0; outer_iter < d_nesterov_solver->solver_max_outer(); outer_iter++) {
        if (*d_nesterov_solver->outer_flag() == 0) {
            // Initialize variables for each thread
            Real v_k = 0.0;
            Real v_next = 0.0;
            Real v_km1 = 0.0;
            Real t = 1.0;

            if (tid == 0) {
                *d_nesterov_solver->prev_norm_g() = 0.0;
                *d_nesterov_solver->norm_g() = 0.0;
                *d_nesterov_solver->inner_flag() = 0;
            }

            grid.sync();

            // Initialize for valid threads only
            if (tid < d_nesterov_solver->get_n_coef() * 3) {
                v_k = d_nesterov_solver->v_guess()(tid);
                v_km1 = d_nesterov_solver->v_guess()(tid);
                t = 1.0;
            }

            Real t_next = 1.0;

            for (int inner_iter = 0; inner_iter < d_nesterov_solver->solver_max_inner(); inner_iter++) {
                grid.sync();

                if (*d_nesterov_solver->inner_flag() == 0) {
                    // if (tid == 0) {
                    //     printf("outer iter: %d, inner iter: %d\n", outer_iter, inner_iter);
                    // }

                    // Step 1: Compute look-ahead velocity
                    Real y = 0.0;
                    if (tid < d_nesterov_solver->get_n_coef() * 3) {
                        t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
                        Real beta = (t - 1.0) / t_next;
                        y = v_k + beta * (v_k - v_km1);

                        d_nesterov_solver->v_guess()(tid) = y;
                    }

                    grid.sync();

                    // Step 2: Update scratch positions
                    if (tid < d_nesterov_solver->get_n_coef()) {
                        data->x_cur()(tid) =
                            d_nesterov_solver->x_prev()(tid) +
                            d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 0);
                        data->y_cur()(tid) =
                            d_nesterov_solver->y_prev()(tid) +
                            d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 1);
                        data->z_cur()(tid) =
                            d_nesterov_solver->z_prev()(tid) +
                            d_nesterov_solver->solver_time_step() * d_nesterov_solver->v_guess()(tid * 3 + 2);
                    }

                    grid.sync();

                    // Compute P at quadrature points
                    if (tid < d_nesterov_solver->get_n_beam() * d_nesterov_solver->gpu_n_total_qp()) {
                        for (int idx = tid; idx < d_nesterov_solver->get_n_beam() * d_nesterov_solver->gpu_n_total_qp();
                             idx += grid.size()) {
                            int elem_idx = idx / d_nesterov_solver->gpu_n_total_qp();
                            int qp_idx = idx % d_nesterov_solver->gpu_n_total_qp();
                            compute_p(elem_idx, qp_idx, data, d_nesterov_solver->v_guess().data(),
                                      d_nesterov_solver->solver_time_step());
                        }
                    }

                    grid.sync();

                    // Clear internal forces
                    if (tid < d_nesterov_solver->get_n_coef() * 3) {
                        clear_internal_force(data);
                    }

                    grid.sync();

                    // Compute internal forces
                    if (tid < d_nesterov_solver->get_n_beam() * d_nesterov_solver->gpu_n_shape()) {
                        for (int idx = tid; idx < d_nesterov_solver->get_n_beam() * d_nesterov_solver->gpu_n_shape();
                             idx += grid.size()) {
                            int elem_idx = idx / d_nesterov_solver->gpu_n_shape();
                            int node_idx = idx % d_nesterov_solver->gpu_n_shape();
                            compute_internal_force(elem_idx, node_idx, data);
                        }
                    }

                    grid.sync();

                    // Compute constraint data
                    if (tid < d_nesterov_solver->gpu_n_constraints() / 3) {
                        compute_constraint_data(data);
                    }

                    grid.sync();

                    // Compute gradient
                    if (tid < d_nesterov_solver->get_n_coef() * 3) {
                        Real g = solver_grad_L(tid, data, d_nesterov_solver);
                        d_nesterov_solver->g()[tid] = g;
                    }

                    grid.sync();

                    if (tid == 0) {
                        // Calculate norm of g
                        Real norm_g = 0.0;
                        for (int i = 0; i < 3 * d_nesterov_solver->get_n_coef(); i++) {
                            norm_g += d_nesterov_solver->g()(i) * d_nesterov_solver->g()(i);
                        }
                        *d_nesterov_solver->norm_g() = sqrt(norm_g);
                        // printf("norm_g: %.17f\n", *d_nesterov_solver->norm_g());

                        if (inner_iter > 0 && abs(*d_nesterov_solver->norm_g() - *d_nesterov_solver->prev_norm_g()) <
                                                  d_nesterov_solver->solver_inner_tol()) {
                            // printf("Converged diff: %.17f\n",
                            //        *d_nesterov_solver->norm_g() - *d_nesterov_solver->prev_norm_g());
                            *d_nesterov_solver->inner_flag() = 1;
                        }
                    }

                    grid.sync();

                    // Step 4: Update velocities
                    if (tid < d_nesterov_solver->get_n_coef() * 3) {
                        v_next = y - d_nesterov_solver->solver_alpha() * d_nesterov_solver->g()[tid];

                        d_nesterov_solver->v_next()[tid] = v_next;
                        d_nesterov_solver->v_k()[tid] = v_k;
                    }

                    grid.sync();

                    // Check velocity convergence
                    if (tid == 0) {
                        Real norm_v_next = 0.0;
                        Real norm_v_k = 0.0;
                        for (int i = 0; i < 3 * d_nesterov_solver->get_n_coef(); i++) {
                            norm_v_next += d_nesterov_solver->v_next()(i) * d_nesterov_solver->v_next()(i);
                            norm_v_k += d_nesterov_solver->v_k()(i) * d_nesterov_solver->v_k()(i);
                        }
                        norm_v_next = sqrt(norm_v_next);
                        norm_v_k = sqrt(norm_v_k);
                        // printf("norm_v_next: %.17f, norm_v_k: %.17f\n", norm_v_next, norm_v_k);

                        if (inner_iter > 0 && abs(norm_v_next - norm_v_k) < d_nesterov_solver->solver_inner_tol()) {
                            // printf("Converged velocity: %.17f\n", abs(norm_v_next - norm_v_k));
                            *d_nesterov_solver->inner_flag() = 1;
                        }
                    }

                    grid.sync();

                    if (tid < d_nesterov_solver->get_n_coef() * 3) {
                        v_km1 = v_k;
                        v_k = v_next;
                        t = t_next;

                        d_nesterov_solver->v_guess()[tid] = v_next;
                    }

                    grid.sync();

                    if (tid == 0) {
                        *d_nesterov_solver->prev_norm_g() = *d_nesterov_solver->norm_g();
                    }

                    grid.sync();
                }
            }

            // Update v_prev for next outer iteration
            if (tid < d_nesterov_solver->get_n_coef() * 3) {
                d_nesterov_solver->v_prev()[tid] = d_nesterov_solver->v_guess()[tid];
            }

            grid.sync();

            // Update positions
            if (tid < d_nesterov_solver->get_n_coef()) {
                data->x_cur()(tid) = d_nesterov_solver->x_prev()(tid) +
                                     d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
                data->y_cur()(tid) = d_nesterov_solver->y_prev()(tid) +
                                     d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
                data->z_cur()(tid) = d_nesterov_solver->z_prev()(tid) +
                                     d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
            }

            grid.sync();

            // Compute constraints at new position
            if (tid < d_nesterov_solver->gpu_n_constraints() / 3) {
                compute_constraint_data(data);
            }

            grid.sync();

            // Dual variable update
            int n_constraints = d_nesterov_solver->gpu_n_constraints();
            for (int i = tid; i < n_constraints; i += grid.size()) {
                Real constraint_val = data->constraint()[i];
                d_nesterov_solver->lambda_guess()[i] +=
                    *d_nesterov_solver->solver_rho() * d_nesterov_solver->solver_time_step() * constraint_val;
            }
            grid.sync();

            if (tid == 0) {
                // Check constraint convergence
                Real norm_constraint = 0.0;
                for (int i = 0; i < d_nesterov_solver->gpu_n_constraints(); i++) {
                    Real constraint_val = data->constraint()[i];
                    norm_constraint += constraint_val * constraint_val;
                }
                norm_constraint = sqrt(norm_constraint);
                // printf("norm_constraint: %.17f\n", norm_constraint);

                if (abs(norm_constraint) < d_nesterov_solver->solver_outer_tol()) {
                    // printf("Converged constraint: %.17f\n", abs(norm_constraint));
                    *d_nesterov_solver->outer_flag() = 1;
                }
            }

            grid.sync();
        }
        grid.sync();
    }

    // Final position update
    if (tid < d_nesterov_solver->get_n_coef()) {
        data->x_cur()(tid) = d_nesterov_solver->x_prev()(tid) +
                             d_nesterov_solver->v_guess()(tid * 3 + 0) * d_nesterov_solver->solver_time_step();
        data->y_cur()(tid) = d_nesterov_solver->y_prev()(tid) +
                             d_nesterov_solver->v_guess()(tid * 3 + 1) * d_nesterov_solver->solver_time_step();
        data->z_cur()(tid) = d_nesterov_solver->z_prev()(tid) +
                             d_nesterov_solver->v_guess()(tid * 3 + 2) * d_nesterov_solver->solver_time_step();
    }

    grid.sync();
}

// Explicit template instantiations
template __global__ void one_step_nesterov_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, SyncedNesterovSolver*);
template __global__ void one_step_nesterov_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, SyncedNesterovSolver*);
template __global__ void one_step_nesterov_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, SyncedNesterovSolver*);
template __global__ void one_step_nesterov_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, SyncedNesterovSolver*);

void SyncedNesterovSolver::OneStepNesterov() {
    cudaEvent_t start, stop;
    MOPHI_GPU_CALL(cudaEventCreate(&start));
    MOPHI_GPU_CALL(cudaEventCreate(&stop));

    int threads = 128;

    cudaDeviceProp props;
    MOPHI_GPU_CALL(cudaGetDeviceProperties(&props, 0));

    int maxBlocksPerSm = 0;
    void* kernelPtr = nullptr;

    // Select the appropriate kernel based on element type
    if (type_ == TYPE_3243) {
        kernelPtr = (void*)one_step_nesterov_kernel<GPU_ANCF3243_Data>;
    } else if (type_ == TYPE_3443) {
        kernelPtr = (void*)one_step_nesterov_kernel<GPU_ANCF3443_Data>;
    } else if (type_ == TYPE_T10) {
        kernelPtr = (void*)one_step_nesterov_kernel<GPU_FEAT10_Data>;
    } else if (type_ == TYPE_T4) {
        kernelPtr = (void*)one_step_nesterov_kernel<GPU_FEAT4_Data>;
    }

    MOPHI_GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm, kernelPtr, threads, 0));
    int maxCoopBlocks = maxBlocksPerSm * props.multiProcessorCount;

    int N = 3 * n_coef_;
    int blocksNeeded = (N + threads - 1) / threads;
    int blocks = std::min(blocksNeeded, maxCoopBlocks);

    void* args[] = {&d_data_, &d_nesterov_solver_};

    MOPHI_GPU_CALL(cudaEventRecord(start));
    MOPHI_GPU_CALL(cudaLaunchCooperativeKernel(kernelPtr, blocks, threads, args));
    MOPHI_GPU_CALL(cudaEventRecord(stop));

    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    float milliseconds = 0;
    MOPHI_GPU_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    // std::cout << "OneStepNesterov kernel time: " << milliseconds << " ms" << std::endl;

    MOPHI_GPU_CALL(cudaEventDestroy(start));
    MOPHI_GPU_CALL(cudaEventDestroy(stop));
}

}  // namespace feris