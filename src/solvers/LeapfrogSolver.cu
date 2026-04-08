/*==============================================================
 *==============================================================
 * Project: TLFEA
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    LeapfrogSolver.cu
 * Brief:   Implements the explicit central-difference (leapfrog) time
 *          integrator. Defines GPU kernels that compute the lumped mass
 *          from the CSR consistent mass matrix, evaluate element stress /
 *          facet tractions, assemble internal forces, update half-step nodal
 *          velocities, enforce fixed-node boundary conditions, and advance
 *          nodal positions. Supports ANCF3243, ANCF3443, FEAT10, FEAT4, and
 *          LDPM4 element types.
 *==============================================================
 *==============================================================*/

#include <algorithm>

#include <MoPhiEssentials.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3243DataFunc.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ANCF3443DataFunc.cuh"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT10DataFunc.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/FEAT4DataFunc.cuh"
#include "../elements/LDPM4Data.cuh"
#include "../elements/LDPM4DataFunc.cuh"
#include "LeapfrogSolver.cuh"

namespace tlfea {

// ---------------------------------------------------------------------------
// leapfrog_compute_lumped_mass_kernel
//
// Computes the lumped (row-sum) nodal mass from the pre-assembled CSR
// consistent mass matrix stored in the element data. One thread per node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_compute_lumped_mass_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int node_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_i >= d_solver->get_n_coef()) return;

    const int* __restrict__ offsets = d_data->csr_offsets();
    const Real* __restrict__ values = d_data->csr_values();

    Real m_i = 0.0;
    for (int idx = offsets[node_i]; idx < offsets[node_i + 1]; idx++) {
        m_i += values[idx];
    }
    d_solver->mass_lump()(node_i) = m_i;
}

// ---------------------------------------------------------------------------
// leapfrog_compute_p_kernel
//
// Computes the deformation gradient F and first Piola-Kirchhoff stress P
// (including viscous contribution) at every quadrature point. Uses a
// grid-stride loop for load balancing. The current half-step velocity
// d_v is passed to compute_p for Rayleigh-type viscous damping (Fdot).
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_compute_p_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = d_solver->get_n_beam() * d_solver->get_n_total_qp();

    for (int idx = tid; idx < total; idx += stride) {
        int elem_idx = idx / d_solver->get_n_total_qp();
        int qp_idx = idx % d_solver->get_n_total_qp();
        compute_p(elem_idx, qp_idx, d_data, d_solver->v().data(), d_solver->solver_time_step());
    }
}

// ---------------------------------------------------------------------------
// leapfrog_clear_f_int_kernel
//
// Zeroes the internal force vector. One thread per global DOF.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_clear_f_int_kernel(ElemType* d_data) {
    clear_internal_force(d_data);
}

// ---------------------------------------------------------------------------
// leapfrog_compute_f_int_kernel
//
// Assembles internal nodal forces by summing stress-weighted shape function
// gradient contributions. Uses a grid-stride loop over (element, local node)
// pairs so that the kernel scales gracefully for any problem size.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_compute_f_int_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = d_solver->get_n_beam() * d_solver->get_n_shape();

    for (int idx = tid; idx < total; idx += stride) {
        int elem_idx = idx / d_solver->get_n_shape();
        int node_local = idx % d_solver->get_n_shape();
        compute_internal_force(elem_idx, node_local, d_data);
    }
}

// ---------------------------------------------------------------------------
// leapfrog_update_velocity_kernel
//
// Central-difference velocity update:
//   v_{n+1/2} = v_{n-1/2} + dt * M_lump^{-1} * (f_ext - f_int)
//
// One thread per global DOF (node * 3 + direction).
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_update_velocity_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef() * 3) return;

    int node_i = tid / 3;
    Real dt = d_solver->solver_time_step();
    Real mass_val = d_solver->mass_lump()(node_i);
    // Guard against orphan nodes (zero lumped mass) – they contribute no dynamics.
    if (mass_val <= Real(0)) return;
    Real inv_mass = Real(1) / mass_val;
    Real net_force = d_data->f_ext()(tid) - d_data->f_int()(tid);

    d_solver->v()(tid) += dt * inv_mass * net_force;
}

// ---------------------------------------------------------------------------
// leapfrog_apply_bc_kernel
//
// Enforces fixed-node (Dirichlet) boundary conditions by zeroing the
// velocity at all fixed DOFs. Fixed nodes are read from the element data.
// One thread per fixed node; each thread zeroes all 3 DOFs of that node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_apply_bc_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_fixed = d_data->gpu_n_constraint() / 3;
    if (tid >= n_fixed) return;

    int node_idx = d_data->fixed_nodes()(tid);
    d_solver->v()(node_idx * 3 + 0) = 0.0;
    d_solver->v()(node_idx * 3 + 1) = 0.0;
    d_solver->v()(node_idx * 3 + 2) = 0.0;
}

// ---------------------------------------------------------------------------
// leapfrog_update_position_kernel
//
// Full-step position update:
//   x_{n+1} = x_n + dt * v_{n+1/2}
//
// One thread per node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_update_position_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef()) return;

    Real dt = d_solver->solver_time_step();
    d_data->x12()(tid) += dt * d_solver->v()(tid * 3 + 0);
    d_data->y12()(tid) += dt * d_solver->v()(tid * 3 + 1);
    d_data->z12()(tid) += dt * d_solver->v()(tid * 3 + 2);
}

// ---------------------------------------------------------------------------
// LeapfrogSolver::Setup
//
// Copies this struct to the device mirror and computes the lumped mass from
// the CSR consistent mass matrix. Must be called after CalcMassMatrix() and
// after SetParameters().
// ---------------------------------------------------------------------------
void LeapfrogSolver::Setup() {
    da_mass_lump_.SetVal(Real(0));
    da_mass_lump_.MakeReadyDevice();

    MOPHI_GPU_CALL(cudaMemcpy(d_leapfrog_solver_, this, sizeof(LeapfrogSolver), cudaMemcpyHostToDevice));

    constexpr int threadsPerBlock = 256;
    int blocks_coef = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;

    if (type_ == TYPE_T4) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_FEAT4_Data*>(d_data_), d_leapfrog_solver_);
    } else if (type_ == TYPE_T10) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_FEAT10_Data*>(d_data_), d_leapfrog_solver_);
    } else if (type_ == TYPE_3243) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_ANCF3243_Data*>(d_data_), d_leapfrog_solver_);
    } else if (type_ == TYPE_3443) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_ANCF3443_Data*>(d_data_), d_leapfrog_solver_);
    } else if (type_ == TYPE_LDPM4) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_LDPM4_Data*>(d_data_), d_leapfrog_solver_);
    }

    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// LeapfrogSolver::OneStepLeapfrog
//
// Advances the simulation by one explicit leapfrog time step:
//   1. Compute P (stress) at all quadrature points.
//   2. Clear then assemble internal nodal forces.
//   3. Update half-step nodal velocities.
//   4. Enforce fixed-node BCs (zero velocity).
//   5. Advance nodal positions.
// ---------------------------------------------------------------------------
void LeapfrogSolver::OneStepLeapfrog() {
    cudaEvent_t start, stop;
    MOPHI_GPU_CALL(cudaEventCreate(&start));
    MOPHI_GPU_CALL(cudaEventCreate(&stop));

    constexpr int threadsPerBlock = 256;

    const int blocks_coef = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;
    const int blocks_dof = (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceProp props;
    MOPHI_GPU_CALL(cudaGetDeviceProperties(&props, 0));
    const int max_grid_stride_blocks = 32 * props.multiProcessorCount;

    int blocks_p = std::max(1, std::min((n_beam_ * n_total_qp_ + threadsPerBlock - 1) / threadsPerBlock,
                                        max_grid_stride_blocks));
    int blocks_if = std::max(1, std::min((n_beam_ * n_shape_ + threadsPerBlock - 1) / threadsPerBlock,
                                         max_grid_stride_blocks));

    auto one_step_typed = [&](auto* typed_data) {
        // Step 1: Compute stress P at all quadrature points.
        leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 2: Clear then assemble internal nodal forces.
        leapfrog_clear_f_int_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data);
        leapfrog_compute_f_int_kernel<<<blocks_if, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 3: Update half-step velocities: v += dt * M_lump^{-1} * (f_ext - f_int).
        leapfrog_update_velocity_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 4: Enforce fixed-node BCs (zero velocity). The kernel is safe
        // to launch even when no nodes are fixed (gpu_n_constraint() == 0).
        leapfrog_apply_bc_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 5: Advance positions: x += dt * v.
        leapfrog_update_position_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);
    };

    MOPHI_GPU_CALL(cudaEventRecord(start));

    if (type_ == TYPE_T4) {
        one_step_typed(static_cast<GPU_FEAT4_Data*>(d_data_));
    } else if (type_ == TYPE_T10) {
        one_step_typed(static_cast<GPU_FEAT10_Data*>(d_data_));
    } else if (type_ == TYPE_3243) {
        one_step_typed(static_cast<GPU_ANCF3243_Data*>(d_data_));
    } else if (type_ == TYPE_3443) {
        one_step_typed(static_cast<GPU_ANCF3443_Data*>(d_data_));
    } else if (type_ == TYPE_LDPM4) {
        one_step_typed(static_cast<GPU_LDPM4_Data*>(d_data_));
    }

    MOPHI_GPU_CALL(cudaEventRecord(stop));
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    float milliseconds = 0;
    MOPHI_GPU_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    MOPHI_INFO("OneStepLeapfrog kernel time: %.3f ms", milliseconds);

    MOPHI_GPU_CALL(cudaEventDestroy(start));
    MOPHI_GPU_CALL(cudaEventDestroy(stop));
}

// ---------------------------------------------------------------------------
// Explicit template instantiations for all supported element types
// ---------------------------------------------------------------------------

template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*,
                                                                                LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*,
                                                                                LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_compute_p_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_clear_f_int_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*);

template __global__ void leapfrog_compute_f_int_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_update_velocity_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_apply_bc_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_update_position_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_LDPM4_Data>(GPU_LDPM4_Data*, LeapfrogSolver*);

}  // namespace tlfea
