/*==============================================================
 *==============================================================
 * Project: FERIS
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 * File:    LeapfrogSolver.cu
 * Brief:   Implements the explicit central-difference (leapfrog) time
 *          integrator. Defines GPU kernels that compute the lumped mass
 *          from the CSR consistent mass matrix, evaluate element stress /
 *          facet tractions, assemble internal forces, update half-step nodal
 *          velocities, enforce fixed-node boundary conditions, and advance
 *          nodal positions. Supports ANCF3243, ANCF3443, FEAT10, FEAT4,
 *          and LDPM_TET4 element types.
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
#include "../elements/LDPMTet4Data.cuh"
#include "../elements/LDPMTet4DataFunc.cuh"
#include "LeapfrogSolver.cuh"

namespace feris {

// ---------------------------------------------------------------------------
// leapfrog_compute_lumped_mass_kernel
//
// Computes the lumped (row-sum) nodal mass from the pre-assembled CSR
// consistent mass matrix stored in the element data. One thread per node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_compute_lumped_mass_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int node_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_i >= d_solver->get_n_coef())
        return;

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
    if (tid >= d_solver->get_n_coef() * 3)
        return;

    int node_i = tid / 3;
    Real dt = d_solver->solver_time_step();
    Real mass_val = d_solver->mass_lump()(node_i);
    // Guard against orphan nodes (zero lumped mass) – they contribute no dynamics.
    if (mass_val <= Real(0))
        return;
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
    if (tid >= n_fixed)
        return;

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
    if (tid >= d_solver->get_n_coef())
        return;

    Real dt = d_solver->solver_time_step();
    d_data->x12()(tid) += dt * d_solver->v()(tid * 3 + 0);
    d_data->y12()(tid) += dt * d_solver->v()(tid * 3 + 1);
    d_data->z12()(tid) += dt * d_solver->v()(tid * 3 + 2);
}

// ---------------------------------------------------------------------------
// leapfrog_compute_lumped_mass_ldpm_tet4_kernel
//
// Specialised mass setup for TYPE_LDPM_TET4.  In addition to computing
// the translational lumped mass from the CSR (same as the generic kernel),
// also copies the per-node rotational inertia from the element data into
// the second half of the solver's mass array (mass_inertia slot).
// ---------------------------------------------------------------------------
__global__ void leapfrog_compute_lumped_mass_ldpm_tet4_kernel(GPU_LDPMTet4_Data* d_data, LeapfrogSolver* d_solver) {
    int node_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_i >= d_solver->get_n_coef())
        return;

    // Translational mass: row-sum of the diagonal CSR
    const int* __restrict__ offsets = d_data->csr_offsets();
    const Real* __restrict__ values = d_data->csr_values();
    Real m_i = 0.0;
    for (int idx = offsets[node_i]; idx < offsets[node_i + 1]; idx++) {
        m_i += values[idx];
    }
    d_solver->mass_lump()(node_i) = m_i;

    // Rotational inertia: precomputed in CalcMassMatrix
    d_solver->mass_inertia()(node_i) = d_data->I_lump(node_i);
}

// ---------------------------------------------------------------------------
// leapfrog_update_velocity_rot_kernel
//
// Rotational velocity update for TYPE_LDPM_TET4:
//   omega_{n+1/2} = omega_{n-1/2} + dt * I_lump^{-1} * (m_ext - m_int)
//
// One thread per rotational DOF (node * 3 + direction).
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_update_velocity_rot_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef() * 3)
        return;

    int node_i = tid / 3;
    Real dt = d_solver->solver_time_step();
    Real inertia_val = d_solver->mass_inertia()(node_i);
    if (inertia_val <= Real(0))
        return;
    Real inv_inertia = Real(1) / inertia_val;
    Real net_moment = d_data->f_ext_r()(tid) - d_data->f_int_r()(tid);

    d_solver->v_rot()(tid) += dt * inv_inertia * net_moment;
}

// ---------------------------------------------------------------------------
// leapfrog_apply_bc_ldpm_tet4_kernel
//
// Enforces fixed-node BCs for TYPE_LDPM_TET4 by zeroing all 6 velocity
// DOFs (3 translational + 3 rotational) at each fixed node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_apply_bc_ldpm_tet4_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_fixed = d_data->gpu_n_constraint() / 6;
    if (tid >= n_fixed)
        return;

    int node_idx = d_data->fixed_nodes()(tid);
    // Translational velocities
    d_solver->v()(node_idx * 3 + 0) = Real(0);
    d_solver->v()(node_idx * 3 + 1) = Real(0);
    d_solver->v()(node_idx * 3 + 2) = Real(0);
    // Rotational velocities
    d_solver->v_rot()(node_idx * 3 + 0) = Real(0);
    d_solver->v_rot()(node_idx * 3 + 1) = Real(0);
    d_solver->v_rot()(node_idx * 3 + 2) = Real(0);
}

// ---------------------------------------------------------------------------
// leapfrog_apply_prescribed_vel_kernel
//
// Overwrites the translational velocity of a specified set of nodes with
// prescribed (constant) values.  Called after the fixed-node BC zeroing
// in both OneStepLeapfrog and HalfKickImpl so that driven nodes always
// carry exactly their target velocity into the position update.
//
// d_v     – the solver's flat translational velocity array (length 3*n_coef).
// d_nodes – device array of node indices to update (length n).
// d_vel   – device array of [vx, vy, vz] values per node (length 3*n).
// n       – number of prescribed-velocity nodes.
//
// Not templated: operates only on the solver velocity array.
// One thread per prescribed node.
// ---------------------------------------------------------------------------
__global__ void leapfrog_apply_prescribed_vel_kernel(Real* d_v, const int* d_nodes, const Real* d_vel, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    const int node_idx = d_nodes[tid];
    d_v[node_idx * 3 + 0] = d_vel[tid * 3 + 0];
    d_v[node_idx * 3 + 1] = d_vel[tid * 3 + 1];
    d_v[node_idx * 3 + 2] = d_vel[tid * 3 + 2];
}


// ---------------------------------------------------------------------------
// leapfrog_half_kick_kernel
//   v += sign * (dt/2) * M_lump^{-1} * (f_ext - f_int)
//
// sign = -1: backward half-kick — InitialHalfKick, converts v_0 → v_{-1/2}
// sign = +1: forward  half-kick — FinalHalfKick,   converts v_{N-1/2} → v_N
//
// One thread per global DOF (node * 3 + direction).
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_half_kick_kernel(ElemType* d_data, LeapfrogSolver* d_solver, Real sign) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef() * 3)
        return;

    int node_i = tid / 3;
    Real half_dt = Real(0.5) * d_solver->solver_time_step();
    Real mass_val = d_solver->mass_lump()(node_i);
    if (mass_val <= Real(0))
        return;
    Real inv_mass = Real(1) / mass_val;
    Real net_force = d_data->f_ext()(tid) - d_data->f_int()(tid);

    d_solver->v()(tid) += sign * half_dt * inv_mass * net_force;
}

// ---------------------------------------------------------------------------
// leapfrog_half_kick_rot_kernel
//
// Half-step rotational velocity update for TYPE_LDPM_TET4:
//   omega += sign * (dt/2) * I_lump^{-1} * (m_ext - m_int)
//
// sign = -1: backward half-kick (InitialHalfKick)
// sign = +1: forward  half-kick (FinalHalfKick)
//
// One thread per rotational DOF (node * 3 + direction).
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_half_kick_rot_kernel(ElemType* d_data, LeapfrogSolver* d_solver, Real sign) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef() * 3)
        return;

    int node_i = tid / 3;
    Real half_dt = Real(0.5) * d_solver->solver_time_step();
    Real inertia_val = d_solver->mass_inertia()(node_i);
    if (inertia_val <= Real(0))
        return;
    Real inv_inertia = Real(1) / inertia_val;
    Real net_moment = d_data->f_ext_r()(tid) - d_data->f_int_r()(tid);

    d_solver->v_rot()(tid) += sign * half_dt * inv_inertia * net_moment;
}

// ---------------------------------------------------------------------------
// leapfrog_update_rotation_kernel
//
// Full-step rotation update for TYPE_LDPM_TET4:
//   theta_{n+1} = theta_n + dt * omega_{n+1/2}
//
// One thread per node.
// ---------------------------------------------------------------------------
template <typename ElemType>
__global__ void leapfrog_update_rotation_kernel(ElemType* d_data, LeapfrogSolver* d_solver) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_solver->get_n_coef())
        return;

    Real dt = d_solver->solver_time_step();
    d_data->rx12()(tid) += dt * d_solver->v_rot()(tid * 3 + 0);
    d_data->ry12()(tid) += dt * d_solver->v_rot()(tid * 3 + 1);
    d_data->rz12()(tid) += dt * d_solver->v_rot()(tid * 3 + 2);
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
    da_mass_lump_.ToDevice();

    MOPHI_GPU_CALL(cudaMemcpy(d_leapfrog_solver_, this, sizeof(LeapfrogSolver), cudaMemcpyHostToDevice));

    constexpr int threadsPerBlock = 256;
    int blocks_coef = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;

    if (type_ == TYPE_T4) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(static_cast<GPU_FEAT4_Data*>(d_data_),
                                                                              d_leapfrog_solver_);
    } else if (type_ == TYPE_T10) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(static_cast<GPU_FEAT10_Data*>(d_data_),
                                                                              d_leapfrog_solver_);
    } else if (type_ == TYPE_3243) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(static_cast<GPU_ANCF3243_Data*>(d_data_),
                                                                              d_leapfrog_solver_);
    } else if (type_ == TYPE_3443) {
        leapfrog_compute_lumped_mass_kernel<<<blocks_coef, threadsPerBlock>>>(static_cast<GPU_ANCF3443_Data*>(d_data_),
                                                                              d_leapfrog_solver_);
    } else if (type_ == TYPE_LDPM_TET4) {
        leapfrog_compute_lumped_mass_ldpm_tet4_kernel<<<blocks_coef, threadsPerBlock>>>(
            static_cast<GPU_LDPMTet4_Data*>(d_data_), d_leapfrog_solver_);
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

    int blocks_p =
        std::max(1, std::min((n_beam_ * n_total_qp_ + threadsPerBlock - 1) / threadsPerBlock, max_grid_stride_blocks));
    int blocks_if =
        std::max(1, std::min((n_beam_ * n_shape_ + threadsPerBlock - 1) / threadsPerBlock, max_grid_stride_blocks));

    auto one_step_typed = [&](auto* typed_data) {
        // Step 1: Compute stress P at all quadrature points.
        leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 2: Clear then assemble internal nodal forces.
        leapfrog_clear_f_int_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data);
        leapfrog_compute_f_int_kernel<<<blocks_if, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 3: Update half-step velocities: v += dt * M_lump^{-1} * (f_ext - f_int).
        leapfrog_update_velocity_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 4a: Enforce fixed-node BCs (zero velocity). The kernel is safe
        // to launch even when no nodes are fixed (gpu_n_constraint() == 0).
        leapfrog_apply_bc_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 4b: Re-impose prescribed (non-zero) velocities on driven nodes.
        if (n_prescribed_vel_ > 0) {
            const int blocks_pv = (n_prescribed_vel_ + threadsPerBlock - 1) / threadsPerBlock;
            leapfrog_apply_prescribed_vel_kernel<<<blocks_pv, threadsPerBlock>>>(
                d_v_, d_pvel_nodes_, d_pvel_values_, n_prescribed_vel_);
        }

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
    } else if (type_ == TYPE_LDPM_TET4) {
        auto* typed_data = static_cast<GPU_LDPMTet4_Data*>(d_data_);

        // Step 1: Compute tractions and moments at all facets.
        leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 2: Clear then assemble internal forces and moments.
        leapfrog_clear_f_int_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data);
        leapfrog_compute_f_int_kernel<<<blocks_if, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 3a: Update half-step translational velocities.
        leapfrog_update_velocity_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 3b: Update half-step rotational velocities.
        leapfrog_update_velocity_rot_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 4a: Enforce fixed-node BCs (zero all 6 velocity DOFs).
        leapfrog_apply_bc_ldpm_tet4_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 4b: Re-impose prescribed (non-zero) translational velocities.
        if (n_prescribed_vel_ > 0) {
            const int blocks_pv = (n_prescribed_vel_ + threadsPerBlock - 1) / threadsPerBlock;
            leapfrog_apply_prescribed_vel_kernel<<<blocks_pv, threadsPerBlock>>>(
                d_v_, d_pvel_nodes_, d_pvel_values_, n_prescribed_vel_);
        }

        // Step 5a: Advance translational positions.
        leapfrog_update_position_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Step 5b: Advance rotational positions.
        leapfrog_update_rotation_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);
    }

    MOPHI_GPU_CALL(cudaEventRecord(stop));
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    float milliseconds = 0;
    MOPHI_GPU_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    // MOPHI_STATUS("LEAPFROG_KERNEL_TIME", "OneStepLeapfrog kernel time: %.3f ms", milliseconds);

    MOPHI_GPU_CALL(cudaEventDestroy(start));
    MOPHI_GPU_CALL(cudaEventDestroy(stop));
}

// ---------------------------------------------------------------------------
// LeapfrogSolver::HalfKickImpl
//
// Shared implementation for InitialHalfKick() and FinalHalfKick().
//
// Computes forces at the current configuration, then nudges every
// nodal velocity by  sign * (dt/2) * M_lump^{-1} * (f_ext - f_int).
//
//   sign = -1  →  backward half-kick  (InitialHalfKick)
//   sign = +1  →  forward  half-kick  (FinalHalfKick)
//
// Fixed-node BCs are enforced (velocity zeroed) and prescribed-velocity BCs
// are re-imposed after the kick.  No position update is performed.
// ---------------------------------------------------------------------------
void LeapfrogSolver::HalfKickImpl(Real sign) {
    constexpr int threadsPerBlock = 256;

    const int blocks_coef = (n_coef_ + threadsPerBlock - 1) / threadsPerBlock;
    const int blocks_dof = (n_coef_ * 3 + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceProp props;
    MOPHI_GPU_CALL(cudaGetDeviceProperties(&props, 0));
    const int max_grid_stride_blocks = 32 * props.multiProcessorCount;

    int blocks_p =
        std::max(1, std::min((n_beam_ * n_total_qp_ + threadsPerBlock - 1) / threadsPerBlock, max_grid_stride_blocks));
    int blocks_if =
        std::max(1, std::min((n_beam_ * n_shape_ + threadsPerBlock - 1) / threadsPerBlock, max_grid_stride_blocks));

    auto half_kick_typed = [&](auto* typed_data) {
        // Recompute forces at the current configuration.
        leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);
        leapfrog_clear_f_int_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data);
        leapfrog_compute_f_int_kernel<<<blocks_if, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Apply the signed half-step velocity nudge.
        leapfrog_half_kick_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_, sign);

        // Enforce fixed-node BCs (zero velocity).
        leapfrog_apply_bc_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Re-impose prescribed (non-zero) velocities so driven nodes keep
        // their target velocity after the half-kick.
        if (n_prescribed_vel_ > 0) {
            const int blocks_pv = (n_prescribed_vel_ + threadsPerBlock - 1) / threadsPerBlock;
            leapfrog_apply_prescribed_vel_kernel<<<blocks_pv, threadsPerBlock>>>(
                d_v_, d_pvel_nodes_, d_pvel_values_, n_prescribed_vel_);
        }
    };

    if (type_ == TYPE_T4) {
        half_kick_typed(static_cast<GPU_FEAT4_Data*>(d_data_));
    } else if (type_ == TYPE_T10) {
        half_kick_typed(static_cast<GPU_FEAT10_Data*>(d_data_));
    } else if (type_ == TYPE_3243) {
        half_kick_typed(static_cast<GPU_ANCF3243_Data*>(d_data_));
    } else if (type_ == TYPE_3443) {
        half_kick_typed(static_cast<GPU_ANCF3443_Data*>(d_data_));
    } else if (type_ == TYPE_LDPM_TET4) {
        auto* typed_data = static_cast<GPU_LDPMTet4_Data*>(d_data_);

        // Recompute forces and moments at the current configuration.
        leapfrog_compute_p_kernel<<<blocks_p, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);
        leapfrog_clear_f_int_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data);
        leapfrog_compute_f_int_kernel<<<blocks_if, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Apply the signed half-step translational and rotational velocity nudges.
        leapfrog_half_kick_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_, sign);
        leapfrog_half_kick_rot_kernel<<<blocks_dof, threadsPerBlock>>>(typed_data, d_leapfrog_solver_, sign);

        // Enforce fixed-node BCs (zero all 6 DOFs per fixed node).
        leapfrog_apply_bc_ldpm_tet4_kernel<<<blocks_coef, threadsPerBlock>>>(typed_data, d_leapfrog_solver_);

        // Re-impose prescribed translational velocities on driven nodes.
        if (n_prescribed_vel_ > 0) {
            const int blocks_pv = (n_prescribed_vel_ + threadsPerBlock - 1) / threadsPerBlock;
            leapfrog_apply_prescribed_vel_kernel<<<blocks_pv, threadsPerBlock>>>(
                d_v_, d_pvel_nodes_, d_pvel_values_, n_prescribed_vel_);
        }
    }

    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// LeapfrogSolver::InitialHalfKick
//
// Converts full-step initial velocities v_0 stored in da_v_ to half-step
// velocities v_{-1/2} needed by the staggered leapfrog:
//
//   v_{-1/2} = v_0 - (dt/2) * M_lump^{-1} * (f_ext - f_int(x_0))
//
// Call this after Setup() when da_v_ has been seeded with non-zero physical
// initial velocities v_0.  For rest-start simulations (v_0 = 0, f_0 = 0)
// this is a no-op and need not be called.
// ---------------------------------------------------------------------------
void LeapfrogSolver::InitialHalfKick() {
    HalfKickImpl(Real(-1));
}

// ---------------------------------------------------------------------------
// LeapfrogSolver::FinalHalfKick
//
// Converts the half-step velocity v_{N-1/2} stored in da_v_ after the last
// Solve() call to the synchronized full-step velocity at the end of the
// simulation:
//
//   v_N = v_{N-1/2} + (dt/2) * M_lump^{-1} * (f_ext - f_int(x_N))
//
// Call this after the last Solve() when a velocity output synchronized with
// the final position x_N is required (e.g. kinetic-energy reporting).
// It is never needed for the regular time-stepping loop.
// ---------------------------------------------------------------------------
void LeapfrogSolver::FinalHalfKick() {
    HalfKickImpl(Real(+1));
}

// ---------------------------------------------------------------------------
// Explicit template instantiations for all supported element types
// ---------------------------------------------------------------------------

template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_lumped_mass_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_compute_p_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_p_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_clear_f_int_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*);
template __global__ void leapfrog_clear_f_int_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*);

template __global__ void leapfrog_compute_f_int_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_compute_f_int_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_update_velocity_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_velocity_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_apply_bc_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);

template __global__ void leapfrog_update_position_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_position_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);

// Kernels exclusive to TYPE_LDPM_TET4
template __global__ void leapfrog_update_velocity_rot_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_apply_bc_ldpm_tet4_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);
template __global__ void leapfrog_update_rotation_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*);

// Half-kick kernels (all element types)
template __global__ void leapfrog_half_kick_kernel<GPU_ANCF3243_Data>(GPU_ANCF3243_Data*, LeapfrogSolver*, Real);
template __global__ void leapfrog_half_kick_kernel<GPU_ANCF3443_Data>(GPU_ANCF3443_Data*, LeapfrogSolver*, Real);
template __global__ void leapfrog_half_kick_kernel<GPU_FEAT10_Data>(GPU_FEAT10_Data*, LeapfrogSolver*, Real);
template __global__ void leapfrog_half_kick_kernel<GPU_FEAT4_Data>(GPU_FEAT4_Data*, LeapfrogSolver*, Real);
template __global__ void leapfrog_half_kick_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*, Real);

// Rotational half-kick kernel (TYPE_LDPM_TET4 only)
template __global__ void leapfrog_half_kick_rot_kernel<GPU_LDPMTet4_Data>(GPU_LDPMTet4_Data*, LeapfrogSolver*, Real);

}  // namespace feris
