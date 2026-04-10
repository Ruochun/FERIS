#pragma once
/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LDPM4DataFunc.cuh
 * Brief:   CUDA device functions for the 3-DOF translational LDPM
 *          element built on a TET4 Delaunay triangulation.
 *
 *          Three free functions with the signatures expected by the
 *          templated LeapfrogSolver kernels:
 *
 *          compute_p(edge_idx, qp_idx, d_data, v_guess, dt)
 *              Resolves the relative nodal displacement onto the
 *              reference facet frame (n, m, l), computes engineering
 *              strains e_N / e_M / e_L, and stores facet tractions
 *              t_N / t_M / t_L via the linear-elastic constitutive law.
 *
 *          compute_internal_force(edge_idx, node_local, d_data)
 *              Assembles the facet traction-area product onto the two
 *              endpoint nodes (node_local = 0 → node_i, = 1 → node_j)
 *              using atomicAdd.
 *
 *          clear_internal_force(d_data)
 *              Zeroes the internal force vector; one thread per DOF.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <cmath>

#include "../materials/LDPM.cuh"
#include "LDPM4Data.cuh"

namespace tlfea {

// ---------------------------------------------------------------------------
// compute_p for LDPM4
//
// Called by leapfrog_compute_p_kernel<GPU_LDPM4_Data> with
//   edge_idx  = elem_idx   (one "element" per edge)
//   qp_idx    = 0          (one facet per edge)
//   v_guess   = half-step nodal velocity array (unused for linear-elastic)
//   dt        = time step (unused for linear-elastic, reserved for damping)
//
// Computes the linearised LDPM strains from the relative displacement
// between the two endpoint particles in the reference frame (n, m, l):
//
//   delta_u   = (x_j - x_i) - l0 * n_ref   (relative displacement)
//   e_N = (delta_u · n) / l0
//   e_M = (delta_u · m) / l0
//   e_L = (delta_u · l) / l0
//
// Stores the resulting tractions in d_data->facet_t(edge_idx, {0,1,2}).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_p(int edge_idx,
                                          int /*qp_idx*/,
                                          GPU_LDPM4_Data* d_data,
                                          const Real* /*v_guess*/,
                                          Real /*dt*/) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);

    // Current edge vector (x_j - x_i)
    Real r[3];
    r[0] = d_data->x12()(nj) - d_data->x12()(ni);
    r[1] = d_data->y12()(nj) - d_data->y12()(ni);
    r[2] = d_data->z12()(nj) - d_data->z12()(ni);

    // Reference unit normal
    Real n0 = d_data->edge_n_comp(edge_idx, 0);
    Real n1 = d_data->edge_n_comp(edge_idx, 1);
    Real n2 = d_data->edge_n_comp(edge_idx, 2);
    const Real l0 = d_data->edge_l0(edge_idx);

    // Relative displacement: delta_u = r - l0 * n_ref
    Real du[3];
    du[0] = r[0] - l0 * n0;
    du[1] = r[1] - l0 * n1;
    du[2] = r[2] - l0 * n2;

    // Engineering strains (linearised LDPM, small deformation)
    const Real inv_l0 = (l0 > Real(0)) ? Real(1) / l0 : Real(0);

    const Real e_N = (du[0] * n0 + du[1] * n1 + du[2] * n2) * inv_l0;

    const Real m0 = d_data->edge_m_comp(edge_idx, 0);
    const Real m1 = d_data->edge_m_comp(edge_idx, 1);
    const Real m2 = d_data->edge_m_comp(edge_idx, 2);
    const Real e_M = (du[0] * m0 + du[1] * m1 + du[2] * m2) * inv_l0;

    const Real l_0 = d_data->edge_lv_comp(edge_idx, 0);
    const Real l_1 = d_data->edge_lv_comp(edge_idx, 1);
    const Real l_2 = d_data->edge_lv_comp(edge_idx, 2);
    const Real e_L = (du[0] * l_0 + du[1] * l_1 + du[2] * l_2) * inv_l0;

    // Constitutive law
    Real t_N, t_M, t_L;
    ldpm_compute_traction(e_N, e_M, e_L, d_data->E_N(), d_data->E_T(), t_N, t_M, t_L);

    // Store facet tractions
    d_data->facet_t(edge_idx, 0) = t_N;
    d_data->facet_t(edge_idx, 1) = t_M;
    d_data->facet_t(edge_idx, 2) = t_L;
}

// ---------------------------------------------------------------------------
// compute_internal_force for LDPM4
//
// Called by leapfrog_compute_f_int_kernel<GPU_LDPM4_Data> with
//   edge_idx  = "element" index (one edge)
//   node_local = 0 → endpoint node i, 1 → endpoint node j
//
// Assembles the facet force contribution onto the endpoint node:
//   node j receives  +A * (t_N * n + t_M * m + t_L * l)   (restoring)
//   node i receives  -A * (t_N * n + t_M * m + t_L * l)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_internal_force(int edge_idx, int node_local, GPU_LDPM4_Data* d_data) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);
    const int global_node = (node_local == 0) ? ni : nj;
    // node j is "positive" end: force = +traction * A
    // node i is "negative" end: force = -traction * A
    const Real sign = (node_local == 0) ? Real(-1) : Real(1);

    const Real A = d_data->facet_area(edge_idx);
    const Real t_N = d_data->facet_t(edge_idx, 0);
    const Real t_M = d_data->facet_t(edge_idx, 1);
    const Real t_L = d_data->facet_t(edge_idx, 2);

    Real f[3];
    for (int k = 0; k < 3; k++) {
        f[k] = sign * A *
               (t_N * d_data->edge_n_comp(edge_idx, k) + t_M * d_data->edge_m_comp(edge_idx, k) +
                t_L * d_data->edge_lv_comp(edge_idx, k));
    }

    atomicAdd(&d_data->f_int()(global_node * 3 + 0), f[0]);
    atomicAdd(&d_data->f_int()(global_node * 3 + 1), f[1]);
    atomicAdd(&d_data->f_int()(global_node * 3 + 2), f[2]);
}

// ---------------------------------------------------------------------------
// clear_internal_force for LDPM4
//
// Zeroes d_f_int; one thread per global DOF (n_coef * 3).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void clear_internal_force(GPU_LDPM4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_data->n_coef * 3) {
        d_data->f_int()[tid] = Real(0);
    }
}

}  // namespace tlfea
