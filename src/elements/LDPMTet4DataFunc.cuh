#pragma once
/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LDPMTet4DataFunc.cuh
 * Brief:   CUDA device functions for the 6-DOF LDPM element
 *          (TYPE_LDPM_TET4) built on a TET4 Delaunay triangulation.
 *
 *          Three free functions with the signatures expected by the
 *          templated LeapfrogSolver kernels:
 *
 *          compute_p(edge_idx, qp_idx, d_data, v_guess, dt)
 *              Resolves both the relative translational displacement
 *              and the difference of nodal rotation vectors onto the
 *              reference facet frame (n, m, l).  Computes 6 strains
 *              (e_N, e_M, e_L, kappa_T, kappa_M, kappa_L) and stores
 *              the resulting 6 traction/moment components.
 *
 *          compute_internal_force(edge_idx, node_local, d_data)
 *              Assembles both the translational facet force and the
 *              rotational facet moment onto the two endpoint nodes
 *              using atomicAdd.
 *
 *          clear_internal_force(d_data)
 *              Zeroes both the translational (f_int) and rotational
 *              (f_int_r) force vectors; one thread handles one DOF
 *              in each vector simultaneously.
 *==============================================================
 *==============================================================*/

#include <cuda_runtime.h>

#include <cmath>

#include "../materials/LDPM.cuh"
#include "LDPMTet4Data.cuh"

namespace feris {

// ---------------------------------------------------------------------------
// compute_p for LDPMTet4
//
// Called by leapfrog_compute_p_kernel<GPU_LDPMTet4_Data> with
//   edge_idx  = elem_idx   (one "element" per edge)
//   qp_idx    = 0          (one facet per edge)
//   v_guess   = half-step translational nodal velocity (unused here)
//   dt        = time step  (unused for quasi-static constitutive law)
//
// Translational strains:
//   delta_u = (x_j - x_i) - l0 * n_ref
//   e_N = (delta_u · n) / l0
//   e_M = (delta_u · m) / l0
//   e_L = (delta_u · l) / l0
//
// Rotational strains (linearised rotation difference):
//   delta_theta = theta_j - theta_i   (component-wise)
//   kappa_T = (delta_theta · n) / l0
//   kappa_M = (delta_theta · m) / l0
//   kappa_L = (delta_theta · l) / l0
//
// Constitutive law: Cusatis LDPM tensile-shear damage (LDPM.cuh).
//   History variable kappa and damage omega are read from and written
//   to the per-edge arrays d_kappa / d_omega.
//
// Stores results in d_data->facet_t(edge_idx, {0..5}):
//   comp 0 = t_N, 1 = t_M, 2 = t_L  (tractions)
//   comp 3 = m_T, 4 = m_M, 5 = m_L  (moments)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_p(int edge_idx,
                                          int /*qp_idx*/,
                                          GPU_LDPMTet4_Data* d_data,
                                          const Real* /*v_guess*/,
                                          Real /*dt*/) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);

    // Reference unit normal and edge length
    const Real n0 = d_data->edge_n_comp(edge_idx, 0);
    const Real n1 = d_data->edge_n_comp(edge_idx, 1);
    const Real n2 = d_data->edge_n_comp(edge_idx, 2);
    const Real l0 = d_data->edge_l0(edge_idx);
    const Real inv_l0 = (l0 > Real(0)) ? Real(1) / l0 : Real(0);

    const Real m0 = d_data->edge_m_comp(edge_idx, 0);
    const Real m1 = d_data->edge_m_comp(edge_idx, 1);
    const Real m2 = d_data->edge_m_comp(edge_idx, 2);

    const Real l_0 = d_data->edge_lv_comp(edge_idx, 0);
    const Real l_1 = d_data->edge_lv_comp(edge_idx, 1);
    const Real l_2 = d_data->edge_lv_comp(edge_idx, 2);

    // ── Translational strains ───────────────────────────────────────────────

    // Current edge vector (x_j - x_i)
    Real r[3];
    r[0] = d_data->x_cur()(nj) - d_data->x_cur()(ni);
    r[1] = d_data->y_cur()(nj) - d_data->y_cur()(ni);
    r[2] = d_data->z_cur()(nj) - d_data->z_cur()(ni);

    // Relative displacement: delta_u = r - l0 * n_ref
    const Real du0 = r[0] - l0 * n0;
    const Real du1 = r[1] - l0 * n1;
    const Real du2 = r[2] - l0 * n2;

    const Real e_N = (du0 * n0 + du1 * n1 + du2 * n2) * inv_l0;
    const Real e_M = (du0 * m0 + du1 * m1 + du2 * m2) * inv_l0;
    const Real e_L = (du0 * l_0 + du1 * l_1 + du2 * l_2) * inv_l0;

    // ── Rotational strains (linearised) ────────────────────────────────────

    // Rotation difference: delta_theta = theta_j - theta_i
    const Real dt0 = d_data->rot_x_cur()(nj) - d_data->rot_x_cur()(ni);
    const Real dt1 = d_data->rot_y_cur()(nj) - d_data->rot_y_cur()(ni);
    const Real dt2 = d_data->rot_z_cur()(nj) - d_data->rot_z_cur()(ni);

    const Real kappa_T = (dt0 * n0 + dt1 * n1 + dt2 * n2) * inv_l0;
    const Real kappa_M = (dt0 * m0 + dt1 * m1 + dt2 * m2) * inv_l0;
    const Real kappa_L = (dt0 * l_0 + dt1 * l_1 + dt2 * l_2) * inv_l0;

    // ── Cusatis LDPM full constitutive update ─────────────────────────────────

    // Build params struct from device scalars
    LDPMParams params;
    params.E_N = d_data->E_N();
    params.E_T = d_data->E_T();
    params.E_kT = d_data->E_kT();
    params.E_kM = d_data->E_kM();
    params.E_kL = d_data->E_kL();
    params.sigma_t = d_data->sigma_t();
    params.l_t = d_data->l_t();
    params.r_st = d_data->r_st();
    params.n_t = d_data->n_t();
    params.sigma_c0 = d_data->sigma_c0();
    params.Hc0_ratio = d_data->Hc0_ratio();
    params.kappa_c0 = d_data->kappa_c0();
    params.Ed_ratio = d_data->Ed_ratio();
    params.mu_0 = d_data->mu_0();
    params.mu_inf = d_data->mu_inf();
    params.sigma_N0 = d_data->sigma_N0();

    // Build state from per-edge arrays
    LDPMState state;
    state.kappa = d_data->edge_kappa(edge_idx);
    state.omega = Real(0);
    state.e_N_comp = d_data->edge_e_N_comp(edge_idx);

    Real t_N, t_M, t_L, m_T, m_M, m_L;

    ldpm_tet4_constitutive_update(e_N, e_M, e_L, kappa_T, kappa_M, kappa_L, params, state, t_N, t_M, t_L, m_T, m_M,
                                  m_L);

    // Write back updated state
    d_data->edge_kappa(edge_idx) = state.kappa;
    d_data->edge_omega(edge_idx) = state.omega;
    d_data->edge_e_N_comp(edge_idx) = state.e_N_comp;

    // Store facet tractions [0..2] and moments [3..5]
    d_data->facet_t(edge_idx, 0) = t_N;
    d_data->facet_t(edge_idx, 1) = t_M;
    d_data->facet_t(edge_idx, 2) = t_L;
    d_data->facet_t(edge_idx, 3) = m_T;
    d_data->facet_t(edge_idx, 4) = m_M;
    d_data->facet_t(edge_idx, 5) = m_L;
}

// ---------------------------------------------------------------------------
// compute_internal_force for LDPMTet4
//
// Called by leapfrog_compute_f_int_kernel<GPU_LDPMTet4_Data> with
//   edge_idx   = "element" index (one edge)
//   node_local = 0 → endpoint node i, 1 → endpoint node j
//
// Translational force assembly:
//   node j receives  +A * (t_N * n + t_M * m + t_L * l)
//   node i receives  -A * (...)
//
// Rotational moment assembly:
//   node j receives  +A * (m_T * n + m_M * m + m_L * l)
//   node i receives  -A * (...)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_internal_force(int edge_idx, int node_local, GPU_LDPMTet4_Data* d_data) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);
    const int global_node = (node_local == 0) ? ni : nj;
    const Real sign = (node_local == 0) ? Real(-1) : Real(1);

    const Real A = d_data->facet_area(edge_idx);

    const Real t_N = d_data->facet_t(edge_idx, 0);
    const Real t_M = d_data->facet_t(edge_idx, 1);
    const Real t_L = d_data->facet_t(edge_idx, 2);
    const Real m_T = d_data->facet_t(edge_idx, 3);
    const Real m_M = d_data->facet_t(edge_idx, 4);
    const Real m_L = d_data->facet_t(edge_idx, 5);

    // Assemble translational internal force
    for (int k = 0; k < 3; k++) {
        const Real f_k = sign * A *
                         (t_N * d_data->edge_n_comp(edge_idx, k) + t_M * d_data->edge_m_comp(edge_idx, k) +
                          t_L * d_data->edge_lv_comp(edge_idx, k));
        atomicAdd(&d_data->f_int()(global_node * 3 + k), f_k);
    }

    // Assemble rotational internal moment
    for (int k = 0; k < 3; k++) {
        const Real m_k = sign * A *
                         (m_T * d_data->edge_n_comp(edge_idx, k) + m_M * d_data->edge_m_comp(edge_idx, k) +
                          m_L * d_data->edge_lv_comp(edge_idx, k));
        atomicAdd(&d_data->f_int_r()(global_node * 3 + k), m_k);
    }
}

// ---------------------------------------------------------------------------
// clear_internal_force for LDPMTet4
//
// Zeroes both translational (f_int) and rotational (f_int_r) force
// vectors.  Each thread (one per translational DOF, n_nodes * 3 total)
// clears the corresponding entry in both arrays.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void clear_internal_force(GPU_LDPMTet4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < d_data->n_nodes * 3) {
        d_data->f_int()[tid] = Real(0);
        d_data->f_int_r()[tid] = Real(0);
    }
}

}  // namespace feris
