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
// Constitutive law:
//   Uses the full LDPM model with tensile fracture, compression, and friction
//   from ldpm_tet4_full_constitutive().
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

    // ── Constitutive law ───────────────────────────────────────────────────

    Real t_N, t_M, t_L, m_T, m_M, m_L;

    // Full LDPM model: incremental update with per-edge state vector.
    // Compute strain increments relative to stored accumulated strain.
    const Real prev_eps_N = d_data->edge_statev(edge_idx, 0);
    const Real prev_eps_M = d_data->edge_statev(edge_idx, 1);
    const Real prev_eps_L = d_data->edge_statev(edge_idx, 2);
    const Real d_eps_N = e_N - prev_eps_N;
    const Real d_eps_M = e_M - prev_eps_M;
    const Real d_eps_L = e_L - prev_eps_L;

    // Volumetric strain approximation: use normal strain component
    // (for a single tet this is adequate; for multi-tet meshes a
    // proper volumetric average could be computed externally).
    const Real eps_V = e_N;

    // Local copy of state vector for the constitutive update
    Real statev[LDPM_N_STATEV];
    for (int k = 0; k < LDPM_N_STATEV; ++k) {
        statev[k] = d_data->edge_statev(edge_idx, k);
    }

    ldpm_tet4_full_constitutive(d_eps_N, d_eps_M, d_eps_L, kappa_T, kappa_M, kappa_L, eps_V, l0, d_data->ldpm_params(),
                                statev, t_N, t_M, t_L, m_T, m_M, m_L);

    // Write back updated state
    for (int k = 0; k < LDPM_N_STATEV; ++k) {
        d_data->edge_statev(edge_idx, k) = statev[k];
    }

    // Maintain kappa/omega for VTK output.
    // Use effective strain as kappa proxy and crack-opening-based damage.
    d_data->edge_kappa(edge_idx) = statev[8];
    const Real crack_w = statev[11];
    const Real eff_crack = (l0 > Real(0)) ? crack_w / l0 : Real(0);
    d_data->edge_omega(edge_idx) = std::min(eff_crack / std::max(statev[8], Real(1e-30)), Real(1));

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
