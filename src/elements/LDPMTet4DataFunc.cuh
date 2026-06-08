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
 *              Thin wrapper calling compute_ldpm_facet_strain_and_stress.
 *
 *          compute_ldpm_facet_strain_and_stress(edge_idx, d_data)
 *              Resolves both the relative translational displacement
 *              and the difference of nodal rotation vectors onto the
 *              reference facet frame (n, m, l).  Computes 6 strains
 *              (e_N, e_M, e_L, kappa_T, kappa_M, kappa_L) and evaluates
 *              the full LDPM constitutive law to produce 6 traction/moment
 *              components.
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

#include <algorithm>
#include <cmath>

#include "../materials/LDPM.cuh"
#include "LDPMTet4Data.cuh"

namespace feris {

struct LDPMFacetKinematics {
    Real l0;
    Real e_N;
    Real e_M;
    Real e_L;
    Real kappa_T;
    Real kappa_M;
    Real kappa_L;
};

__device__ __forceinline__ LDPMFacetKinematics compute_ldpm_facet_kinematics(int edge_idx, GPU_LDPMTet4_Data* d_data) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);

    const Real n0 = d_data->edge_n_comp(edge_idx, 0);
    const Real n1 = d_data->edge_n_comp(edge_idx, 1);
    const Real n2 = d_data->edge_n_comp(edge_idx, 2);
    const Real l0 = d_data->edge_l0(edge_idx);
    const Real inv_l0 = (l0 > Real(0)) ? Real(1) / l0 : Real(0);

    const Real m0 = d_data->edge_m_comp(edge_idx, 0);
    const Real m1 = d_data->edge_m_comp(edge_idx, 1);
    const Real m2 = d_data->edge_m_comp(edge_idx, 2);

    const Real lvec0 = d_data->edge_lv_comp(edge_idx, 0);
    const Real lvec1 = d_data->edge_lv_comp(edge_idx, 1);
    const Real lvec2 = d_data->edge_lv_comp(edge_idx, 2);

    const Real du0 = d_data->x_cur()(nj) - d_data->x_cur()(ni) - l0 * n0;
    const Real du1 = d_data->y_cur()(nj) - d_data->y_cur()(ni) - l0 * n1;
    const Real du2 = d_data->z_cur()(nj) - d_data->z_cur()(ni) - l0 * n2;

    const Real dt0 = d_data->rot_x_cur()(nj) - d_data->rot_x_cur()(ni);
    const Real dt1 = d_data->rot_y_cur()(nj) - d_data->rot_y_cur()(ni);
    const Real dt2 = d_data->rot_z_cur()(nj) - d_data->rot_z_cur()(ni);

    LDPMFacetKinematics kin;
    kin.l0 = l0;
    kin.e_N = (du0 * n0 + du1 * n1 + du2 * n2) * inv_l0;
    kin.e_M = (du0 * m0 + du1 * m1 + du2 * m2) * inv_l0;
    kin.e_L = (du0 * lvec0 + du1 * lvec1 + du2 * lvec2) * inv_l0;
    kin.kappa_T = (dt0 * n0 + dt1 * n1 + dt2 * n2) * inv_l0;
    kin.kappa_M = (dt0 * m0 + dt1 * m1 + dt2 * m2) * inv_l0;
    kin.kappa_L = (dt0 * lvec0 + dt1 * lvec1 + dt2 * lvec2) * inv_l0;
    return kin;
}

// ---------------------------------------------------------------------------
// compute_ldpm_facet_strain_and_stress
//
// Core LDPM constitutive computation for a single edge (facet strut).
// Resolves the relative translational displacement and rotation difference
// onto the reference facet frame (n, m, l), computes 6 strains
// (e_N, e_M, e_L, kappa_T, kappa_M, kappa_L), and evaluates the full LDPM
// constitutive law (tensile fracture, compression with volumetric strain
// averaging, and frictional shear).
//
// Stores results in d_data->facet_t(edge_idx, {0..5}):
//   comp 0 = t_N, 1 = t_M, 2 = t_L  (tractions)
//   comp 3 = m_T, 4 = m_M, 5 = m_L  (moments)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_ldpm_facet_strain_and_stress(int edge_idx, GPU_LDPMTet4_Data* d_data) {
    const LDPMFacetKinematics kin = compute_ldpm_facet_kinematics(edge_idx, d_data);

    // ── Constitutive law ───────────────────────────────────────────────────

    Real t_N, t_M, t_L, m_T, m_M, m_L;

    // Full LDPM model: incremental update with per-edge state vector.
    // Compute strain increments relative to stored accumulated strain.
    const Real prev_eps_N = d_data->edge_statev(edge_idx, 0);
    const Real prev_eps_M = d_data->edge_statev(edge_idx, 1);
    const Real prev_eps_L = d_data->edge_statev(edge_idx, 2);
    const Real d_eps_N = kin.e_N - prev_eps_N;
    const Real d_eps_M = kin.e_M - prev_eps_M;
    const Real d_eps_L = kin.e_L - prev_eps_L;

    // Volumetric strain: precomputed as the average of the volumetric
    // strains of the tets sharing this edge (matches CPU reference where
    // eps_V = (V_cur - V_ref) / (3 * V_ref) per tet element).
    const Real eps_V = d_data->edge_vol_strain(edge_idx);

    // Local copy of state vector for the constitutive update
    Real statev[LDPM_N_STATEV];
    for (int k = 0; k < LDPM_N_STATEV; ++k) {
        statev[k] = d_data->edge_statev(edge_idx, k);
    }

    ldpm_tet4_full_constitutive(d_eps_N, d_eps_M, d_eps_L, kin.kappa_T, kin.kappa_M, kin.kappa_L, eps_V, kin.l0,
                                d_data->ldpm_params(), statev, t_N, t_M, t_L, m_T, m_M, m_L);

    // Write back updated state
    for (int k = 0; k < LDPM_N_STATEV; ++k) {
        d_data->edge_statev(edge_idx, k) = statev[k];
    }

    // Maintain kappa/omega for VTK output.
    // Use effective strain as kappa proxy and crack-opening-based damage.
    d_data->edge_kappa(edge_idx) = statev[8];
    const Real crack_w = statev[11];
    const Real eff_crack = (kin.l0 > Real(0)) ? crack_w / kin.l0 : Real(0);
    d_data->edge_omega(edge_idx) = std::min(eff_crack / std::max(statev[8], Real(1e-30)), Real(1));

    // Store facet tractions [0..2] and moments [3..5]
    d_data->facet_t(edge_idx, LDPM_FACET_T_N) = t_N;
    d_data->facet_t(edge_idx, LDPM_FACET_T_M) = t_M;
    d_data->facet_t(edge_idx, LDPM_FACET_T_L) = t_L;
    d_data->facet_t(edge_idx, LDPM_FACET_M_T) = m_T;
    d_data->facet_t(edge_idx, LDPM_FACET_M_M) = m_M;
    d_data->facet_t(edge_idx, LDPM_FACET_M_L) = m_L;
}

// ---------------------------------------------------------------------------
// compute_p for LDPMTet4
//
// Thin wrapper satisfying the solver template interface
// (leapfrog_compute_p_kernel calls compute_p for each element type).
// Delegates to compute_ldpm_facet_strain_and_stress.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_p(int edge_idx,
                                          int /*qp_idx*/,
                                          GPU_LDPMTet4_Data* d_data,
                                          const Real* /*v_guess*/,
                                          Real /*dt*/) {
    compute_ldpm_facet_strain_and_stress(edge_idx, d_data);
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
//   node j receives  +A * (r_j × t + m_T * n + m_M * m + m_L * l)
//   node i receives  -A * (r_i × t + m_T * n + m_M * m + m_L * l)
// where r_i/r_j point from each endpoint to the interaction-facet center.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void compute_internal_force(int edge_idx, int node_local, GPU_LDPMTet4_Data* d_data) {
    const int ni = d_data->edge_node(edge_idx, 0);
    const int nj = d_data->edge_node(edge_idx, 1);
    const int global_node = (node_local == 0) ? ni : nj;
    const Real sign = (node_local == 0) ? Real(-1) : Real(1);

    const Real A = d_data->facet_area(edge_idx);

    const Real t_N = d_data->facet_t(edge_idx, LDPM_FACET_T_N);
    const Real t_M = d_data->facet_t(edge_idx, LDPM_FACET_T_M);
    const Real t_L = d_data->facet_t(edge_idx, LDPM_FACET_T_L);
    const Real m_T = d_data->facet_t(edge_idx, LDPM_FACET_M_T);
    const Real m_M = d_data->facet_t(edge_idx, LDPM_FACET_M_M);
    const Real m_L = d_data->facet_t(edge_idx, LDPM_FACET_M_L);

    Real f_vec[3];
    Real couple_vec[3];

    // Assemble translational internal force
    for (int k = 0; k < 3; k++) {
        f_vec[k] = t_N * d_data->edge_n_comp(edge_idx, k) + t_M * d_data->edge_m_comp(edge_idx, k) +
                   t_L * d_data->edge_lv_comp(edge_idx, k);
        couple_vec[k] = m_T * d_data->edge_n_comp(edge_idx, k) + m_M * d_data->edge_m_comp(edge_idx, k) +
                        m_L * d_data->edge_lv_comp(edge_idx, k);
        const Real f_k = sign * A * f_vec[k];
        atomicAdd(&d_data->f_int()(global_node * 3 + k), f_k);
    }

    const Real node_x = (node_local == 0) ? d_data->x_cur()(ni) : d_data->x_cur()(nj);
    const Real node_y = (node_local == 0) ? d_data->y_cur()(ni) : d_data->y_cur()(nj);
    const Real node_z = (node_local == 0) ? d_data->z_cur()(ni) : d_data->z_cur()(nj);

    const Real r0 = d_data->edge_center_comp(edge_idx, 0) - node_x;
    const Real r1 = d_data->edge_center_comp(edge_idx, 1) - node_y;
    const Real r2 = d_data->edge_center_comp(edge_idx, 2) - node_z;

    const Real torque0 = r1 * f_vec[2] - r2 * f_vec[1];
    const Real torque1 = r2 * f_vec[0] - r0 * f_vec[2];
    const Real torque2 = r0 * f_vec[1] - r1 * f_vec[0];

    atomicAdd(&d_data->f_int_r()(global_node * 3 + 0), sign * A * (torque0 + couple_vec[0]));
    atomicAdd(&d_data->f_int_r()(global_node * 3 + 1), sign * A * (torque1 + couple_vec[1]));
    atomicAdd(&d_data->f_int_r()(global_node * 3 + 2), sign * A * (torque2 + couple_vec[2]));
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
