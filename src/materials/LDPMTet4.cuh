/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LDPMTet4.cuh
 * Brief:   Linear-elastic facet constitutive law for the 6-DOF
 *          LDPM element built on TET4 (TYPE_LDPM_TET4).
 *
 *          Extends the 3-DOF LDPM constitutive law with three
 *          rotational strain components resolved on the local
 *          facet frame (n, m, l):
 *
 *            Translational strains → tractions:
 *              t_N = E_N * e_N   (normal)
 *              t_M = E_T * e_M   (shear, direction m)
 *              t_L = E_T * e_L   (shear, direction l)
 *
 *            Rotational strains → moments:
 *              kappa_T = (Δθ · n) / l0   → m_T = E_kT * kappa_T (twist)
 *              kappa_M = (Δθ · m) / l0   → m_M = E_kM * kappa_M (bending, m)
 *              kappa_L = (Δθ · l) / l0   → m_L = E_kL * kappa_L (bending, l)
 *
 *          where Δθ = θ_j − θ_i is the difference of nodal rotation
 *          vectors (linearised / small rotations).
 *==============================================================
 *==============================================================*/

#pragma once

#include "../types.h"

namespace tlfea {

// Linear-elastic LDPM-TET4 constitutive law.
//
// Inputs:
//   e_N, e_M, e_L         – translational engineering strains
//   kappa_T, kappa_M, kappa_L  – rotational strains (1/length)
//   E_N, E_T              – normal and shear translational moduli (Pa)
//   E_kT, E_kM, E_kL      – twist and bending moduli (Pa·m²)
//
// Outputs:
//   t_N, t_M, t_L         – facet tractions  (Pa)
//   m_T, m_M, m_L         – facet moments per unit area (Pa·m)
__device__ __forceinline__ void ldpm_tet4_compute_traction(Real e_N,
                                                           Real e_M,
                                                           Real e_L,
                                                           Real kappa_T,
                                                           Real kappa_M,
                                                           Real kappa_L,
                                                           Real E_N,
                                                           Real E_T,
                                                           Real E_kT,
                                                           Real E_kM,
                                                           Real E_kL,
                                                           Real& t_N,
                                                           Real& t_M,
                                                           Real& t_L,
                                                           Real& m_T,
                                                           Real& m_M,
                                                           Real& m_L) {
    t_N = E_N * e_N;
    t_M = E_T * e_M;
    t_L = E_T * e_L;
    m_T = E_kT * kappa_T;
    m_M = E_kM * kappa_M;
    m_L = E_kL * kappa_L;
}

}  // namespace tlfea
