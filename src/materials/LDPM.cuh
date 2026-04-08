/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LDPM.cuh
 * Brief:   Linear-elastic facet constitutive law for the Lattice
 *          Discrete Particle Model (LDPM).  Each interaction facet
 *          has a normal modulus E_N and a shear modulus E_T.  The
 *          traction vector (t_N, t_M, t_L) is computed from the
 *          engineering strains (e_N, e_M, e_L) resolved along the
 *          local facet frame (n, m, l).
 *
 *          Reference: Cusatis, Pelessone & Mencarelli (2011),
 *          "Lattice Discrete Particle Model (LDPM) for failure
 *          behavior of concrete", Cement & Concrete Composites.
 *==============================================================
 *==============================================================*/

#pragma once

#include "../types.h"

namespace tlfea {

// Linear-elastic LDPM constitutive law.
//
// Computes facet tractions t_N, t_M, t_L from engineering strains
// e_N, e_M, e_L and facet moduli E_N (normal) and E_T (shear).
//
//   t_N = E_N * e_N   (normal traction)
//   t_M = E_T * e_M   (shear traction, direction m)
//   t_L = E_T * e_L   (shear traction, direction l)
__device__ __forceinline__ void ldpm_compute_traction(Real e_N,
                                                      Real e_M,
                                                      Real e_L,
                                                      Real E_N,
                                                      Real E_T,
                                                      Real& t_N,
                                                      Real& t_M,
                                                      Real& t_L) {
    t_N = E_N * e_N;
    t_M = E_T * e_M;
    t_L = E_T * e_L;
}

}  // namespace tlfea
