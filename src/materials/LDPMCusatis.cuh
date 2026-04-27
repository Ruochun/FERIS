/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LDPMCusatis.cuh
 * Brief:   Cusatis LDPM tensile-shear damage constitutive law
 *          for the 6-DOF LDPMTet4 element.
 *
 *          Implements the mesoscale tensile-shear coupled damage
 *          model for concrete-like materials, after:
 *
 *            Cusatis, Pelessone & Mencarelli (2011),
 *            "Lattice Discrete Particle Model (LDPM) for failure
 *            behavior of concrete I: Theory",
 *            Cement & Concrete Composites 33(9):881-890.
 *
 * ── Damage driving strain ─────────────────────────────────────
 *
 *   Characteristic (effective) strain for tensile-shear coupling:
 *
 *     e_D = sqrt( max(e_N, 0)²  +  alpha * (e_M² + e_L²) )
 *
 *   where  alpha = E_T / E_N  (shear-to-normal modulus ratio).
 *
 * ── History variable ─────────────────────────────────────────
 *
 *   kappa is the maximum e_D ever reached by the facet.
 *   It is a persistent per-edge state variable that only grows:
 *
 *     kappa_new = max(kappa_old, e_D)
 *
 * ── Damage variable ──────────────────────────────────────────
 *
 *   Peak / threshold strain:
 *     e_t0 = sigma_t / E_N
 *
 *   Exponential tensile softening (omega ∈ [0, 1]):
 *     omega = 0                                  if kappa ≤ e_t0
 *     omega = 1 - (e_t0/kappa)
 *               * exp(-H_t*(kappa - e_t0))       otherwise
 *
 *   H_t is the softening modulus [dimensionless, i.e. per unit
 *   strain].  Its physical value ties to the mode-I fracture
 *   energy G_ft [N/mm] via:
 *
 *     H_t  ≈  l_ch * sigma_t / G_ft
 *
 *   where l_ch is the characteristic crack-bandwidth length
 *   (≈ edge length l0 of the facet strut).
 *
 *   Representative values for normal-strength concrete (mm/N/s):
 *     sigma_t = 4 N/mm²,  E_N = 60 000 N/mm²
 *     G_ft    = 0.05 N/mm  →  e_t0 ≈ 6.67e-5
 *     l_ch    = 10 mm      →  H_t  ≈ 800
 *
 * ── Traction law ─────────────────────────────────────────────
 *
 *   Normal traction:
 *     t_N = E_N * e_N                  (e_N ≤ 0 — elastic compression)
 *     t_N = E_N * e_N * (1 - omega)    (e_N > 0 — tension, degraded)
 *
 *   Shear tractions (damage always applied; residual stiffness
 *   persists after the facet has opened and re-closed):
 *     t_M = E_T * e_M * (1 - omega)
 *     t_L = E_T * e_L * (1 - omega)
 *
 *   Rotational moments (remain linear-elastic; the first-order
 *   Cusatis model does not couple flexural response to damage):
 *     m_T = E_kT * kappa_T
 *     m_M = E_kM * kappa_M
 *     m_L = E_kL * kappa_L
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include "../types.h"

namespace tlfea {

// ---------------------------------------------------------------------------
// ldpm_tet4_cusatis_traction
//
// Cusatis tensile-shear damage constitutive law for a single LDPM-TET4
// facet strut.  Called once per edge per time step from compute_p().
//
// Inputs (strains):
//   e_N, e_M, e_L           — translational engineering strains
//   kappa_T, kappa_M, kappa_L — rotational strains (1/length)
//
// Material parameters:
//   E_N, E_T               — normal and shear translational moduli
//   E_kT, E_kM, E_kL       — twist and bending moduli
//   sigma_t                — mesoscale tensile strength
//   H_t                    — softening modulus (per unit strain)
//
// State (in/out — written back to per-edge arrays by the caller):
//   kappa_in               — history variable (max e_D so far) on entry
//   kappa_out              — updated history variable on exit
//   omega_out              — damage variable on exit ∈ [0, 1]
//
// Outputs (tractions and moments):
//   t_N, t_M, t_L          — facet tractions  [same units as E_N * strain]
//   m_T, m_M, m_L          — facet moments per unit area
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldpm_tet4_cusatis_traction(Real e_N,
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
                                                           Real sigma_t,
                                                           Real H_t,
                                                           Real kappa_in,
                                                           Real& kappa_out,
                                                           Real& omega_out,
                                                           Real& t_N,
                                                           Real& t_M,
                                                           Real& t_L,
                                                           Real& m_T,
                                                           Real& m_M,
                                                           Real& m_L) {
    // ── Characteristic damage driving strain ─────────────────────────────────
    //   e_D = sqrt( <e_N>_+²  +  (E_T/E_N) * (e_M² + e_L²) )
    const Real e_N_pos = (e_N > Real(0)) ? e_N : Real(0);
    const Real alpha   = (E_N > Real(0)) ? E_T / E_N : Real(0);
    const Real e_D     = sqrt(e_N_pos * e_N_pos + alpha * (e_M * e_M + e_L * e_L));

    // ── History update ────────────────────────────────────────────────────────
    kappa_out = (e_D > kappa_in) ? e_D : kappa_in;

    // ── Damage variable ───────────────────────────────────────────────────────
    const Real e_t0 = (E_N > Real(0)) ? sigma_t / E_N : Real(0);

    Real omega = Real(0);
    if (kappa_out > e_t0 && e_t0 > Real(0)) {
        omega = Real(1) - (e_t0 / kappa_out) * exp(-H_t * (kappa_out - e_t0));
        // Clamp to [0, 1] for numerical robustness.
        omega = (omega < Real(0)) ? Real(0) : omega;
        omega = (omega > Real(1)) ? Real(1) : omega;
    }
    omega_out = omega;

    // ── Translational tractions ───────────────────────────────────────────────
    //   Normal: damage applies only in tension.
    t_N = (e_N > Real(0)) ? E_N * e_N * (Real(1) - omega) : E_N * e_N;

    //   Shear: always reduced by current damage (residual stiffness after crack
    //   closure is physically appropriate for rough-crack frictional sliding).
    t_M = E_T * e_M * (Real(1) - omega);
    t_L = E_T * e_L * (Real(1) - omega);

    // ── Rotational moments (linear-elastic, no damage coupling) ──────────────
    m_T = E_kT * kappa_T;
    m_M = E_kM * kappa_M;
    m_L = E_kL * kappa_L;
}

}  // namespace tlfea
