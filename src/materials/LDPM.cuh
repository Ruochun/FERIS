/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LDPM.cuh
 * Brief:   Full modern LDPM constitutive law for the 6-DOF
 *          LDPMTet4 element, after:
 *
 *            Cusatis, Pelessone & Mencarelli (2011),
 *            "Lattice Discrete Particle Model (LDPM) for failure
 *            behavior of concrete I: Theory",
 *            Cement & Concrete Composites 33(9):881-890.
 *
 *          Implements three coupled constitutive mechanisms:
 *
 *          1. Tensile fracturing with mode-mixity dependent
 *             boundary strength and exponential softening.
 *          2. Compressive yielding, hardening, pore collapse,
 *             and densification.
 *          3. Pressure-dependent frictional shear under
 *             compression.
 *
 *          Rotational moments remain linear-elastic (standard
 *          first-order LDPM formulation).
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include "../types.h"

namespace feris {

// ---------------------------------------------------------------------------
// LDPM material parameter struct (all passed by value to the device kernel).
// ---------------------------------------------------------------------------
struct LDPMParams {
    // Elastic
    Real E_N;   // Normal modulus [stress units, e.g. MPa]
    Real E_T;   // Shear modulus = alpha * E_N
    Real E_kT;  // Twist rotational modulus
    Real E_kM;  // Bending modulus (m-direction)
    Real E_kL;  // Bending modulus (l-direction)

    // Tensile fracturing
    Real sigma_t;  // Mesoscale tensile strength
    Real l_t;      // Tensile characteristic length [mm]
    Real r_st;     // Shear-to-tensile strength ratio (sigma_s / sigma_t)
    Real n_t;      // Softening exponent (mode-mixity shape parameter)

    // Compression / compaction
    Real sigma_c0;   // Initial compressive yield stress (positive value)
    Real Hc0_ratio;  // H_c0 / E_N ratio (initial hardening slope)
    Real kappa_c0;   // Strain ratio at rehardening transition
    Real Ed_ratio;   // E_d / E_N densification stiffness ratio

    // Friction
    Real mu_0;      // Low-pressure friction coefficient
    Real mu_inf;    // High-pressure (asymptotic) friction coefficient
    Real sigma_N0;  // Transition normal stress for friction law
};

// ---------------------------------------------------------------------------
// LDPM per-edge state variables (read/written each step).
// ---------------------------------------------------------------------------
struct LDPMState {
    Real kappa;     // Max tensile effective strain (history, grows only)
    Real omega;     // Tensile damage variable [0, 1]
    Real e_N_comp;  // Max compressive normal strain (negative, history)
};

// ---------------------------------------------------------------------------
// ldpm_tet4_constitutive_update
//
// Full modern LDPM constitutive update for a single facet strut.
// Implements tensile fracturing with mode-mixity, compressive boundary,
// and pressure-dependent friction.
//
// Inputs (strains):
//   e_N, e_M, e_L           — translational engineering strains
//   kappa_T, kappa_M, kappa_L — rotational strains (1/length)
//
// Material parameters:
//   params                  — full LDPM parameter set
//
// State (in/out):
//   state                   — per-edge history variables (updated in place)
//
// Outputs (tractions and moments):
//   t_N, t_M, t_L          — facet tractions
//   m_T, m_M, m_L          — facet moments per unit area
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldpm_tet4_constitutive_update(Real e_N,
                                                              Real e_M,
                                                              Real e_L,
                                                              Real kappa_T,
                                                              Real kappa_M,
                                                              Real kappa_L,
                                                              const LDPMParams& params,
                                                              LDPMState& state,
                                                              Real& t_N,
                                                              Real& t_M,
                                                              Real& t_L,
                                                              Real& m_T,
                                                              Real& m_M,
                                                              Real& m_L) {
    const Real E_N = params.E_N;
    const Real E_T = params.E_T;

    // ── Total shear strain magnitude ─────────────────────────────────────────
    const Real e_T = sqrt(e_M * e_M + e_L * e_L);

    if (e_N > Real(0)) {
        // ═══════════════════════════════════════════════════════════════════════
        // TENSILE / OPENING REGIME — Mode-mixity dependent damage boundary
        // ═══════════════════════════════════════════════════════════════════════

        // Shear-to-normal modulus ratio for effective strain
        const Real alpha = (E_N > Real(0)) ? E_T / E_N : Real(0);

        // Effective strain: e* = sqrt( e_N^2 + alpha * e_T^2 )
        const Real e_eff = sqrt(e_N * e_N + alpha * e_T * e_T);

        // Mode-mixity angle: tan(omega_angle) = sqrt(alpha) * e_T / e_N
        // This determines the mixed-mode boundary strength.
        const Real tan_omega = (e_N > Real(1e-30)) ? sqrt(alpha) * e_T / e_N : Real(0);

        // Mode-mixity dependent boundary strength:
        //   sigma_bt(omega) = sigma_t / sqrt( 1 + (tan_omega / r_st)^2 )
        // Using the canonical form with n_t exponent:
        //   sigma_bt = sigma_t * [ 1 - (tan_omega / r_st)^n_t ]^(1/n_t)
        //   but clamped to a minimum to avoid zero/negative strength
        const Real r_st = params.r_st;
        const Real n_t = params.n_t;
        Real sigma_bt = params.sigma_t;

        if (r_st > Real(0) && tan_omega > Real(0)) {
            const Real ratio = tan_omega / r_st;
            // Use the smooth power-law interpolation
            // For the common case n_t=2: sigma_bt = sigma_t / sqrt(1 + ratio^2)
            // General: sigma_bt = sigma_t * (1 + ratio^n_t)^(-1/n_t)
            // This formulation always gives positive values.
            const Real ratio_pow = pow(ratio, n_t);
            sigma_bt = params.sigma_t * pow(Real(1) + ratio_pow, Real(-1) / n_t);
        }

        // Threshold strain at peak for this mode mixity
        const Real e_t0 = (E_N > Real(0)) ? sigma_bt / E_N : Real(0);

        // Softening modulus from characteristic length:
        //   H_t = 2 * E_N / (sigma_bt * l_t)
        // This ensures correct fracture energy dissipation.
        const Real H_t =
            (sigma_bt > Real(0) && params.l_t > Real(0)) ? Real(2) * E_N / (sigma_bt * params.l_t) : Real(0);

        // Update history variable
        state.kappa = (e_eff > state.kappa) ? e_eff : state.kappa;

        // Compute damage variable with exponential softening
        Real omega = Real(0);
        if (state.kappa > e_t0 && e_t0 > Real(0)) {
            omega = Real(1) - (e_t0 / state.kappa) * exp(-H_t * (state.kappa - e_t0));
            omega = (omega < Real(0)) ? Real(0) : omega;
            omega = (omega > Real(1)) ? Real(1) : omega;
        }
        state.omega = omega;

        // Tractions in tension: damage reduces all components
        t_N = E_N * e_N * (Real(1) - omega);
        t_M = E_T * e_M * (Real(1) - omega);
        t_L = E_T * e_L * (Real(1) - omega);

    } else {
        // ═══════════════════════════════════════════════════════════════════════
        // COMPRESSIVE / CLOSURE REGIME — Yielding, hardening, and friction
        // ═══════════════════════════════════════════════════════════════════════

        // --- Compressive boundary ---
        // Track max compressive strain (most negative)
        if (e_N < state.e_N_comp)
            state.e_N_comp = e_N;

        // Compressive boundary strength (positive value for comparison):
        //   sigma_bc = sigma_c0 + H_c0 * |e_N_comp_history|
        // With densification at large strains:
        //   sigma_bc = sigma_c0 * [1 + (Hc0/E_N)*E_N*|e_N|/sigma_c0]
        //            = sigma_c0 + H_c0 * |e_N_comp|
        // Then after kappa_c0 * sigma_c0/E_N, densification kicks in.
        const Real abs_e_N = -e_N;  // positive value of compressive strain
        const Real sigma_c0 = params.sigma_c0;
        const Real H_c0 = params.Hc0_ratio * E_N;

        // Elastic limit strain for compression
        const Real e_c0 = (E_N > Real(0)) ? sigma_c0 / E_N : Real(0);

        if (abs_e_N <= e_c0 || sigma_c0 <= Real(0)) {
            // Below compressive yield — purely elastic
            t_N = E_N * e_N;
        } else {
            // Beyond compressive yield — hardening
            // Compressive boundary: piecewise linear hardening then densification
            const Real e_plastic = abs_e_N - e_c0;
            const Real kappa_c0_strain = params.kappa_c0 * e_c0;

            Real sigma_bc;
            if (e_plastic <= kappa_c0_strain) {
                // Linear hardening phase
                sigma_bc = sigma_c0 + H_c0 * e_plastic;
            } else {
                // Densification phase (exponential stiffening)
                const Real sigma_at_transition = sigma_c0 + H_c0 * kappa_c0_strain;
                const Real E_d = params.Ed_ratio * E_N;
                sigma_bc = sigma_at_transition + E_d * (e_plastic - kappa_c0_strain);
            }

            // Normal traction (compressive, negative)
            t_N = -sigma_bc;
        }

        // --- Pressure-dependent friction for shear under compression ---
        // Friction coefficient: mu(sigma_N) = mu_0 + (mu_inf - mu_0) * (1 - exp(-|sigma_N|/sigma_N0))
        const Real abs_sigma_N = -t_N;  // positive magnitude of compressive traction

        Real mu_f = params.mu_0;
        if (params.sigma_N0 > Real(0)) {
            mu_f = params.mu_0 + (params.mu_inf - params.mu_0) * (Real(1) - exp(-abs_sigma_N / params.sigma_N0));
        }

        // Shear strength limit: tau_max = mu_f * |sigma_N|
        const Real tau_max = mu_f * abs_sigma_N;

        // Elastic trial shear tractions
        const Real t_M_trial = E_T * e_M;
        const Real t_L_trial = E_T * e_L;
        const Real tau_trial = sqrt(t_M_trial * t_M_trial + t_L_trial * t_L_trial);

        if (tau_trial <= tau_max || tau_max <= Real(0) || tau_trial <= Real(1e-30)) {
            // Below friction limit — elastic shear
            t_M = t_M_trial;
            t_L = t_L_trial;
        } else {
            // Frictional return mapping — scale shear to friction limit
            const Real scale = tau_max / tau_trial;
            t_M = t_M_trial * scale;
            t_L = t_L_trial * scale;
        }

        // Tensile damage does not affect shear in compression regime
        // (friction governs instead)
    }

    // ── Rotational moments (linear-elastic, no damage coupling) ──────────────
    m_T = params.E_kT * kappa_T;
    m_M = params.E_kM * kappa_M;
    m_L = params.E_kL * kappa_L;
}

// ---------------------------------------------------------------------------
// Legacy wrapper: ldpm_tet4_cusatis_traction
//
// Maintains backward compatibility with existing code that uses the old
// simplified interface (elastic + tensile-only damage).  Internally delegates
// to the full constitutive update with default compression/friction params
// that reproduce the old elastic compression + tensile damage behavior.
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
    // Build params struct with legacy behavior:
    // - Tensile: use H_t directly (derive l_t from H_t and sigma_t)
    //   H_t = 2*E_N / (sigma_t * l_t)  =>  l_t = 2*E_N / (sigma_t * H_t)
    // - Compression: sigma_c0 = 0 means no yield (stays elastic)
    // - Friction: mu_0 = 0, no friction limit
    LDPMParams params;
    params.E_N = E_N;
    params.E_T = E_T;
    params.E_kT = E_kT;
    params.E_kM = E_kM;
    params.E_kL = E_kL;
    params.sigma_t = sigma_t;
    // Convert legacy H_t to l_t:  l_t = 2*E_N / (sigma_t * H_t)
    params.l_t = (sigma_t > Real(0) && H_t > Real(0)) ? Real(2) * E_N / (sigma_t * H_t) : Real(1e10);
    params.r_st = Real(1e10);  // Very large: no mode-mixity reduction (legacy behavior)
    params.n_t = Real(2);
    params.sigma_c0 = Real(0);  // No compressive yield (elastic compression)
    params.Hc0_ratio = Real(0);
    params.kappa_c0 = Real(1);
    params.Ed_ratio = Real(0);
    params.mu_0 = Real(0);  // No friction limit
    params.mu_inf = Real(0);
    params.sigma_N0 = Real(1);

    LDPMState state;
    state.kappa = kappa_in;
    state.omega = Real(0);
    state.e_N_comp = Real(0);

    ldpm_tet4_constitutive_update(e_N, e_M, e_L, kappa_T, kappa_M, kappa_L, params, state, t_N, t_M, t_L, m_T, m_M,
                                  m_L);

    kappa_out = state.kappa;
    omega_out = state.omega;
}

}  // namespace feris
