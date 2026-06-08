/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LDPM.cuh
 * Brief:   Full LDPM constitutive law for the 6-DOF LDPMTet4
 *          element, implementing the complete facet-level model:
 *
 *          1. Tensile fracture with mode-mixity dependence
 *          2. Compressive yielding, hardening, pore collapse
 *          3. Pressure-dependent frictional shear
 *
 *          After: Cusatis, Pelessone & Mencarelli (2011),
 *            "Lattice Discrete Particle Model (LDPM) for failure
 *            behavior of concrete I: Theory",
 *            Cement & Concrete Composites 33(9):881-890.
 *
 *          And the CPU reference implementation in:
 *            chrono-mechanics (base_dev branch) / chrono_ldpm /
 *            ChMaterialVECT.cpp
 *
 * ── Per-edge state vector layout (16 components) ──────────────
 *
 *   statev[0]  = eps_N  (normal strain, accumulated)
 *   statev[1]  = eps_M  (shear M strain, accumulated)
 *   statev[2]  = eps_L  (shear L strain, accumulated)
 *   statev[3]  = sigma_N (normal stress; previous step on entry, current step on return)
 *   statev[4]  = sigma_M (shear M stress; previous step on entry, current step on return)
 *   statev[5]  = sigma_L (shear L stress; previous step on entry, current step on return)
 *   statev[6]  = max_eps_N (max normal strain history)
 *   statev[7]  = max_eps_T (max shear magnitude history)
 *   statev[8]  = effective strain (previous step on entry, current step on return)
 *   statev[9]  = effective stress (previous step on entry, current step on return)
 *   statev[10] = internal work
 *   statev[11] = crack opening
 *   statev[12] = eps_N net (reserved for a future eigenstrain path; currently not updated)
 *   statev[13] = eps_M net (reserved for a future eigenstrain path; currently not updated)
 *   statev[14] = eps_L net (reserved for a future eigenstrain path; currently not updated)
 *   statev[15] = dissipated energy
 *
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>

#include "../types.h"

namespace feris {

// Number of state variables per edge for the full LDPM model.
static constexpr int LDPM_N_STATEV = 16;

// ---------------------------------------------------------------------------
// LDPMParams — full LDPM material parameter set stored on device.
// ---------------------------------------------------------------------------
struct LDPMParams {
    Real E0;            // Normal modulus
    Real alpha;         // Shear-to-normal ratio (E_T = alpha * E0)
    Real sigma_t;       // Tensile strength
    Real sigma_s;       // Shear strength (sigma_s = r_st * sigma_t)
    Real n_t;           // Softening exponent (mode-mixity shape)
    Real l_t;           // Tensile characteristic length [mm]
    Real r_s;           // Shear softening modulus ratio
    Real k_t;           // Tensile unloading parameter
    Real E_d;           // Densified normal modulus (or Ed/E0 ratio * E0)
    Real sigma_c0;      // Compressive yield strength
    Real H_c0;          // Initial hardening modulus
    Real H_c1;          // Final hardening modulus
    Real kc0;           // Transitional strain ratio
    Real kc1;           // Deviatoric strain threshold ratio
    Real kc2;           // Deviatoric damage parameter
    Real kc3;           // Volumetric strain parameter
    Real beta;          // Volumetric-deviatoric coupling
    Real mu_0;          // Initial friction coefficient
    Real mu_inf;        // Asymptotic friction coefficient
    Real sigma_N0;      // Transitional stress for friction law
    Real E_kT;          // Retained for API compatibility; unused by Chrono LDPM path
    Real E_kM;          // Retained for API compatibility; unused by Chrono LDPM path
    Real E_kL;          // Retained for API compatibility; unused by Chrono LDPM path
    Real rho;           // Density
    bool elastic_flag;  // If true, skip all inelastic updates
};

// ---------------------------------------------------------------------------
// ldpm_fracture_boundary
//
// Tensile-shear mode-mixity softening boundary.
// Returns the effective stress sigma_fr on the fracture surface.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real ldpm_fracture_boundary(Real eps_N,
                                                       Real eps_M,
                                                       Real eps_L,
                                                       Real l0,
                                                       Real max_eps_N,
                                                       Real max_eps_T,
                                                       Real prev_eff_strain,
                                                       Real prev_eff_stress,
                                                       const LDPMParams& p) {
    const Real E0 = p.E0;
    const Real alpha = p.alpha;
    const Real sigma_t = p.sigma_t;
    const Real sigma_s = p.sigma_s;
    const Real n_t = p.n_t;
    const Real l_t = p.l_t;
    const Real k_t = p.k_t;
    const Real r_s = p.r_s;

    const Real eps_T = sqrt(eps_M * eps_M + eps_L * eps_L);
    const Real eps_Q = sqrt(eps_N * eps_N + alpha * (eps_M * eps_M + eps_L * eps_L));

    // Mode-mixity angle omega
    Real omega_angle;
    Real sin_w, cos_w, cos_w2;

    if (eps_T < Real(1e-16)) {
        omega_angle = Real(3.14159265358979323846) * Real(0.5);
        sin_w = Real(1);
        cos_w = Real(0);
        cos_w2 = Real(0);
    } else {
        omega_angle = atan(eps_N / (sqrt(alpha) * eps_T));
        sin_w = sin(omega_angle);
        cos_w = cos(omega_angle);
        cos_w2 = cos_w * cos_w;
    }

    // Effective strength sigma0 depending on mode mixity
    Real sigma0;
    const Real r_st = (sigma_t > Real(0)) ? sigma_s / sigma_t : Real(1);

    if (cos_w2 < Real(1e-16)) {
        sigma0 = sigma_t;
    } else {
        sigma0 = sigma_t * (-sin_w + sqrt(sin_w * sin_w + Real(4) * alpha * cos_w2 / (r_st * r_st))) /
                 (Real(2) * alpha * cos_w2 / (r_st * r_st));
    }

    // Threshold strain and softening modulus
    const Real eps0 = sigma0 / E0;
    const Real Ht = Real(2) * E0 / (l_t / l0 - Real(1));
    const Real Hs = r_s * E0;
    const Real pi_val = Real(3.14159265358979323846);
    const Real H0 = Hs / alpha + (Ht - Hs / alpha) * pow(Real(2) * omega_angle / pi_val, n_t);

    // Maximum effective strain history
    const Real eps_max = sqrt(max_eps_N * max_eps_N + alpha * max_eps_T * max_eps_T);

    // Stress boundary with exponential softening
    const Real decay_arg = H0 * std::max(eps_max - eps0, Real(0)) / sigma0;
    const Real sigma_bt = sigma0 * exp(-decay_arg);

    // Unloading check
    const Real eps_tr = k_t * (eps_max - sigma_bt / E0);
    Real sigma_bt_eff = sigma_bt;
    if (eps_tr > eps_Q) {
        sigma_bt_eff = Real(0);
    }

    // Elastic predictor
    const Real strs_ela = E0 * (eps_Q - prev_eff_strain) + prev_eff_stress;
    Real sigma_fr = std::min(std::max(strs_ela, Real(0)), sigma_bt_eff);

    if (p.elastic_flag) {
        sigma_fr = strs_ela;
    }
    return sigma_fr;
}

// ---------------------------------------------------------------------------
// ldpm_compress_boundary
//
// Compressive yielding / hardening / pore-collapse boundary.
// Returns the clamped normal stress sigma_com (negative in compression).
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real
ldpm_compress_boundary(Real eps_N, Real eps_V, Real prev_sigma_N, Real prev_eps_N, const LDPMParams& p) {
    const Real E0 = p.E0;
    const Real E_d = p.E_d;
    const Real sigma_c0 = p.sigma_c0;
    const Real beta_param = p.beta;
    const Real H_c0 = p.H_c0;
    const Real H_c1 = p.H_c1;
    const Real kc0 = p.kc0;
    const Real kc1 = p.kc1;
    const Real kc2 = p.kc2;
    const Real kc3 = p.kc3;

    const Real eps_D = eps_N - eps_V;
    const Real eps_DV = eps_V + beta_param * eps_D;
    const Real eps_c0 = sigma_c0 / E0;
    const Real eps_c1 = eps_c0 * kc0;
    const Real eps_v0 = eps_c0 * kc3;

    // Deviatoric-volumetric ratio
    Real r_DV;
    if (eps_V <= Real(0)) {
        r_DV = -abs(eps_D) / (eps_V - eps_v0);
    } else {
        r_DV = abs(eps_D) / eps_v0;
    }

    // Confinement-sensitive hardening modulus
    const Real Hc = (H_c0 - H_c1) / (Real(1) + kc2 * std::max(r_DV - kc1, Real(0))) + H_c1;
    const Real sigma_c1 = sigma_c0 + (eps_c1 - eps_c0) * Hc;

    // Compressive boundary stress
    Real sigma_bc;
    if (-eps_DV <= Real(0)) {
        sigma_bc = sigma_c0;
    } else if (-eps_DV > Real(0) && -eps_DV <= eps_c1) {
        sigma_bc = sigma_c0 + std::max(-eps_DV - eps_c0, Real(0)) * Hc;
    } else {
        sigma_bc = sigma_c1 * exp((-eps_DV - eps_c1) * Hc / sigma_c1);
    }

    // Densification modulus switching
    Real ENc;
    if (-prev_sigma_N < sigma_c0) {
        ENc = E0;
    } else {
        ENc = E_d;
    }

    // Elastic predictor clamped to boundary
    const Real sigma_ela = prev_sigma_N + ENc * (eps_N - prev_eps_N);
    Real sigma_com = std::min(std::max(sigma_ela, -sigma_bc), Real(0));

    if (p.elastic_flag) {
        sigma_com = sigma_ela;
    }
    return sigma_com;
}

// ---------------------------------------------------------------------------
// ldpm_shear_boundary
//
// Pressure-dependent frictional shear limit.
// Returns (sigma_M, sigma_L) clamped to the shear boundary.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldpm_shear_boundary(Real eps_M,
                                                    Real eps_L,
                                                    Real sigma_N_current,
                                                    Real prev_sigma_M,
                                                    Real prev_sigma_L,
                                                    Real prev_eps_M,
                                                    Real prev_eps_L,
                                                    Real& sigma_M_out,
                                                    Real& sigma_L_out,
                                                    const LDPMParams& p) {
    const Real E0 = p.E0;
    const Real alpha = p.alpha;
    const Real mu0 = p.mu_0;
    const Real mu_inf = p.mu_inf;
    const Real sigma_s = p.sigma_s;
    const Real sigma_N0 = p.sigma_N0;

    // Shear boundary stress (pressure-dependent friction)
    const Real sigma_bs = sigma_s + (mu0 - mu_inf) * sigma_N0 - mu_inf * sigma_N_current -
                          (mu0 - mu_inf) * sigma_N0 * exp(sigma_N_current / sigma_N0);

    const Real ET = alpha * E0;
    const Real sigma_M_ela = prev_sigma_M + (eps_M - prev_eps_M) * ET;
    const Real sigma_L_ela = prev_sigma_L + (eps_L - prev_eps_L) * ET;
    const Real sigma_T_ela = sqrt(sigma_M_ela * sigma_M_ela + sigma_L_ela * sigma_L_ela);

    Real sigma_T = std::min(std::max(sigma_T_ela, Real(0)), sigma_bs);

    if (p.elastic_flag) {
        sigma_T = sigma_T_ela;
    }

    if (sigma_T_ela > Real(1e-30)) {
        sigma_M_out = sigma_T * sigma_M_ela / sigma_T_ela;
        sigma_L_out = sigma_T * sigma_L_ela / sigma_T_ela;
    } else {
        sigma_M_out = Real(0);
        sigma_L_out = Real(0);
    }
}

// ---------------------------------------------------------------------------
// ldpm_tet4_full_constitutive
//
// Full LDPM constitutive update for a single facet strut.
// Implements tensile fracture, compressive boundary, and shear friction.
//
// Inputs:
//   d_eps_N, d_eps_M, d_eps_L — strain INCREMENTS this step
//   kappa_T, kappa_M, kappa_L — rotational strain placeholders
//   eps_V                     — volumetric strain (mean of adjacent element)
//   l0                        — reference edge length
//   params                    — full LDPM material parameters
//   statev[LDPM_N_STATEV]     — in/out per-edge state vector
//
// Outputs:
//   t_N, t_M, t_L — facet tractions
//   m_T, m_M, m_L — facet couple moments per unit area; zero for Chrono LDPM
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldpm_tet4_full_constitutive(Real d_eps_N,
                                                            Real d_eps_M,
                                                            Real d_eps_L,
                                                            Real /*kappa_T*/,
                                                            Real /*kappa_M*/,
                                                            Real /*kappa_L*/,
                                                            Real eps_V,
                                                            Real l0,
                                                            const LDPMParams& params,
                                                            Real* statev,
                                                            Real& t_N,
                                                            Real& t_M,
                                                            Real& t_L,
                                                            Real& m_T,
                                                            Real& m_M,
                                                            Real& m_L) {
    const Real E0 = params.E0;
    const Real alpha = params.alpha;

    // The state vector follows the external LDPM reference layout. Slots 0..5
    // are previous-step values on entry and are overwritten with current-step
    // values before return.
    // Accumulate total strains
    const Real eps_N = statev[0] + d_eps_N;
    const Real eps_M = statev[1] + d_eps_M;
    const Real eps_L = statev[2] + d_eps_L;

    const Real eps_Q = sqrt(eps_N * eps_N + alpha * (eps_M * eps_M + eps_L * eps_L));
    const Real eps_T = sqrt(eps_M * eps_M + eps_L * eps_L);

    // Update history maxima
    if (eps_N > statev[6]) {
        statev[6] = eps_N;
    }
    if (eps_T > statev[7]) {
        statev[7] = eps_T;
    }

    Real sigma_N, sigma_M, sigma_L;

    if (eps_Q > Real(1e-30)) {
        if (eps_N > Real(1e-15)) {
            // ── Tensile / fracture regime ────────────────────────────────────
            const Real strs_Q =
                ldpm_fracture_boundary(eps_N, eps_M, eps_L, l0, statev[6], statev[7], statev[8], statev[9], params);
            sigma_N = strs_Q * eps_N / eps_Q;
            sigma_M = alpha * strs_Q * eps_M / eps_Q;
            sigma_L = alpha * strs_Q * eps_L / eps_Q;
        } else {
            // ── Compressive regime ───────────────────────────────────────────
            sigma_N = ldpm_compress_boundary(eps_N, eps_V, statev[3], statev[0], params);

            // ── Shear under compression (frictional) ─────────────────────────
            ldpm_shear_boundary(eps_M, eps_L, sigma_N, statev[4], statev[5], statev[1], statev[2], sigma_M, sigma_L,
                                params);
        }

        // ── Energy bookkeeping ───────────────────────────────────────────────
        const Real d_eps_vec[3] = {d_eps_N, d_eps_M, d_eps_L};
        const Real avg_sigma[3] = {(sigma_N + statev[3]) * Real(0.5), (sigma_M + statev[4]) * Real(0.5),
                                   (sigma_L + statev[5]) * Real(0.5)};
        const Real Wint =
            l0 * (avg_sigma[0] * d_eps_vec[0] + avg_sigma[1] * d_eps_vec[1] + avg_sigma[2] * d_eps_vec[2]);

        // Inelastic strain increment for dissipated energy
        const Real d_stress_in_N = (sigma_N - statev[3]) / E0;
        const Real d_stress_in_M = (sigma_M - statev[4]) / (E0 * alpha);
        const Real d_stress_in_L = (sigma_L - statev[5]) / (E0 * alpha);
        const Real d_eps_in_N = d_eps_N - d_stress_in_N;
        const Real d_eps_in_M = d_eps_M - d_stress_in_M;
        const Real d_eps_in_L = d_eps_L - d_stress_in_L;
        const Real DE = l0 * (avg_sigma[0] * d_eps_in_N + avg_sigma[1] * d_eps_in_M + avg_sigma[2] * d_eps_in_L);

        // Crack opening
        const Real w_N = l0 * (eps_N - sigma_N / E0);
        const Real w_M = l0 * (eps_M - sigma_M / (E0 * alpha));
        const Real w_L = l0 * (eps_L - sigma_L / (E0 * alpha));
        const Real w = sqrt(w_N * w_N + w_M * w_M + w_L * w_L);

        // ── Update state vector ──────────────────────────────────────────────
        statev[0] = eps_N;
        statev[1] = eps_M;
        statev[2] = eps_L;
        statev[3] = sigma_N;
        statev[4] = sigma_M;
        statev[5] = sigma_L;
        statev[8] = sqrt(eps_N * eps_N + alpha * (eps_M * eps_M + eps_L * eps_L));
        statev[9] = sqrt(sigma_N * sigma_N + (sigma_M * sigma_M + sigma_L * sigma_L) / alpha);
        statev[10] += Wint;
        statev[11] = w;
        statev[15] += DE;
    } else {
        // Zero strain — keep existing state
        sigma_N = Real(0);
        sigma_M = Real(0);
        sigma_L = Real(0);
        statev[0] = eps_N;
        statev[1] = eps_M;
        statev[2] = eps_L;
    }

    // ── Output tractions ─────────────────────────────────────────────────────
    t_N = sigma_N;
    t_M = sigma_M;
    t_L = sigma_L;

    // Chrono's LDPM element has no separate rotational couple-stress law.
    // Nodal rotational residuals come from facet tractions acting through
    // reference facet-center lever arms in the element assembly.
    m_T = Real(0);
    m_M = Real(0);
    m_L = Real(0);
}

}  // namespace feris
