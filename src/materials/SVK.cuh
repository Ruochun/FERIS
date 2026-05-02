/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    SVK.cuh
 * Brief:   Defines CUDA device routines for the St. Venant–Kirchhoff (SVK)
 *          hyperelastic material model (stress and tangent block calculations).
 *==============================================================
 *==============================================================*/

#pragma once
#if defined(__CUDACC__)

namespace feris {

__device__ __forceinline__ void svk_compute_P_from_trFtF_and_FFtF(const Real F[3][3],
                                                                  Real trFtF,
                                                                  const Real FFtF[3][3],
                                                                  Real lambda,
                                                                  Real mu,
                                                                  Real P_out[3][3]) {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
            P_out[i][j] = 0.0;
        }
    }

    Real lambda_factor = lambda * (0.5 * trFtF - 1.5);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
            P_out[i][j] = lambda_factor * F[i][j] + mu * (FFtF[i][j] - F[i][j]);
        }
    }
}

__device__ __forceinline__ void svk_compute_tangent_block(const Real Fh_i[3],
                                                          const Real Fh_j[3],
                                                          Real hij,
                                                          Real trE,
                                                          Real Fhj_dot_Fhi,
                                                          const Real FFT[3][3],
                                                          Real lambda,
                                                          Real mu,
                                                          Real dV,
                                                          Real Kblock[3][3]) {
    #pragma unroll
    for (int d = 0; d < 3; d++) {
    #pragma unroll
        for (int e = 0; e < 3; e++) {
            Real delta = (d == e) ? 1.0 : 0.0;

            Real A_de = lambda * Fh_i[d] * Fh_j[e];
            Real B_de = lambda * trE * hij * delta;
            Real C1_de = mu * Fhj_dot_Fhi * delta;
            Real D_de = mu * Fh_j[d] * Fh_i[e];
            Real Etrm_de = mu * hij * FFT[d][e];
            Real Ftrm_de = -mu * hij * delta;

            Kblock[d][e] = (A_de + B_de + C1_de + D_de + Etrm_de + Ftrm_de) * dV;
        }
    }
}

__device__ __forceinline__ void svk_compute_P(const Real F[3][3], Real lambda, Real mu, Real P_out[3][3]) {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
            P_out[i][j] = 0.0;
        }
    }

    Real FtF[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FtF[i][j] += F[k][i] * F[k][j];
            }
        }
    }

    Real trFtF = FtF[0][0] + FtF[1][1] + FtF[2][2];

    Real FFt[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FFt[i][j] += F[i][k] * F[j][k];
            }
        }
    }

    Real FFtF[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FFtF[i][j] += FFt[i][k] * F[k][j];
            }
        }
    }

    svk_compute_P_from_trFtF_and_FFtF(F, trFtF, FFtF, lambda, mu, P_out);
}

}  // namespace feris

#endif
