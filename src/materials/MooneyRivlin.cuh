/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    MooneyRivlin.cuh
 * Brief:   Defines CUDA device routines for the Mooney-Rivlin hyperelastic
 *          material model (stress and tangent operator calculations).
 *==============================================================
 *==============================================================*/

#pragma once

#if defined(__CUDACC__)
    #include <cmath>

namespace feris {

__device__ __forceinline__ Real mr_det3x3(const Real A[3][3]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

__device__ __forceinline__ void mr_invT3x3(const Real A[3][3], Real detA, Real invT_out[3][3]) {
    const Real eps = 1e-12;
    Real safe_det = detA;
    if (fabs(safe_det) < eps) {
        safe_det = (safe_det >= 0.0) ? eps : -eps;
    }
    Real inv_det = 1.0 / safe_det;

    invT_out[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    invT_out[0][1] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
    invT_out[0][2] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;

    invT_out[1][0] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
    invT_out[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    invT_out[1][2] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;

    invT_out[2][0] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
    invT_out[2][1] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;
    invT_out[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;
}

__device__ __forceinline__ void mr_compute_P(const Real F[3][3], Real mu10, Real mu01, Real kappa, Real P_out[3][3]) {
    Real C[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                C[i][j] += F[k][i] * F[k][j];
            }
        }
    }

    Real I1 = C[0][0] + C[1][1] + C[2][2];

    Real C2[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                C2[i][j] += C[i][k] * C[k][j];
            }
        }
    }
    Real trC2 = C2[0][0] + C2[1][1] + C2[2][2];
    Real I2 = 0.5 * (I1 * I1 - trC2);

    Real J = mr_det3x3(F);

    Real FinvT[3][3];
    mr_invT3x3(F, J, FinvT);

    Real J13 = cbrt(J);
    Real Jm23 = 1.0 / (J13 * J13);
    Real Jm43 = Jm23 * Jm23;

    Real FC[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FC[i][j] += F[i][k] * C[k][j];
            }
        }
    }

    Real t1 = 2.0 * mu10 * Jm23;
    Real t2 = 2.0 * mu01 * Jm43;
    Real t3 = kappa * (J - 1.0) * J;

    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
            Real term1 = F[i][j] - (I1 / 3.0) * FinvT[i][j];
            Real term2 = I1 * F[i][j] - FC[i][j] - (2.0 * I2 / 3.0) * FinvT[i][j];
            Real term3 = FinvT[i][j];
            P_out[i][j] = t1 * term1 + t2 * term2 + t3 * term3;
        }
    }
}

__device__ __forceinline__ void mr_compute_tangent_tensor(const Real F[3][3],
                                                          Real mu10,
                                                          Real mu01,
                                                          Real kappa,
                                                          Real A[3][3][3][3]) {
    Real C[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                C[i][j] += F[k][i] * F[k][j];
            }
        }
    }

    Real I1 = C[0][0] + C[1][1] + C[2][2];

    Real C2[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                C2[i][j] += C[i][k] * C[k][j];
            }
        }
    }
    Real trC2 = C2[0][0] + C2[1][1] + C2[2][2];
    Real I2 = 0.5 * (I1 * I1 - trC2);

    Real J = mr_det3x3(F);

    Real FinvT[3][3];
    mr_invT3x3(F, J, FinvT);

    Real J13 = cbrt(J);
    Real Jm23 = 1.0 / (J13 * J13);
    Real Jm43 = Jm23 * Jm23;

    Real FC[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FC[i][j] += F[i][k] * C[k][j];
            }
        }
    }

    Real FFT[3][3] = {{0.0}};
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
                FFT[i][j] += F[i][k] * F[j][k];
            }
        }
    }

    Real t1 = 2.0 * mu10 * Jm23;
    Real t2 = 2.0 * mu01 * Jm43;
    Real t3 = kappa * (J - 1.0) * J;

    Real term1[3][3];
    Real term2[3][3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
            term1[i][j] = F[i][j] - (I1 / 3.0) * FinvT[i][j];
            term2[i][j] = I1 * F[i][j] - FC[i][j] - (2.0 * I2 / 3.0) * FinvT[i][j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 3; i++) {
    #pragma unroll
        for (int j = 0; j < 3; j++) {
    #pragma unroll
            for (int k = 0; k < 3; k++) {
    #pragma unroll
                for (int l = 0; l < 3; l++) {
                    Real delta_ik = (i == k) ? 1.0 : 0.0;
                    Real delta_jl = (j == l) ? 1.0 : 0.0;

                    Real dFinvT = -FinvT[i][l] * FinvT[k][j];

                    Real dt1 = (-2.0 / 3.0) * t1 * FinvT[k][l];
                    Real dt2 = (-4.0 / 3.0) * t2 * FinvT[k][l];
                    Real dt3 = (kappa * (2.0 * J - 1.0) * J) * FinvT[k][l];

                    Real dT1 = delta_ik * delta_jl - (2.0 / 3.0) * F[k][l] * FinvT[i][j] +
                               (I1 / 3.0) * FinvT[i][l] * FinvT[k][j];

                    Real dT2 = 2.0 * F[k][l] * F[i][j] + I1 * delta_ik * delta_jl -
                               (delta_ik * C[l][j] + F[i][l] * F[k][j] + delta_jl * FFT[i][k]) -
                               (4.0 / 3.0) * (I1 * F[k][l] - FC[k][l]) * FinvT[i][j] +
                               (2.0 * I2 / 3.0) * FinvT[i][l] * FinvT[k][j];

                    A[i][j][k][l] =
                        dt1 * term1[i][j] + t1 * dT1 + dt2 * term2[i][j] + t2 * dT2 + dt3 * FinvT[i][j] + t3 * dFinvT;
                }
            }
        }
    }
}

}  // namespace feris

#endif
