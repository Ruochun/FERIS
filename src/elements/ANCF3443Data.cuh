/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ANCF3443Data.cuh
 * Brief:   Declares the GPU_ANCF3443_Data structure and associated host/GPU
 *          interfaces for ANCF 3443 shell elements. Stores connectivity,
 *          quadrature meta-data, CSR mass matrices, and element-level force
 *          and constraint data used by the solvers.
 *==============================================================
 *==============================================================*/

#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <MoPhiEssentials.h>
#include "../types.h"

#include "../utils/cpu_utils.h"
#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "../materials/MaterialModel.cuh"
#include "ElementBase.h"

namespace feris {

//
// define a SAP data strucutre
struct GPU_ANCF3443_Data : public ElementBase {
    // Constraint modes:
    // - kConstraintFixedCoefficients: legacy fixed coefficient constraints via
    //   d_fixed_nodes[] and x12_jac/y12_jac/z12_jac targets.
    // - kConstraintLinearCSR: general linear constraints with explicit J in CSR
    //   and per-row rhs targets.
    static constexpr int kConstraintNone = 0;
    static constexpr int kConstraintFixedCoefficients = 1;
    static constexpr int kConstraintLinearCSR = 2;

#if defined(__CUDACC__)

    // Const get functions
    __device__ const Map<MatrixXR> B_inv(int elem_idx) const {
        const int row_size = Quadrature::N_SHAPE_3443;
        const int col_size = Quadrature::N_SHAPE_3443;
        return Map<MatrixXR>(d_B_inv + elem_idx * row_size * col_size, row_size, col_size);
    }

    __device__ Map<MatrixXR> grad_N_ref(int elem_idx, int qp_idx) const {
        const int row_size = Quadrature::N_SHAPE_3443;
        const int col_size = 3;
        Real* qp_data = d_grad_N_ref + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * row_size * col_size;
        return Map<MatrixXR>(qp_data, row_size, col_size);
    }

    __device__ Real& detJ_ref(int elem_idx, int qp_idx) {
        return d_detJ_ref[elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx];
    }

    __device__ Real detJ_ref(int elem_idx, int qp_idx) const {
        return d_detJ_ref[elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx];
    }

    // ==============================================================

    __device__ const Map<VectorXR> gauss_xi_m() const {
        return Map<VectorXR>(d_gauss_xi_m, Quadrature::N_QP_7);
    }

    __device__ const Map<VectorXR> gauss_eta_m() const {
        return Map<VectorXR>(d_gauss_eta_m, Quadrature::N_QP_7);
    }

    __device__ const Map<VectorXR> gauss_zeta_m() const {
        return Map<VectorXR>(d_gauss_zeta_m, Quadrature::N_QP_3);
    }

    __device__ const Map<VectorXR> gauss_xi() const {
        return Map<VectorXR>(d_gauss_xi, Quadrature::N_QP_4);
    }

    __device__ const Map<VectorXR> gauss_eta() const {
        return Map<VectorXR>(d_gauss_eta, Quadrature::N_QP_4);
    }

    __device__ const Map<VectorXR> gauss_zeta() const {
        return Map<VectorXR>(d_gauss_zeta, Quadrature::N_QP_3);
    }

    __device__ const Map<VectorXR> weight_xi_m() const {
        return Map<VectorXR>(d_weight_xi_m, Quadrature::N_QP_7);
    }

    __device__ const Map<VectorXR> weight_eta_m() const {
        return Map<VectorXR>(d_weight_eta_m, Quadrature::N_QP_7);
    }

    __device__ const Map<VectorXR> weight_zeta_m() const {
        return Map<VectorXR>(d_weight_zeta_m, Quadrature::N_QP_3);
    }

    __device__ const Map<VectorXR> weight_xi() const {
        return Map<VectorXR>(d_weight_xi, Quadrature::N_QP_4);
    }

    __device__ const Map<VectorXR> weight_eta() const {
        return Map<VectorXR>(d_weight_eta, Quadrature::N_QP_4);
    }

    __device__ const Map<VectorXR> weight_zeta() const {
        return Map<VectorXR>(d_weight_zeta, Quadrature::N_QP_3);
    }

    // ==============================================================

    __device__ Map<VectorXR> x12_jac() {
        return Map<VectorXR>(d_x12_jac, n_coef);
    }

    __device__ Map<VectorXR> const x12_jac() const {
        return Map<VectorXR>(d_x12_jac, n_coef);
    }

    __device__ Map<VectorXR> y12_jac() {
        return Map<VectorXR>(d_y12_jac, n_coef);
    }

    __device__ Map<VectorXR> const y12_jac() const {
        return Map<VectorXR>(d_y12_jac, n_coef);
    }

    __device__ Map<VectorXR> z12_jac() {
        return Map<VectorXR>(d_z12_jac, n_coef);
    }

    __device__ Map<VectorXR> const z12_jac() const {
        return Map<VectorXR>(d_z12_jac, n_coef);
    }

    __device__ Map<VectorXR> x12() {
        return Map<VectorXR>(d_x12, n_coef);
    }

    __device__ Map<VectorXR> const x12() const {
        return Map<VectorXR>(d_x12, n_coef);
    }

    __device__ Map<VectorXR> y12() {
        return Map<VectorXR>(d_y12, n_coef);
    }

    __device__ Map<VectorXR> const y12() const {
        return Map<VectorXR>(d_y12, n_coef);
    }

    __device__ Map<VectorXR> z12() {
        return Map<VectorXR>(d_z12, n_coef);
    }

    __device__ Map<VectorXR> const z12() const {
        return Map<VectorXR>(d_z12, n_coef);
    }

    // Helper: gather 16 DOFs for an element using connectivity
    __device__ void gather_element_dofs(const Real* global, Map<MatrixXi> connectivity, int elem, Real* local) const {
        // Each element has 4 nodes, each node has 4 DOFs
        for (int n = 0; n < 4; ++n) {
            int node = connectivity(elem, n);
    #pragma unroll
            for (int d = 0; d < 4; ++d) {
                local[n * 4 + d] = global[node * 4 + d];
            }
        }
    }

    // Accessor for x12 for a given element (gathered by connectivity)
    __device__ void x12_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_x12, this->element_connectivity(), elem, buffer);
    }

    __device__ void x12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_x12_jac, this->element_connectivity(), elem, buffer);
    }

    // Accessor for y12 for a given element
    __device__ void y12_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_y12, this->element_connectivity(), elem, buffer);
    }

    __device__ void y12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_y12_jac, this->element_connectivity(), elem, buffer);
    }

    // Accessor for z12 for a given element
    __device__ void z12_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_z12, this->element_connectivity(), elem, buffer);
    }

    __device__ void z12_jac_elem(int elem, Real* buffer) const {
        gather_element_dofs(d_z12_jac, this->element_connectivity(), elem, buffer);
    }

    __device__ Map<MatrixXi> element_connectivity() const {
        return Map<MatrixXi>(d_element_connectivity, n_beam, 4);
    }

    __device__ Map<MatrixXR> F(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> F(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<MatrixXR> P(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    // Time-derivative of deformation gradient (viscous computation)
    __device__ Map<MatrixXR> Fdot(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> Fdot(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    // Viscous Piola stress storage
    __device__ Map<MatrixXR> P_vis(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P_vis(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_TOTAL_QP_4_4_3 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<VectorXR> f_int(int global_node_idx) {
        return Map<VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ const Map<VectorXR> f_int(int global_node_idx) const {
        return Map<VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ Map<VectorXR> f_int() {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ const Map<VectorXR> f_int() const {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }

    __device__ Map<VectorXR> f_ext(int global_node_idx) {
        return Map<VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ const Map<VectorXR> f_ext(int global_node_idx) const {
        return Map<VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    __device__ Map<VectorXR> f_ext() {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ const Map<VectorXR> f_ext() const {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }

    __device__ Map<VectorXR> constraint() {
        return Map<VectorXR>(d_constraint, n_constraint);
    }

    __device__ const Map<VectorXR> constraint() const {
        return Map<VectorXR>(d_constraint, n_constraint);
    }

    __device__ const Real* constraint_rhs() const {
        return d_constraint_rhs;
    }

    __device__ int constraint_mode_device() const {
        return constraint_mode;
    }

    __device__ Map<VectorXi> fixed_nodes() {
        return Map<VectorXi>(d_fixed_nodes, n_constraint / 3);
    }

    // ================================

    __device__ Real L(int elem_idx) const {
        return d_L[elem_idx];
    }

    __device__ Real W(int elem_idx) const {
        return d_W[elem_idx];
    }

    __device__ Real H(int elem_idx) const {
        return d_H[elem_idx];
    }

    __device__ Real rho0() const {
        return *d_rho0;
    }

    __device__ Real nu() const {
        return *d_nu;
    }

    __device__ Real E() const {
        return *d_E;
    }

    __device__ Real lambda() const {
        return *d_lambda;
    }

    __device__ Real mu() const {
        return *d_mu;
    }

    __device__ int material_model() const {
        return *d_material_model;
    }

    __device__ Real mu10() const {
        return *d_mu10;
    }

    __device__ Real mu01() const {
        return *d_mu01;
    }

    __device__ Real kappa() const {
        return *d_kappa;
    }

    __device__ Real eta_damp() const {
        return *d_eta_damp;
    }

    __device__ Real lambda_damp() const {
        return *d_lambda_damp;
    }

    //===========================================

    __device__ int* csr_offsets() {
        return d_csr_offsets;
    }

    __device__ int* csr_columns() {
        return d_csr_columns;
    }

    __device__ Real* csr_values() {
        return d_csr_values;
    }

    __device__ int* cj_csr_offsets() {
        return d_cj_csr_offsets;
    }

    __device__ int* cj_csr_columns() {
        return d_cj_csr_columns;
    }

    __device__ Real* cj_csr_values() {
        return d_cj_csr_values;
    }

    __device__ int* j_csr_offsets() {
        return d_j_csr_offsets;
    }

    __device__ int* j_csr_columns() {
        return d_j_csr_columns;
    }

    __device__ Real* j_csr_values() {
        return d_j_csr_values;
    }

    __device__ int nnz() {
        return *d_nnz;
    }

    __device__ int gpu_n_beam() const {
        return n_beam;
    }

    __device__ int gpu_n_coef() const {
        return n_coef;
    }

    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }

#endif

    __host__ __device__ int get_n_beam() const {
        return n_beam;
    }
    __host__ __device__ int get_n_coef() const {
        return n_coef;
    }
    __host__ __device__ int get_n_constraint() const {
        return n_constraint;
    }

    // Constructors
    //
    // 1) "Strip" constructor (backward compatible with existing tests):
    //    assumes a chain of quad elements where each new element introduces 2 new
    //    nodes (8 coefficients), i.e. n_nodes = 4 + 2*(n_beam-1).
    GPU_ANCF3443_Data(int num_beams) : n_beam(num_beams) {
        n_nodes = 4 + 2 * (n_beam - 1);
        n_coef = 4 * n_nodes;
        type = TYPE_3443;
    }

    // 2) General mesh constructor: explicit node/element counts.
    GPU_ANCF3443_Data(int num_nodes, int num_elements) : n_beam(num_elements), n_nodes(num_nodes) {
        n_coef = 4 * n_nodes;
        type = TYPE_3443;
    }

    void Initialize() {
        // Long arrays: use DualArray (manages both pinned host and device memory).
        da_B_inv.resize(n_beam * Quadrature::N_SHAPE_3443 * Quadrature::N_SHAPE_3443);
        da_B_inv.BindDevicePointer(&d_B_inv);
        da_grad_N_ref.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3 * Quadrature::N_SHAPE_3443 * 3);
        da_grad_N_ref.BindDevicePointer(&d_grad_N_ref);
        da_detJ_ref.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3);
        da_detJ_ref.BindDevicePointer(&d_detJ_ref);

        da_gauss_xi_m.resize(Quadrature::N_QP_7);
        da_gauss_xi_m.BindDevicePointer(&d_gauss_xi_m);
        da_gauss_eta_m.resize(Quadrature::N_QP_7);
        da_gauss_eta_m.BindDevicePointer(&d_gauss_eta_m);
        da_gauss_zeta_m.resize(Quadrature::N_QP_3);
        da_gauss_zeta_m.BindDevicePointer(&d_gauss_zeta_m);
        da_gauss_xi.resize(Quadrature::N_QP_4);
        da_gauss_xi.BindDevicePointer(&d_gauss_xi);
        da_gauss_eta.resize(Quadrature::N_QP_4);
        da_gauss_eta.BindDevicePointer(&d_gauss_eta);
        da_gauss_zeta.resize(Quadrature::N_QP_3);
        da_gauss_zeta.BindDevicePointer(&d_gauss_zeta);

        da_weight_xi_m.resize(Quadrature::N_QP_7);
        da_weight_xi_m.BindDevicePointer(&d_weight_xi_m);
        da_weight_eta_m.resize(Quadrature::N_QP_7);
        da_weight_eta_m.BindDevicePointer(&d_weight_eta_m);
        da_weight_zeta_m.resize(Quadrature::N_QP_3);
        da_weight_zeta_m.BindDevicePointer(&d_weight_zeta_m);
        da_weight_xi.resize(Quadrature::N_QP_4);
        da_weight_xi.BindDevicePointer(&d_weight_xi);
        da_weight_eta.resize(Quadrature::N_QP_4);
        da_weight_eta.BindDevicePointer(&d_weight_eta);
        da_weight_zeta.resize(Quadrature::N_QP_3);
        da_weight_zeta.BindDevicePointer(&d_weight_zeta);

        da_x12_jac.resize(n_coef);
        da_x12_jac.BindDevicePointer(&d_x12_jac);
        da_y12_jac.resize(n_coef);
        da_y12_jac.BindDevicePointer(&d_y12_jac);
        da_z12_jac.resize(n_coef);
        da_z12_jac.BindDevicePointer(&d_z12_jac);
        da_x12.resize(n_coef);
        da_x12.BindDevicePointer(&d_x12);
        da_y12.resize(n_coef);
        da_y12.BindDevicePointer(&d_y12);
        da_z12.resize(n_coef);
        da_z12.BindDevicePointer(&d_z12);

        da_element_connectivity.resize(n_beam * 4);
        da_element_connectivity.BindDevicePointer(&d_element_connectivity);

        da_F.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3);
        da_F.BindDevicePointer(&d_F);
        da_P.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3);
        da_P.BindDevicePointer(&d_P);
        da_Fdot.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3);
        da_Fdot.BindDevicePointer(&d_Fdot);
        da_P_vis.resize(n_beam * Quadrature::N_TOTAL_QP_4_4_3 * 3 * 3);
        da_P_vis.BindDevicePointer(&d_P_vis);
        da_f_int.resize(n_coef * 3);
        da_f_int.BindDevicePointer(&d_f_int);
        da_f_ext.resize(n_coef * 3);
        da_f_ext.BindDevicePointer(&d_f_ext);

        // Scalar damping parameters: single values on device only.
        MOPHI_GPU_CALL(cudaMalloc(&d_eta_damp, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda_damp, sizeof(Real)));

        // copy struct to device
        MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_ANCF3443_Data)));

        // beam dimension data
        da_H.resize(n_beam);
        da_H.BindDevicePointer(&d_H);
        da_W.resize(n_beam);
        da_W.BindDevicePointer(&d_W);
        da_L.resize(n_beam);
        da_L.BindDevicePointer(&d_L);

        // Scalar material parameters: single values on device only.
        MOPHI_GPU_CALL(cudaMalloc(&d_rho0, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_nu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_E, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_material_model, sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu10, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu01, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_kappa, sizeof(Real)));
    }

    void Setup(const VectorXR& length,
               const VectorXR& width,
               const VectorXR& height,
               const VectorXR& gauss_xi_m,
               const VectorXR& gauss_eta_m,
               const VectorXR& gauss_zeta_m,
               const VectorXR& gauss_xi,
               const VectorXR& gauss_eta,
               const VectorXR& gauss_zeta,
               const VectorXR& weight_xi_m,
               const VectorXR& weight_eta_m,
               const VectorXR& weight_zeta_m,
               const VectorXR& weight_xi,
               const VectorXR& weight_eta,
               const VectorXR& weight_zeta,
               const VectorXR& h_x12,
               const VectorXR& h_y12,
               const VectorXR& h_z12,
               const MatrixXi& element_connectivity) {
        if (is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data is already set up.");
            return;
        }

        if (length.size() != n_beam || width.size() != n_beam || height.size() != n_beam) {
            MOPHI_ERROR("GPU_ANCF3443_Data::Setup: length/width/height must have size n_beam.");
            return;
        }

        VectorXR h_B_inv_flat;
        try {
            ANCFCPUUtils::ANCF3443_B12_matrix_flat_per_element(length, width, height, h_B_inv_flat,
                                                               Quadrature::N_SHAPE_3443);
        } catch (const std::exception& e) {
            MOPHI_ERROR("GPU_ANCF3443_Data::Setup: failed to build per-element B_inv: %s", e.what());
            return;
        }
        const int n_binv = static_cast<int>(h_B_inv_flat.size());

        std::copy(h_B_inv_flat.data(), h_B_inv_flat.data() + n_binv, da_B_inv.host());
        da_B_inv.ToDevice();

        std::copy(gauss_xi_m.data(), gauss_xi_m.data() + Quadrature::N_QP_7, da_gauss_xi_m.host());
        da_gauss_xi_m.ToDevice();
        std::copy(gauss_eta_m.data(), gauss_eta_m.data() + Quadrature::N_QP_7, da_gauss_eta_m.host());
        da_gauss_eta_m.ToDevice();
        std::copy(gauss_zeta_m.data(), gauss_zeta_m.data() + Quadrature::N_QP_3, da_gauss_zeta_m.host());
        da_gauss_zeta_m.ToDevice();
        std::copy(gauss_xi.data(), gauss_xi.data() + Quadrature::N_QP_4, da_gauss_xi.host());
        da_gauss_xi.ToDevice();
        std::copy(gauss_eta.data(), gauss_eta.data() + Quadrature::N_QP_4, da_gauss_eta.host());
        da_gauss_eta.ToDevice();
        std::copy(gauss_zeta.data(), gauss_zeta.data() + Quadrature::N_QP_3, da_gauss_zeta.host());
        da_gauss_zeta.ToDevice();

        std::copy(weight_xi_m.data(), weight_xi_m.data() + Quadrature::N_QP_7, da_weight_xi_m.host());
        da_weight_xi_m.ToDevice();
        std::copy(weight_eta_m.data(), weight_eta_m.data() + Quadrature::N_QP_7, da_weight_eta_m.host());
        da_weight_eta_m.ToDevice();
        std::copy(weight_zeta_m.data(), weight_zeta_m.data() + Quadrature::N_QP_3, da_weight_zeta_m.host());
        da_weight_zeta_m.ToDevice();
        std::copy(weight_xi.data(), weight_xi.data() + Quadrature::N_QP_4, da_weight_xi.host());
        da_weight_xi.ToDevice();
        std::copy(weight_eta.data(), weight_eta.data() + Quadrature::N_QP_4, da_weight_eta.host());
        da_weight_eta.ToDevice();
        std::copy(weight_zeta.data(), weight_zeta.data() + Quadrature::N_QP_3, da_weight_zeta.host());
        da_weight_zeta.ToDevice();

        std::copy(h_x12.data(), h_x12.data() + n_coef, da_x12_jac.host());
        da_x12_jac.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_y12_jac.host());
        da_y12_jac.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_z12_jac.host());
        da_z12_jac.ToDevice();
        std::copy(h_x12.data(), h_x12.data() + n_coef, da_x12.host());
        da_x12.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_y12.host());
        da_y12.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_z12.host());
        da_z12.ToDevice();

        std::copy(element_connectivity.data(), element_connectivity.data() + n_beam * 4,
                  da_element_connectivity.host());
        da_element_connectivity.ToDevice();

        da_f_int.SetVal(Real(0));
        da_f_int.ToDevice();

        da_F.SetVal(Real(0));
        da_F.ToDevice();
        da_P.SetVal(Real(0));
        da_P.ToDevice();
        da_Fdot.SetVal(Real(0));
        da_Fdot.ToDevice();
        da_P_vis.SetVal(Real(0));
        da_P_vis.ToDevice();

        std::copy(height.data(), height.data() + n_beam, da_H.host());
        da_H.ToDevice();
        std::copy(width.data(), width.data() + n_beam, da_W.host());
        da_W.ToDevice();
        std::copy(length.data(), length.data() + n_beam, da_L.host());
        da_L.ToDevice();

        Real rho0 = 0.0;
        Real nu = 0.0;
        Real E = 0.0;
        Real mu = E / (2 * (1 + nu));                        // Shear modulus μ
        Real lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));  // Lamé’s first parameter λ
        Real eta_damp = 0.0;
        Real lambda_damp = 0.0;
        int material_model = MATERIAL_MODEL_SVK;
        Real mu10 = 0.0;
        Real mu01 = 0.0;
        Real kappa = 0.0;

        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(Real), cudaMemcpyHostToDevice));
        // copy damping scalars to device (single Real each)
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data), cudaMemcpyHostToDevice));

        is_setup = true;
        is_reference_precomputed = false;
    }

    void Setup(Real length,
               Real width,
               Real height,
               const VectorXR& gauss_xi_m,
               const VectorXR& gauss_eta_m,
               const VectorXR& gauss_zeta_m,
               const VectorXR& gauss_xi,
               const VectorXR& gauss_eta,
               const VectorXR& gauss_zeta,
               const VectorXR& weight_xi_m,
               const VectorXR& weight_eta_m,
               const VectorXR& weight_zeta_m,
               const VectorXR& weight_xi,
               const VectorXR& weight_eta,
               const VectorXR& weight_zeta,
               const VectorXR& h_x12,
               const VectorXR& h_y12,
               const VectorXR& h_z12,
               const MatrixXi& element_connectivity) {
        VectorXR lengths = VectorXR::Constant(n_beam, length);
        VectorXR widths = VectorXR::Constant(n_beam, width);
        VectorXR heights = VectorXR::Constant(n_beam, height);
        Setup(lengths, widths, heights, gauss_xi_m, gauss_eta_m, gauss_zeta_m, gauss_xi, gauss_eta, gauss_zeta,
              weight_xi_m, weight_eta_m, weight_zeta_m, weight_xi, weight_eta, weight_zeta, h_x12, h_y12, h_z12,
              element_connectivity);
    }

    /**
     * Set reference density (used for mass/inertial terms).
     */
    void SetDensity(Real rho0) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data must be set up before setting density.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice));
    }

    /**
     * Set Kelvin-Voigt damping parameters.
     * eta_damp: shear-like damping coefficient
     * lambda_damp: volumetric-like damping coefficient
     */
    void SetDamping(Real eta_damp, Real lambda_damp) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data must be set up before setting damping.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
    }

    /**
     * Select Saint Venant-Kirchhoff (SVK) material model using current E/nu.
     */
    void SetSVK() {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_SVK;
        Real mu10 = 0.0;
        Real mu01 = 0.0;
        Real kappa = 0.0;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetSVK(Real E, Real nu) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data must be set up before setting material.");
            return;
        }

        MOPHI_GPU_CALL(cudaMemcpy(d_nu, &nu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_E, &E, sizeof(Real), cudaMemcpyHostToDevice));

        Real mu = E / (2 * (1 + nu));
        Real lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu, &mu, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda, &lambda, sizeof(Real), cudaMemcpyHostToDevice));

        SetSVK();
    }

    /**
     * Set compressible Mooney-Rivlin parameters.
     * mu10, mu01: isochoric Mooney-Rivlin coefficients
     * kappa: volumetric penalty (bulk-modulus-like) coefficient
     */
    void SetMooneyRivlin(Real mu10, Real mu01, Real kappa) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetExternalForce(const VectorXR& f_ext) {
        if (f_ext.size() != n_coef * 3) {
            MOPHI_ERROR("External force vector size mismatch.");
            return;
        }
        std::copy(f_ext.data(), f_ext.data() + n_coef * 3, da_f_ext.host());
        da_f_ext.ToDevice();
    }
    void SetExternalForce(const VectorReal3& f_ext) {
        if (static_cast<int>(f_ext.size()) != n_coef) {
            MOPHI_ERROR("External force vector size mismatch.");
            return;
        }
        VectorXR flat = FlattenVectorReal3(f_ext);
        SetExternalForce(flat);
    }

    void SetNodalFixed(const VectorXi& fixed_nodes) {
        if (is_constraints_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data CONSTRAINT is already set up.");
            return;
        }

        n_constraint = fixed_nodes.size() * 3;
        constraint_mode = kConstraintFixedCoefficients;

        da_constraint.resize(n_constraint);
        da_constraint.BindDevicePointer(&d_constraint);
        da_fixed_nodes.resize(fixed_nodes.size());
        da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);

        da_constraint.SetVal(Real(0));
        da_constraint.ToDevice();
        std::copy(fixed_nodes.data(), fixed_nodes.data() + fixed_nodes.size(), da_fixed_nodes.host());
        da_fixed_nodes.ToDevice();

        is_constraints_setup = true;
        if (d_data) {
            MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data), cudaMemcpyHostToDevice));
        }
    }

    // General linear constraints:
    //   c = J * x - rhs
    // where x is the flattened coefficient vector in solver ordering
    // (coef_index major, then xyz component), J is CSR (rows = constraints,
    // cols = n_coef*3), and rhs is per-row target.
    //
    // This uploads both J (CSR) and J^T (CSR).
    void SetLinearConstraintsCSR(const std::vector<int>& j_offsets,
                                 const std::vector<int>& j_columns,
                                 const std::vector<Real>& j_values,
                                 const VectorXR& rhs) {
        if (is_constraints_setup) {
            MOPHI_ERROR("GPU_ANCF3443_Data CONSTRAINT is already set up.");
            return;
        }
        if (j_offsets.empty() || j_offsets.front() != 0) {
            MOPHI_ERROR("SetLinearConstraintsCSR: invalid offsets.");
            return;
        }
        if (static_cast<int>(rhs.size()) + 1 != static_cast<int>(j_offsets.size())) {
            MOPHI_ERROR("SetLinearConstraintsCSR: offsets/rhs size mismatch.");
            return;
        }
        if (j_columns.size() != j_values.size()) {
            MOPHI_ERROR("SetLinearConstraintsCSR: columns/values size mismatch.");
            return;
        }

        const int nnz = static_cast<int>(j_columns.size());
        if (j_offsets.back() != nnz) {
            MOPHI_ERROR("SetLinearConstraintsCSR: offsets.back != nnz.");
            return;
        }

        const int n_dofs = n_coef * 3;
        for (int c : j_columns) {
            if (c < 0 || c >= n_dofs) {
                MOPHI_ERROR("SetLinearConstraintsCSR: column out of range.");
                return;
            }
        }

        d_fixed_nodes = nullptr;
        n_constraint = static_cast<int>(rhs.size());
        constraint_mode = kConstraintLinearCSR;

        da_constraint.resize(n_constraint);
        da_constraint.BindDevicePointer(&d_constraint);
        da_constraint.SetVal(Real(0));
        da_constraint.ToDevice();

        MOPHI_GPU_CALL(cudaMalloc(&d_constraint_rhs, n_constraint * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMemcpy(d_constraint_rhs, rhs.data(), n_constraint * sizeof(Real), cudaMemcpyHostToDevice));

        // J (CSR): rows=constraints, cols=dofs.
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_offsets, static_cast<size_t>(n_constraint + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_j_nnz, sizeof(int)));

        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_offsets, j_offsets.data(),
                                  static_cast<size_t>(n_constraint + 1) * sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_columns, j_columns.data(), static_cast<size_t>(nnz) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_csr_values, j_values.data(), static_cast<size_t>(nnz) * sizeof(Real),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_j_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
        is_j_csr_setup = true;

        // Build J^T (CSR) on host: rows=dofs, cols=constraints.
        std::vector<int> jt_offsets(static_cast<size_t>(n_dofs + 1), 0);
        std::vector<int> jt_columns(static_cast<size_t>(nnz), 0);
        std::vector<Real> jt_values(static_cast<size_t>(nnz), 0.0);

        std::vector<int> counts(static_cast<size_t>(n_dofs), 0);
        for (int idx = 0; idx < nnz; ++idx) {
            counts[static_cast<size_t>(j_columns[static_cast<size_t>(idx)])] += 1;
        }
        int running = 0;
        for (int i = 0; i < n_dofs; ++i) {
            jt_offsets[static_cast<size_t>(i)] = running;
            running += counts[static_cast<size_t>(i)];
        }
        jt_offsets[static_cast<size_t>(n_dofs)] = nnz;

        std::vector<int> positions = jt_offsets;
        for (int row = 0; row < n_constraint; ++row) {
            const int start = j_offsets[static_cast<size_t>(row)];
            const int end = j_offsets[static_cast<size_t>(row + 1)];
            for (int idx = start; idx < end; ++idx) {
                const int col = j_columns[static_cast<size_t>(idx)];
                const int out = positions[static_cast<size_t>(col)]++;
                jt_columns[static_cast<size_t>(out)] = row;
                jt_values[static_cast<size_t>(out)] = j_values[static_cast<size_t>(idx)];
            }
        }

        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_offsets, static_cast<size_t>(n_dofs + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_columns, static_cast<size_t>(nnz) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_csr_values, static_cast<size_t>(nnz) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_cj_nnz, sizeof(int)));

        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_offsets, jt_offsets.data(), static_cast<size_t>(n_dofs + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_columns, jt_columns.data(), static_cast<size_t>(nnz) * sizeof(int),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_csr_values, jt_values.data(), static_cast<size_t>(nnz) * sizeof(Real),
                                  cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_cj_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice));
        is_cj_csr_setup = true;

        is_constraints_setup = true;
        if (d_data) {
            MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_ANCF3443_Data), cudaMemcpyHostToDevice));
        }
    }

    // For kConstraintLinearCSR, update only RHS values (keeps J/J^T and sparsity
    // fixed).
    void UpdateLinearConstraintRHS(const VectorXR& rhs) {
        if (!is_constraints_setup || n_constraint == 0) {
            MOPHI_ERROR("UpdateLinearConstraintRHS: constraints not set up.");
            return;
        }
        if (constraint_mode != kConstraintLinearCSR) {
            MOPHI_ERROR("UpdateLinearConstraintRHS: constraint mode is not CSR.");
            return;
        }
        if (static_cast<int>(rhs.size()) != n_constraint) {
            MOPHI_ERROR("UpdateLinearConstraintRHS: size mismatch.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_constraint_rhs, rhs.data(), static_cast<size_t>(n_constraint) * sizeof(Real),
                                  cudaMemcpyHostToDevice));
    }

    // Free memory
    void Destroy() {
        // Long arrays managed by DualArrays
        da_B_inv.free();
        da_grad_N_ref.free();
        da_detJ_ref.free();

        da_gauss_xi_m.free();
        da_gauss_eta_m.free();
        da_gauss_zeta_m.free();
        da_gauss_xi.free();
        da_gauss_eta.free();
        da_gauss_zeta.free();
        da_weight_xi_m.free();
        da_weight_eta_m.free();
        da_weight_zeta_m.free();
        da_weight_xi.free();
        da_weight_eta.free();
        da_weight_zeta.free();

        da_x12_jac.free();
        da_y12_jac.free();
        da_z12_jac.free();
        da_x12.free();
        da_y12.free();
        da_z12.free();

        da_element_connectivity.free();

        if (is_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_nnz));
        }

        if (is_cj_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_cj_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_cj_nnz));
            is_cj_csr_setup = false;
        }

        if (is_j_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_j_nnz));
            is_j_csr_setup = false;
        }

        da_F.free();
        da_P.free();
        da_Fdot.free();
        da_P_vis.free();
        da_f_int.free();
        da_f_ext.free();
        MOPHI_GPU_CALL(cudaFree(d_eta_damp));
        MOPHI_GPU_CALL(cudaFree(d_lambda_damp));

        da_H.free();
        da_W.free();
        da_L.free();

        MOPHI_GPU_CALL(cudaFree(d_rho0));
        MOPHI_GPU_CALL(cudaFree(d_nu));
        MOPHI_GPU_CALL(cudaFree(d_E));
        MOPHI_GPU_CALL(cudaFree(d_lambda));
        MOPHI_GPU_CALL(cudaFree(d_mu));
        MOPHI_GPU_CALL(cudaFree(d_material_model));
        MOPHI_GPU_CALL(cudaFree(d_mu10));
        MOPHI_GPU_CALL(cudaFree(d_mu01));
        MOPHI_GPU_CALL(cudaFree(d_kappa));

        MOPHI_GPU_CALL(cudaFree(d_data));

        if (is_constraints_setup) {
            da_constraint.free();
            da_fixed_nodes.free();
            if (d_constraint_rhs) {
                MOPHI_GPU_CALL(cudaFree(d_constraint_rhs));
                d_constraint_rhs = nullptr;
            }
        }
    }

    void CalcDsDuPre();

    void CalcMassMatrix();

    void BuildMassCSRPattern();

    void ConvertToCSR_ConstraintJacT();

    void BuildConstraintJacobianTransposeCSR() {
        ConvertToCSR_ConstraintJacT();
    }

    void ConvertToCSR_ConstraintJac();

    void BuildConstraintJacobianCSR() {
        ConvertToCSR_ConstraintJac();
    }

    void CalcP();

    void CalcInternalForce();

    void CalcConstraintData() override;

    void PrintDsDuPre();

    void RetrieveConnectivityToCPU(MatrixXi& connectivity);

    void RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ);

    void RetrieveMassCSRToCPU(std::vector<int>& offsets, std::vector<int>& columns, std::vector<Real>& values);

    void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>& deformation_gradient);

    void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F);

    void RetrieveInternalForceToCPU(VectorXR& internal_force);
    void RetrieveInternalForceToCPU(VectorReal3& internal_force) {
        VectorXR flat;
        RetrieveInternalForceToCPU(flat);
        if (!UnflattenVectorReal3(flat, internal_force)) {
            MOPHI_ERROR(
                "GPU_ANCF3443_Data::RetrieveInternalForceToCPU: internal force size (%d) is not divisible by 3.",
                static_cast<int>(flat.size()));
            return;
        }
    }

    void RetrieveConstraintDataToCPU(VectorXR& constraint);

    void RetrieveConstraintJacobianToCPU(MatrixXR& constraint_jac);

    void RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12);

    // ── Legacy accessors (kept for backward compatibility) ───────────────────
    // Prefer the virtual base-class methods IsConstraintSetup() and
    // GetConstraintDevicePtr() when working through an ElementBase pointer.
    Real* Get_Constraint_Ptr() {
        return d_constraint;
    }

    bool Get_Is_Constraint_Setup() {
        return is_constraints_setup;
    }

    int GetConstraintMode() const {
        return constraint_mode;
    }

    // ── ElementBase dispatch helpers ─────────────────────────────────────────

    ElementBase* GetDevicePtr() override {
        return d_data;
    }

    void PrepareSolverData() override {
        CalcDsDuPre();
    }

    bool IsConstraintSetup() override {
        return is_constraints_setup;
    }

    Real* GetConstraintDevicePtr() override {
        return d_constraint;
    }

    void RetrieveConstraintJacobianCSRToCPU(std::vector<int>& offsets,
                                            std::vector<int>& columns,
                                            std::vector<Real>& values);

    GPU_ANCF3443_Data* d_data;  // Storing GPU copy of SAPGPUData

    int n_beam;   // number of elements
    int n_nodes;  // number of nodes
    int n_coef;   // number of coefficients (= 4 * n_nodes)
    int n_constraint;

  private:
    // DualArrays for long arrays (manage both pinned host and device memory).
    // The raw device pointers below are bound to these DualArrays via
    // BindDevicePointer so that GPU kernels can access data through them.
    mophi::DualArray<Real> da_B_inv;
    mophi::DualArray<Real> da_grad_N_ref;
    mophi::DualArray<Real> da_detJ_ref;
    mophi::DualArray<Real> da_gauss_xi_m, da_gauss_eta_m, da_gauss_zeta_m;
    mophi::DualArray<Real> da_gauss_xi, da_gauss_eta, da_gauss_zeta;
    mophi::DualArray<Real> da_weight_xi_m, da_weight_eta_m, da_weight_zeta_m;
    mophi::DualArray<Real> da_weight_xi, da_weight_eta, da_weight_zeta;
    mophi::DualArray<Real> da_x12_jac, da_y12_jac, da_z12_jac;
    mophi::DualArray<Real> da_x12, da_y12, da_z12;
    mophi::DualArray<int> da_element_connectivity;
    mophi::DualArray<Real> da_F, da_P, da_Fdot, da_P_vis;
    mophi::DualArray<Real> da_H, da_W, da_L;
    mophi::DualArray<Real> da_f_int, da_f_ext;
    mophi::DualArray<Real> da_constraint;
    mophi::DualArray<int> da_fixed_nodes;

    // Raw device pointers for GPU kernel access (managed by DualArrays above
    // via BindDevicePointer; scalars and CSR arrays kept as raw pointers).
    Real* d_B_inv;
    Real* d_grad_N_ref;  // (n_beam, N_QP, N_SHAPE, 3)
    Real* d_detJ_ref;    // (n_beam, N_QP)
    Real *d_gauss_xi_m, *d_gauss_eta_m, *d_gauss_zeta_m, *d_gauss_xi, *d_gauss_eta, *d_gauss_zeta;
    Real *d_weight_xi_m, *d_weight_eta_m, *d_weight_zeta_m, *d_weight_xi, *d_weight_eta, *d_weight_zeta;

    Real *d_x12_jac, *d_y12_jac, *d_z12_jac;
    Real *d_x12, *d_y12, *d_z12;

    int* d_element_connectivity;
    int *d_csr_offsets, *d_csr_columns;
    Real* d_csr_values;
    int* d_nnz;

    Real *d_F, *d_P;
    Real *d_Fdot, *d_P_vis;

    // per-element damping parameters (scalar device values)
    Real *d_eta_damp, *d_lambda_damp;

    Real *d_H, *d_W, *d_L;

    Real *d_rho0, *d_nu, *d_E, *d_lambda, *d_mu;
    int* d_material_model;
    Real *d_mu10, *d_mu01, *d_kappa;

    Real* d_constraint;
    Real* d_constraint_rhs = nullptr;
    int* d_fixed_nodes;

    // Constraint Jacobian J^T in CSR format (dynamically sized)
    int *d_cj_csr_offsets, *d_cj_csr_columns;
    Real* d_cj_csr_values;
    int* d_cj_nnz;

    // Constraint Jacobian J in CSR format (dynamically sized)
    int *d_j_csr_offsets, *d_j_csr_columns;
    Real* d_j_csr_values;
    int* d_j_nnz;

    // force related parameters
    Real *d_f_int, *d_f_ext;

    bool is_setup = false;
    bool is_constraints_setup = false;
    bool is_csr_setup = false;
    bool is_cj_csr_setup = false;
    bool is_j_csr_setup = false;
    bool is_reference_precomputed = false;

    int constraint_mode = kConstraintNone;
};

}  // namespace feris
