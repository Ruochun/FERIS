#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <iostream>
#include <vector>

#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "../materials/MaterialModel.cuh"
#include "ElementBase.h"
#include "../types.h"
#include <MoPhiEssentials.h>

// GPU data structure for 4-node linear tetrahedral (TET4) elements.
#pragma once

namespace tlfea {

struct GPU_FEAT4_Data : public ElementBase {
    // Static compile-time constants for template dispatch
    static constexpr int N_NODES_PER_ELEM = Quadrature::N_NODE_T4_4;
    static constexpr int N_QP_PER_ELEM = Quadrature::N_QP_T4_1;

#if defined(__CUDACC__)

    __device__ Map<MatrixXi> element_connectivity() const {
        return Map<MatrixXi>(d_element_connectivity, n_elem, Quadrature::N_NODE_T4_4);
    }

    __device__ Map<MatrixXR> grad_N_ref(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_grad_N_ref + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 4 * 3, 4, 3);
    }

    __device__ const Map<MatrixXR> grad_N_ref(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_grad_N_ref + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 4 * 3, 4, 3);
    }

    __device__ Real& detJ_ref(int elem_idx, int qp_idx) {
        return d_detJ_ref[elem_idx * Quadrature::N_QP_T4_1 + qp_idx];
    }

    __device__ Real detJ_ref(int elem_idx, int qp_idx) const {
        return d_detJ_ref[elem_idx * Quadrature::N_QP_T4_1 + qp_idx];
    }

    __device__ Real tet1pt_x(int qp_idx) {
        return d_tet1pt_x[qp_idx];
    }

    __device__ Real tet1pt_y(int qp_idx) {
        return d_tet1pt_y[qp_idx];
    }

    __device__ Real tet1pt_z(int qp_idx) {
        return d_tet1pt_z[qp_idx];
    }

    __device__ Real tet1pt_weights(int qp_idx) {
        return d_tet1pt_weights[qp_idx];
    }

    __device__ Map<VectorXR> x12() {
        return Map<VectorXR>(d_h_x12, n_coef);
    }

    __device__ Map<VectorXR> const x12() const {
        return Map<VectorXR>(d_h_x12, n_coef);
    }

    __device__ Map<VectorXR> y12() {
        return Map<VectorXR>(d_h_y12, n_coef);
    }

    __device__ Map<VectorXR> const y12() const {
        return Map<VectorXR>(d_h_y12, n_coef);
    }

    __device__ Map<VectorXR> z12() {
        return Map<VectorXR>(d_h_z12, n_coef);
    }

    __device__ Map<VectorXR> const z12() const {
        return Map<VectorXR>(d_h_z12, n_coef);
    }

    __device__ Map<VectorXR> const x12_jac() const {
        return Map<VectorXR>(d_h_x12_jac, n_coef);
    }

    __device__ Map<VectorXR> const y12_jac() const {
        return Map<VectorXR>(d_h_y12_jac, n_coef);
    }

    __device__ Map<VectorXR> const z12_jac() const {
        return Map<VectorXR>(d_h_z12_jac, n_coef);
    }

    __device__ Map<MatrixXR> F(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> F(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_F + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<MatrixXR> P(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<MatrixXR> Fdot(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> Fdot(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_Fdot + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ Map<MatrixXR> P_vis(int elem_idx, int qp_idx) {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
    }

    __device__ const Map<MatrixXR> P_vis(int elem_idx, int qp_idx) const {
        return Map<MatrixXR>(d_P_vis + (elem_idx * Quadrature::N_QP_T4_1 + qp_idx) * 9, 3, 3);
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

    __device__ Map<VectorXi> fixed_nodes() {
        return Map<VectorXi>(d_fixed_nodes, n_constraint / 3);
    }

    // ================================
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

    __device__ Real eta_damp() const {
        return *d_eta_damp;
    }

    __device__ Real lambda_damp() const {
        return *d_lambda_damp;
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

    __device__ int gpu_n_elem() const {
        return n_elem;
    }

    __device__ int gpu_n_coef() const {
        return n_coef;
    }

    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }

    // ======================================================

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
#endif

    __host__ __device__ int get_n_elem() const {
        return n_elem;
    }
    __host__ __device__ int get_n_coef() const {
        return n_coef;
    }
    __host__ __device__ int get_n_constraint() const {
        return n_constraint;
    }

    __host__ __device__ int get_n_beam() const override {
        return n_elem;
    }

    void CalcDnDuPre();

    void CalcMassMatrix() override;

    void BuildMassCSRPattern();

    void ConvertToCSR_ConstraintJacT();

    void BuildConstraintJacobianTransposeCSR() {
        ConvertToCSR_ConstraintJacT();
    }

    void ConvertToCSR_ConstraintJac();

    void BuildConstraintJacobianCSR() {
        ConvertToCSR_ConstraintJac();
    }

    void CalcInternalForce() override;

    void CalcConstraintData() override;

    void CalcP() override;

    void RetrieveMassCSRToCPU(std::vector<int>& offsets, std::vector<int>& columns, std::vector<Real>& values);

    void RetrieveInternalForceToCPU(VectorXR& internal_force) override;

    void RetrieveExternalForceToCPU(VectorXR& external_force);

    void RetrieveConstraintDataToCPU(VectorXR& constraint) override {}

    void RetrieveConstraintJacobianToCPU(MatrixXR& constraint_jac) override {}

    void RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12) override;

    void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>& deformation_gradient) override {}

    void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F) override;

    void RetrieveDnDuPreToCPU(std::vector<std::vector<MatrixXR>>& dn_du_pre);

    void RetrieveDetJToCPU(std::vector<std::vector<Real>>& detJ);

    void RetrieveConnectivityToCPU(MatrixXi& connectivity);

    // Constructor
    GPU_FEAT4_Data(int num_elements, int num_nodes) : n_elem(num_elements), n_coef(num_nodes), n_constraint(0) {
        type = TYPE_T4;
    }

    void Initialize() {
        da_h_x12.resize(n_coef);
        da_h_x12.BindDevicePointer(&d_h_x12);
        da_h_y12.resize(n_coef);
        da_h_y12.BindDevicePointer(&d_h_y12);
        da_h_z12.resize(n_coef);
        da_h_z12.BindDevicePointer(&d_h_z12);
        da_h_x12_jac.resize(n_coef);
        da_h_x12_jac.BindDevicePointer(&d_h_x12_jac);
        da_h_y12_jac.resize(n_coef);
        da_h_y12_jac.BindDevicePointer(&d_h_y12_jac);
        da_h_z12_jac.resize(n_coef);
        da_h_z12_jac.BindDevicePointer(&d_h_z12_jac);
        da_element_connectivity.resize(n_elem * Quadrature::N_NODE_T4_4);
        da_element_connectivity.BindDevicePointer(&d_element_connectivity);
        da_tet1pt_x.resize(Quadrature::N_QP_T4_1);
        da_tet1pt_x.BindDevicePointer(&d_tet1pt_x);
        da_tet1pt_y.resize(Quadrature::N_QP_T4_1);
        da_tet1pt_y.BindDevicePointer(&d_tet1pt_y);
        da_tet1pt_z.resize(Quadrature::N_QP_T4_1);
        da_tet1pt_z.BindDevicePointer(&d_tet1pt_z);
        da_tet1pt_weights.resize(Quadrature::N_QP_T4_1);
        da_tet1pt_weights.BindDevicePointer(&d_tet1pt_weights);
        da_grad_N_ref.resize(n_elem * Quadrature::N_QP_T4_1 * 4 * 3);
        da_grad_N_ref.BindDevicePointer(&d_grad_N_ref);
        da_detJ_ref.resize(n_elem * Quadrature::N_QP_T4_1);
        da_detJ_ref.BindDevicePointer(&d_detJ_ref);
        da_F.resize(n_elem * Quadrature::N_QP_T4_1 * 3 * 3);
        da_F.BindDevicePointer(&d_F);
        da_P.resize(n_elem * Quadrature::N_QP_T4_1 * 3 * 3);
        da_P.BindDevicePointer(&d_P);
        da_Fdot.resize(n_elem * Quadrature::N_QP_T4_1 * 3 * 3);
        da_Fdot.BindDevicePointer(&d_Fdot);
        da_P_vis.resize(n_elem * Quadrature::N_QP_T4_1 * 3 * 3);
        da_P_vis.BindDevicePointer(&d_P_vis);
        da_f_int.resize(n_coef * 3);
        da_f_int.BindDevicePointer(&d_f_int);
        da_f_ext.resize(n_coef * 3);
        da_f_ext.BindDevicePointer(&d_f_ext);

        MOPHI_GPU_CALL(cudaMalloc(&d_rho0, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_nu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_E, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_material_model, sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu10, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_mu01, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_kappa, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_eta_damp, sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc(&d_lambda_damp, sizeof(Real)));

        MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_FEAT4_Data)));
    }

    void Setup(const VectorXR& tet1pt_x_host,
               const VectorXR& tet1pt_y_host,
               const VectorXR& tet1pt_z_host,
               const VectorXR& tet1pt_weights_host,
               const VectorXR& h_x12,
               const VectorXR& h_y12,
               const VectorXR& h_z12,
               const MatrixXi& element_connectivity) {
        if (is_setup) {
            MOPHI_ERROR(std::string("GPU_FEAT4_Data is already set up."));
            return;
        }

        std::copy(h_x12.data(), h_x12.data() + n_coef, da_h_x12.host());
        da_h_x12.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_h_y12.host());
        da_h_y12.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_h_z12.host());
        da_h_z12.ToDevice();
        std::copy(h_x12.data(), h_x12.data() + n_coef, da_h_x12_jac.host());
        da_h_x12_jac.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_h_y12_jac.host());
        da_h_y12_jac.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_h_z12_jac.host());
        da_h_z12_jac.ToDevice();

        std::copy(element_connectivity.data(), element_connectivity.data() + n_elem * Quadrature::N_NODE_T4_4,
                  da_element_connectivity.host());
        da_element_connectivity.ToDevice();

        std::copy(tet1pt_x_host.data(), tet1pt_x_host.data() + Quadrature::N_QP_T4_1, da_tet1pt_x.host());
        da_tet1pt_x.ToDevice();
        std::copy(tet1pt_y_host.data(), tet1pt_y_host.data() + Quadrature::N_QP_T4_1, da_tet1pt_y.host());
        da_tet1pt_y.ToDevice();
        std::copy(tet1pt_z_host.data(), tet1pt_z_host.data() + Quadrature::N_QP_T4_1, da_tet1pt_z.host());
        da_tet1pt_z.ToDevice();
        std::copy(tet1pt_weights_host.data(), tet1pt_weights_host.data() + Quadrature::N_QP_T4_1,
                  da_tet1pt_weights.host());
        da_tet1pt_weights.ToDevice();
        da_grad_N_ref.SetVal(Real(0));
        da_grad_N_ref.MakeReadyDevice();
        da_detJ_ref.SetVal(Real(0));
        da_detJ_ref.MakeReadyDevice();

        da_f_int.SetVal(Real(0));
        da_f_int.MakeReadyDevice();

        da_F.SetVal(Real(0));
        da_F.MakeReadyDevice();
        da_P.SetVal(Real(0));
        da_P.MakeReadyDevice();
        da_Fdot.SetVal(Real(0));
        da_Fdot.MakeReadyDevice();
        da_P_vis.SetVal(Real(0));
        da_P_vis.MakeReadyDevice();

        Real rho0 = 0.0;
        Real nu = 0.0;
        Real E = 0.0;
        Real mu = E / (2 * (1 + nu));
        Real lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
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
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));

        MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_FEAT4_Data), cudaMemcpyHostToDevice));

        is_setup = true;
    }

    void SetDensity(Real rho0) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT4_Data must be set up before setting density.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetDamping(Real eta_damp, Real lambda_damp) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT4_Data must be set up before setting damping.");
            return;
        }
        MOPHI_GPU_CALL(cudaMemcpy(d_eta_damp, &eta_damp, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_lambda_damp, &lambda_damp, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetSVK() {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT4_Data must be set up before setting material.");
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
            MOPHI_ERROR("GPU_FEAT4_Data must be set up before setting material.");
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

    void SetMooneyRivlin(Real mu10, Real mu01, Real kappa) {
        if (!is_setup) {
            MOPHI_ERROR("GPU_FEAT4_Data must be set up before setting material.");
            return;
        }

        int material_model = MATERIAL_MODEL_MOONEY_RIVLIN;
        MOPHI_GPU_CALL(cudaMemcpy(d_material_model, &material_model, sizeof(int), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu10, &mu10, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_mu01, &mu01, sizeof(Real), cudaMemcpyHostToDevice));
        MOPHI_GPU_CALL(cudaMemcpy(d_kappa, &kappa, sizeof(Real), cudaMemcpyHostToDevice));
    }

    void SetExternalForce(const VectorXR& h_f_ext) {
        if (h_f_ext.size() != n_coef * 3) {
            MOPHI_ERROR("External force vector size mismatch.");
            return;
        }

        std::copy(h_f_ext.data(), h_f_ext.data() + n_coef * 3, da_f_ext.host());
        da_f_ext.ToDevice();
    }

    const Real* GetX12DevicePtr() const {
        return d_h_x12;
    }

    const Real* GetY12DevicePtr() const {
        return d_h_y12;
    }

    const Real* GetZ12DevicePtr() const {
        return d_h_z12;
    }

    Real* GetExternalForceDevicePtr() {
        return d_f_ext;
    }

    const Real* GetExternalForceDevicePtr() const {
        return d_f_ext;
    }

    void UpdatePositions(const VectorXR& h_x12, const VectorXR& h_y12, const VectorXR& h_z12) {
        if (h_x12.size() != n_coef || h_y12.size() != n_coef || h_z12.size() != n_coef) {
            MOPHI_ERROR("Position vector size mismatch.");
            return;
        }
        std::copy(h_x12.data(), h_x12.data() + n_coef, da_h_x12.host());
        da_h_x12.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_h_y12.host());
        da_h_y12.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_h_z12.host());
        da_h_z12.ToDevice();
    }

    void UpdateConstraintTargets(const VectorXR& h_x12, const VectorXR& h_y12, const VectorXR& h_z12) {
        if (h_x12.size() != n_coef || h_y12.size() != n_coef || h_z12.size() != n_coef) {
            MOPHI_ERROR("Position vector size mismatch.");
            return;
        }
        std::copy(h_x12.data(), h_x12.data() + n_coef, da_h_x12_jac.host());
        da_h_x12_jac.ToDevice();
        std::copy(h_y12.data(), h_y12.data() + n_coef, da_h_y12_jac.host());
        da_h_y12_jac.ToDevice();
        std::copy(h_z12.data(), h_z12.data() + n_coef, da_h_z12_jac.host());
        da_h_z12_jac.ToDevice();
    }

    void SetNodalFixed(const VectorXi& fixed_nodes);

    void UpdateNodalFixed(const VectorXi& fixed_nodes);

    void Destroy() {
        da_h_x12.free();
        da_h_y12.free();
        da_h_z12.free();
        da_h_x12_jac.free();
        da_h_y12_jac.free();
        da_h_z12_jac.free();
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
        }

        if (is_j_csr_setup) {
            MOPHI_GPU_CALL(cudaFree(d_j_csr_offsets));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_columns));
            MOPHI_GPU_CALL(cudaFree(d_j_csr_values));
            MOPHI_GPU_CALL(cudaFree(d_j_nnz));
        }

        da_tet1pt_x.free();
        da_tet1pt_y.free();
        da_tet1pt_z.free();
        da_tet1pt_weights.free();

        da_grad_N_ref.free();
        da_detJ_ref.free();

        da_F.free();
        da_P.free();
        da_Fdot.free();
        da_P_vis.free();
        da_f_int.free();
        da_f_ext.free();

        MOPHI_GPU_CALL(cudaFree(d_rho0));
        MOPHI_GPU_CALL(cudaFree(d_nu));
        MOPHI_GPU_CALL(cudaFree(d_E));
        MOPHI_GPU_CALL(cudaFree(d_lambda));
        MOPHI_GPU_CALL(cudaFree(d_mu));
        MOPHI_GPU_CALL(cudaFree(d_material_model));
        MOPHI_GPU_CALL(cudaFree(d_mu10));
        MOPHI_GPU_CALL(cudaFree(d_mu01));
        MOPHI_GPU_CALL(cudaFree(d_kappa));
        MOPHI_GPU_CALL(cudaFree(d_eta_damp));
        MOPHI_GPU_CALL(cudaFree(d_lambda_damp));

        MOPHI_GPU_CALL(cudaFree(d_data));

        if (is_constraints_setup) {
            da_constraint.free();
            da_fixed_nodes.free();
        }
    }

    Real* Get_Constraint_Ptr() {
        return d_constraint;
    }

    bool Get_Is_Constraint_Setup() {
        return is_constraints_setup;
    }

    // ── ElementBase dispatch helpers ─────────────────────────────────────────

    ElementBase* GetDevicePtr() override {
        return d_data;
    }

    bool IsConstraintSetup() override {
        return is_constraints_setup;
    }

    Real* GetConstraintDevicePtr() override {
        return d_constraint;
    }

    GPU_FEAT4_Data* d_data;  // GPU copy of this struct

    int n_elem;
    int n_coef;
    int n_constraint;

  private:
    mophi::DualArray<Real> da_h_x12, da_h_y12, da_h_z12;
    mophi::DualArray<Real> da_h_x12_jac, da_h_y12_jac, da_h_z12_jac;
    mophi::DualArray<int> da_element_connectivity;
    mophi::DualArray<Real> da_tet1pt_x, da_tet1pt_y, da_tet1pt_z, da_tet1pt_weights;
    mophi::DualArray<Real> da_grad_N_ref;
    mophi::DualArray<Real> da_detJ_ref;
    mophi::DualArray<Real> da_F, da_P, da_Fdot, da_P_vis;
    mophi::DualArray<Real> da_f_int, da_f_ext;
    mophi::DualArray<Real> da_constraint;
    mophi::DualArray<int> da_fixed_nodes;

    Real *d_h_x12, *d_h_y12, *d_h_z12;
    Real *d_h_x12_jac, *d_h_y12_jac, *d_h_z12_jac;

    int* d_element_connectivity;  // (n_elem, 4)

    // Mass Matrix in CSR format
    int *d_csr_offsets, *d_csr_columns;
    Real* d_csr_values;
    int* d_nnz;

    Real *d_tet1pt_x, *d_tet1pt_y, *d_tet1pt_z;
    Real* d_tet1pt_weights;

    Real* d_grad_N_ref;  // (n_elem, 1, 4, 3)
    Real* d_detJ_ref;    // (n_elem, 1)

    Real* d_F;      // (n_elem, n_qp, 3, 3)
    Real* d_P;      // (n_elem, n_qp, 3, 3)
    Real* d_Fdot;   // (n_elem, n_qp, 3, 3)
    Real* d_P_vis;  // (n_elem, n_qp, 3, 3)

    Real *d_E, *d_nu, *d_rho0, *d_lambda, *d_mu;
    int* d_material_model;
    Real *d_mu10, *d_mu01, *d_kappa;
    Real *d_eta_damp, *d_lambda_damp;

    Real* d_constraint;
    int* d_fixed_nodes;
    int *d_cj_csr_offsets, *d_cj_csr_columns;
    Real* d_cj_csr_values;
    int* d_cj_nnz;

    int *d_j_csr_offsets, *d_j_csr_columns;
    Real* d_j_csr_values;
    int* d_j_nnz;

    Real *d_f_int, *d_f_ext;

    bool is_setup = false;
    bool is_constraints_setup = false;
    bool is_csr_setup = false;
    bool is_cj_csr_setup = false;
    bool is_j_csr_setup = false;
};

}  // namespace tlfea
