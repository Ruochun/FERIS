#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include "../types.h"
#include "../utils/cuda_utils.h"
#include "ElementBase.h"
#include <MoPhiEssentials.h>

// GPU data structure for the 3-DOF translational Lattice Discrete Particle
// Model (LDPM) built on a TET4 Delaunay triangulation.
//
// Interaction model summary
// ─────────────────────────
//  • Every unique edge (i, j) of the TET4 mesh is one interaction "strut".
//  • The Voronoi facet area A_ij is precomputed from the local tetrahedral
//    sub-facets in Setup().
//  • Each step, the relative displacement between particles i and j is
//    projected onto the reference facet frame (n, m, l) to obtain strains
//    e_N, e_M, e_L, which drive the linear-elastic constitutive law.
//  • Facet forces are assembled back to the two endpoint nodes via atomicAdd.
//
// Compatibility with LeapfrogSolver
// ──────────────────────────────────
//  • 3 translational DOFs per node → fits the existing 3*n_coef_ velocity
//    layout in LeapfrogSolver without modification.
//  • Lumped mass is stored as a diagonal-only CSR so the existing
//    leapfrog_compute_lumped_mass_kernel works unchanged.
//  • get_n_beam() returns n_edge; get_n_coef() returns n_coef.
//  • n_total_qp_ = 1 and n_shape_ = 2 are set in the solver constructor.
#pragma once

namespace tlfea {

struct GPU_LDPM4_Data : public ElementBase {
#if defined(__CUDACC__)

    // ── Nodal positions (current configuration) ─────────────────────────────

    __device__ Map<VectorXR> x12() {
        return Map<VectorXR>(d_x12, n_coef);
    }
    __device__ const Map<VectorXR> x12() const {
        return Map<VectorXR>(d_x12, n_coef);
    }

    __device__ Map<VectorXR> y12() {
        return Map<VectorXR>(d_y12, n_coef);
    }
    __device__ const Map<VectorXR> y12() const {
        return Map<VectorXR>(d_y12, n_coef);
    }

    __device__ Map<VectorXR> z12() {
        return Map<VectorXR>(d_z12, n_coef);
    }
    __device__ const Map<VectorXR> z12() const {
        return Map<VectorXR>(d_z12, n_coef);
    }

    // ── Nodal forces ─────────────────────────────────────────────────────────

    __device__ Map<VectorXR> f_int() {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }
    __device__ const Map<VectorXR> f_int() const {
        return Map<VectorXR>(d_f_int, n_coef * 3);
    }
    __device__ Map<VectorXR> f_int(int global_node_idx) {
        return Map<VectorXR>(d_f_int + global_node_idx * 3, 3);
    }

    __device__ Map<VectorXR> f_ext() {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }
    __device__ const Map<VectorXR> f_ext() const {
        return Map<VectorXR>(d_f_ext, n_coef * 3);
    }
    __device__ Map<VectorXR> f_ext(int global_node_idx) {
        return Map<VectorXR>(d_f_ext + global_node_idx * 3, 3);
    }

    // ── Boundary conditions ──────────────────────────────────────────────────

    __device__ Map<VectorXi> fixed_nodes() {
        return Map<VectorXi>(d_fixed_nodes, n_constraint / 3);
    }
    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }
    __device__ int gpu_n_coef() const {
        return n_coef;
    }
    __device__ int gpu_n_edge() const {
        return n_edge;
    }

    // ── Edge geometry (precomputed, read-only during simulation) ─────────────

    // Returns node index (0=node_i, 1=node_j) for the given edge.
    __device__ int edge_node(int edge_idx, int local_node) const {
        return d_edge_nodes[edge_idx * 2 + local_node];
    }

    // Reference edge length (distance between the two endpoint particles).
    __device__ Real edge_l0(int edge_idx) const {
        return d_l0[edge_idx];
    }

    // Voronoi facet area assigned to this edge.
    __device__ Real facet_area(int edge_idx) const {
        return d_facet_area[edge_idx];
    }

    // Component k (0..2) of the unit normal vector along the reference edge.
    __device__ Real edge_n_comp(int edge_idx, int k) const {
        return d_edge_n[edge_idx * 3 + k];
    }

    // Component k of the first shear direction (orthogonal to n).
    __device__ Real edge_m_comp(int edge_idx, int k) const {
        return d_edge_m[edge_idx * 3 + k];
    }

    // Component k of the second shear direction (= n × m).
    __device__ Real edge_lv_comp(int edge_idx, int k) const {
        return d_edge_lv[edge_idx * 3 + k];
    }

    // ── Per-step facet tractions ─────────────────────────────────────────────

    // Read-write access: comp 0 = t_N, 1 = t_M, 2 = t_L.
    __device__ Real& facet_t(int edge_idx, int comp) {
        return d_facet_t[edge_idx * 3 + comp];
    }
    __device__ Real facet_t(int edge_idx, int comp) const {
        return d_facet_t[edge_idx * 3 + comp];
    }

    // ── Material parameters ──────────────────────────────────────────────────

    __device__ Real E_N() const {
        return *d_E_N;
    }
    __device__ Real E_T() const {
        return *d_E_T;
    }

    // ── Diagonal CSR mass matrix (for leapfrog_compute_lumped_mass_kernel) ───

    __device__ int* csr_offsets() {
        return d_csr_offsets;
    }
    __device__ int* csr_columns() {
        return d_csr_columns;
    }
    __device__ Real* csr_values() {
        return d_csr_values;
    }
    __device__ int nnz() {
        return *d_nnz;
    }

#endif  // __CUDACC__

    // ── Host/device scalar accessors ─────────────────────────────────────────

    __host__ __device__ int get_n_coef() const override {
        return n_coef;
    }
    // LeapfrogSolver reads get_n_beam() as the "element" count.
    // For LDPM, one edge = one element.
    __host__ __device__ int get_n_beam() const override {
        return n_edge;
    }

    // ── Virtual method implementations ───────────────────────────────────────

    // Builds the diagonal CSR mass matrix from per-node Voronoi volumes and
    // the density set via SetDensity().  Must be called after SetDensity().
    void CalcMassMatrix() override;

    // Clears f_int then assembles it from the current facet tractions.
    void CalcInternalForce() override;

    // Not needed for leapfrog (constraints enforced by velocity zeroing).
    void CalcConstraintData() override {}

    // Computes facet tractions at the current nodal positions (no velocities).
    void CalcP() override;

    void RetrieveInternalForceToCPU(VectorXR& internal_force) override;
    void RetrieveConstraintDataToCPU(VectorXR&) override {}
    void RetrieveConstraintJacobianToCPU(MatrixXR&) override {}
    void RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12) override;
    void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>&) override {}
    void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>&) override {}

    // ── Constructor ──────────────────────────────────────────────────────────

    GPU_LDPM4_Data(int num_nodes, int num_elems)
        : n_coef(num_nodes), n_elem(num_elems), n_edge(0), n_constraint(0), h_rho(0) {
        type = TYPE_LDPM4;
    }

    // ── Setup / configuration ─────────────────────────────────────────────────

    // Allocates GPU buffers, extracts unique edges from the TET4 connectivity,
    // computes reference geometry (l0, n, m, l_vec, facet area), and copies
    // everything to device.  Must be called first.
    void Setup(const VectorXR& h_x, const VectorXR& h_y, const VectorXR& h_z, const MatrixXi& tet_connectivity);

    // Set LDPM normal and shear moduli.  Must be called after Setup().
    void SetMaterial(Real E_N_val, Real E_T_val);

    // Set particle density (kg/m³).  Must be called after Setup() and before
    // CalcMassMatrix().
    void SetDensity(Real rho_val);

    // Set the per-DOF external force vector (length n_coef * 3).
    void SetExternalForce(const VectorXR& h_f_ext);

    // Mark nodes as fixed (velocity will be zeroed each step).
    void SetNodalFixed(const VectorXi& fixed_nodes_in);

    // Retrieve nodal positions from the GPU to host vectors.
    void RetrieveExternalForceToCPU(VectorXR& f_ext_out);

    // Get raw device pointers (for use by the LeapfrogSolver internals).
    const Real* GetX12DevicePtr() const {
        return d_x12;
    }
    const Real* GetY12DevicePtr() const {
        return d_y12;
    }
    const Real* GetZ12DevicePtr() const {
        return d_z12;
    }
    Real* GetExternalForceDevicePtr() {
        return d_f_ext;
    }

    // Free all GPU and host buffers.
    void Destroy();

    // ── Public data members (needed by LeapfrogSolver) ───────────────────────

    GPU_LDPM4_Data* d_data;  // Device-side mirror of this struct

    int n_coef;        // Number of particles (nodes)
    int n_elem;        // Number of TET4 elements
    int n_edge;        // Number of unique edges (set by Setup)
    int n_constraint;  // 3 * number of fixed nodes

  private:
    // ── Position arrays ──────────────────────────────────────────────────────
    mophi::DualArray<Real> da_x12, da_y12, da_z12;
    Real *d_x12, *d_y12, *d_z12;

    // ── Force arrays ─────────────────────────────────────────────────────────
    mophi::DualArray<Real> da_f_int, da_f_ext;
    Real *d_f_int, *d_f_ext;

    // ── TET4 connectivity (stored for CalcMassMatrix) ─────────────────────────
    std::vector<int> h_tet_conn_vec;  // host copy: n_elem × 4 (row-major)

    // ── Edge geometry (precomputed in Setup, read-only during time-stepping) ──
    mophi::DualArray<int> da_edge_nodes;   // [n_edge * 2]
    mophi::DualArray<Real> da_l0;          // [n_edge]
    mophi::DualArray<Real> da_facet_area;  // [n_edge]
    mophi::DualArray<Real> da_edge_n;      // [n_edge * 3]
    mophi::DualArray<Real> da_edge_m;      // [n_edge * 3]
    mophi::DualArray<Real> da_edge_lv;     // [n_edge * 3]
    int* d_edge_nodes;
    Real* d_l0;
    Real* d_facet_area;
    Real* d_edge_n;
    Real* d_edge_m;
    Real* d_edge_lv;

    // ── Per-step facet tractions ──────────────────────────────────────────────
    mophi::DualArray<Real> da_facet_t;  // [n_edge * 3]
    Real* d_facet_t;

    // ── Boundary conditions ───────────────────────────────────────────────────
    mophi::DualArray<int> da_fixed_nodes;
    int* d_fixed_nodes;

    // ── Diagonal CSR mass matrix ──────────────────────────────────────────────
    int* d_csr_offsets;
    int* d_csr_columns;
    Real* d_csr_values;
    int* d_nnz;

    // ── Material parameters (device scalars) ──────────────────────────────────
    Real* d_E_N;
    Real* d_E_T;
    Real* d_rho;
    Real h_rho;  // Host-side copy for CalcMassMatrix

    bool is_setup = false;
    bool is_csr_setup = false;
    bool is_constraints_setup = false;
};

}  // namespace tlfea
