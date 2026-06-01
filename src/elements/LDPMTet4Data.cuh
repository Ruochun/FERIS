#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include "../materials/LDPM.cuh"
#include "../types.h"
#include "../utils/cuda_utils.h"
#include "../utils/ldpm_mesh_utils.h"
#include "ElementBase.h"
#include <MoPhiEssentials.h>

// GPU data structure for the 6-DOF (translational + rotational) Lattice
// Discrete Particle Model (LDPM) built on a TET4 Delaunay triangulation.
//
// Interaction model summary
// ─────────────────────────
//  • Every unique edge (i, j) of the TET4 mesh is one interaction "strut".
//  • The Voronoi facet area A_ij is precomputed from the local tetrahedral
//    sub-facets in Setup().
//  • Each step, the relative displacement between particles i and j is
//    projected onto the reference facet frame (n, m, l) to obtain strains
//    e_N, e_M, e_L; likewise the difference in rotation vectors gives
//    kappa_T, kappa_M, kappa_L (linearised rotations).
//  • The linear-elastic constitutive law maps all 6 strains to 3 tractions
//    and 3 moments, which are assembled onto the two endpoint nodes.
//
// DOF layout
// ──────────
//  • Translational positions: d_x_cur, d_y_cur, d_z_cur  [n_nodes each]
//  • Rotational positions:    d_rot_x_cur, d_rot_y_cur, d_rot_z_cur [n_nodes each]
//  • Translational forces:    d_f_int_t, d_f_ext_t  [n_nodes*3 each]
//  • Rotational moments:      d_f_int_r, d_f_ext_r  [n_nodes*3 each]
//
// Compatibility with LeapfrogSolver
// ──────────────────────────────────
//  • TYPE_LDPM_TET4 uses a 6*n_nodes velocity array in the solver split as:
//      [v_trans (3*n_nodes) | v_rot (3*n_nodes)]
//  • Lumped mass array is 2*n_nodes:
//      [m_trans (n_nodes)   | I_rot  (n_nodes)]
//  • n_total_qp_ = 1 and n_shape_ = 2 (one facet per edge, two endpoint nodes).
//  • Rotational inertia: I_lump = ALPHA_ROT * m_lump * l_min_node^2
//    with ALPHA_ROT = 0.25.

namespace feris {

// Rotational inertia scaling factor  (I = alpha * m * l_min^2).
static constexpr Real LDPM_TET4_ALPHA_ROT = Real(0.25);

enum LDPMFacetComponent {
    LDPM_FACET_T_N = 0,
    LDPM_FACET_T_M = 1,
    LDPM_FACET_T_L = 2,
    LDPM_FACET_M_T = 3,
    LDPM_FACET_M_M = 4,
    LDPM_FACET_M_L = 5,
    LDPM_FACET_N_COMPONENTS = 6
};

struct GPU_LDPMTet4_Data : public ElementBase {
#if defined(__CUDACC__)

    // ── Translational nodal positions ────────────────────────────────────────

    __device__ Map<VectorXR> x_cur() {
        return Map<VectorXR>(d_x_cur, n_nodes);
    }
    __device__ const Map<VectorXR> x_cur() const {
        return Map<VectorXR>(d_x_cur, n_nodes);
    }
    __device__ Map<VectorXR> y_cur() {
        return Map<VectorXR>(d_y_cur, n_nodes);
    }
    __device__ const Map<VectorXR> y_cur() const {
        return Map<VectorXR>(d_y_cur, n_nodes);
    }
    __device__ Map<VectorXR> z_cur() {
        return Map<VectorXR>(d_z_cur, n_nodes);
    }
    __device__ const Map<VectorXR> z_cur() const {
        return Map<VectorXR>(d_z_cur, n_nodes);
    }

    // ── Rotational nodal positions (θ_x, θ_y, θ_z per node) ─────────────────

    __device__ Map<VectorXR> rot_x_cur() {
        return Map<VectorXR>(d_rot_x_cur, n_nodes);
    }
    __device__ const Map<VectorXR> rot_x_cur() const {
        return Map<VectorXR>(d_rot_x_cur, n_nodes);
    }
    __device__ Map<VectorXR> rot_y_cur() {
        return Map<VectorXR>(d_rot_y_cur, n_nodes);
    }
    __device__ const Map<VectorXR> rot_y_cur() const {
        return Map<VectorXR>(d_rot_y_cur, n_nodes);
    }
    __device__ Map<VectorXR> rot_z_cur() {
        return Map<VectorXR>(d_rot_z_cur, n_nodes);
    }
    __device__ const Map<VectorXR> rot_z_cur() const {
        return Map<VectorXR>(d_rot_z_cur, n_nodes);
    }

    // ── Translational nodal forces ───────────────────────────────────────────

    __device__ Map<VectorXR> f_int() {
        return Map<VectorXR>(d_f_int_t, n_nodes * 3);
    }
    __device__ const Map<VectorXR> f_int() const {
        return Map<VectorXR>(d_f_int_t, n_nodes * 3);
    }
    __device__ Map<VectorXR> f_ext() {
        return Map<VectorXR>(d_f_ext_t, n_nodes * 3);
    }
    __device__ const Map<VectorXR> f_ext() const {
        return Map<VectorXR>(d_f_ext_t, n_nodes * 3);
    }

    // ── Rotational nodal moments ─────────────────────────────────────────────

    __device__ Map<VectorXR> f_int_r() {
        return Map<VectorXR>(d_f_int_r, n_nodes * 3);
    }
    __device__ const Map<VectorXR> f_int_r() const {
        return Map<VectorXR>(d_f_int_r, n_nodes * 3);
    }
    __device__ Map<VectorXR> f_ext_r() {
        return Map<VectorXR>(d_f_ext_r, n_nodes * 3);
    }
    __device__ const Map<VectorXR> f_ext_r() const {
        return Map<VectorXR>(d_f_ext_r, n_nodes * 3);
    }

    // ── Boundary conditions ──────────────────────────────────────────────────

    __device__ Map<VectorXi> fixed_nodes() {
        return Map<VectorXi>(d_fixed_nodes, n_constraint / 6);
    }
    __device__ int gpu_n_constraint() const {
        return n_constraint;
    }
    __device__ int gpu_n_coef() const {
        return n_nodes;
    }
    __device__ int gpu_n_edge() const {
        return n_edge;
    }

    // ── Edge geometry ────────────────────────────────────────────────────────

    __device__ int edge_node(int edge_idx, int local_node) const {
        return d_edge_nodes[edge_idx * 2 + local_node];
    }
    __device__ Real edge_l0(int edge_idx) const {
        return d_l0[edge_idx];
    }
    __device__ Real facet_area(int edge_idx) const {
        return d_facet_area[edge_idx];
    }
    __device__ Real edge_n_comp(int edge_idx, int k) const {
        return d_edge_n[edge_idx * 3 + k];
    }
    __device__ Real edge_m_comp(int edge_idx, int k) const {
        return d_edge_m[edge_idx * 3 + k];
    }
    __device__ Real edge_lv_comp(int edge_idx, int k) const {
        return d_edge_lv[edge_idx * 3 + k];
    }

    // ── Per-step facet tractions and moments ─────────────────────────────────
    // Layout per edge uses LDPMFacetComponent:
    // [t_N, t_M, t_L, m_T, m_M, m_L].

    __device__ Real& facet_t(int edge_idx, int comp) {
        return d_facet_t[edge_idx * LDPM_FACET_N_COMPONENTS + comp];
    }
    __device__ Real facet_t(int edge_idx, int comp) const {
        return d_facet_t[edge_idx * LDPM_FACET_N_COMPONENTS + comp];
    }

    // ── Per-edge damage state (legacy) ──────────────────────────────────────────
    // kappa : max effective damage-driving strain (history variable, grows only)
    // omega : current damage variable ∈ [0, 1]

    __device__ Real& edge_kappa(int edge_idx) {
        return d_kappa[edge_idx];
    }
    __device__ Real edge_kappa(int edge_idx) const {
        return d_kappa[edge_idx];
    }
    __device__ Real& edge_omega(int edge_idx) {
        return d_omega[edge_idx];
    }
    __device__ Real edge_omega(int edge_idx) const {
        return d_omega[edge_idx];
    }

    // ── Per-edge full LDPM state vector (LDPM_N_STATEV per edge) ─────────────

    __device__ Real& edge_statev(int edge_idx, int comp) {
        return d_statev[edge_idx * LDPM_N_STATEV + comp];
    }
    __device__ Real edge_statev(int edge_idx, int comp) const {
        return d_statev[edge_idx * LDPM_N_STATEV + comp];
    }

    // ── Per-edge precomputed volumetric strain ───────────────────────────────

    __device__ Real edge_vol_strain(int edge_idx) const {
        return d_edge_vol_strain[edge_idx];
    }

    // ── Full LDPM parameters (device struct) ─────────────────────────────────

    __device__ const LDPMParams& ldpm_params() const {
        return *d_ldpm_params;
    }

    // ── Material parameters ──────────────────────────────────────────────────

    __device__ Real E_N() const {
        return *d_E_N;
    }
    __device__ Real E_T() const {
        return *d_E_T;
    }
    __device__ Real E_kT() const {
        return *d_E_kT;
    }
    __device__ Real E_kM() const {
        return *d_E_kM;
    }
    __device__ Real E_kL() const {
        return *d_E_kL;
    }

    // ── Per-node rotational inertia ──────────────────────────────────────────

    __device__ Real I_lump(int node_i) const {
        return d_I_lump[node_i];
    }

    // ── Diagonal CSR mass matrix ─────────────────────────────────────────────

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
        return n_nodes;
    }
    __host__ __device__ int get_n_beam() const override {
        return n_edge;
    }

    // ── Virtual method implementations ───────────────────────────────────────

    void CalcMassMatrix() override;
    void CalcInternalForce() override;
    void CalcConstraintData() override {}
    void CalcP() override;

    // Runs only the volumetric strain pre-computation steps (Steps A+B of CalcP):
    //   A. Per-tet volumetric strain from current node positions.
    //   B. Average per-tet strains onto per-edge values via CSR mapping.
    // Called by the LeapfrogSolver before each facet constitutive update so
    // that eps_V is up-to-date when compute_ldpm_facet_strain_and_stress runs.
    void ComputeVolumetricStrain();

    void RetrieveInternalForceToCPU(VectorXR& internal_force) override;
    void RetrieveInternalForceToCPU(VectorReal3& internal_force) {
        VectorXR flat;
        RetrieveInternalForceToCPU(flat);
        if (!UnflattenVectorReal3(flat, internal_force)) {
            MOPHI_ERROR(
                "GPU_LDPMTet4_Data::RetrieveInternalForceToCPU: internal force size (%d) is not divisible by 3.",
                static_cast<int>(flat.size()));
            return;
        }
    }
    void RetrieveInternalMomentToCPU(VectorXR& internal_moment);
    void RetrieveInternalMomentToCPU(VectorReal3& internal_moment) {
        VectorXR flat;
        RetrieveInternalMomentToCPU(flat);
        if (!UnflattenVectorReal3(flat, internal_moment)) {
            MOPHI_ERROR(
                "GPU_LDPMTet4_Data::RetrieveInternalMomentToCPU: internal moment size (%d) is not divisible by 3.",
                static_cast<int>(flat.size()));
            return;
        }
    }
    void RetrieveConstraintDataToCPU(VectorXR&) override {}
    void RetrieveConstraintJacobianToCPU(MatrixXR&) override {}
    void RetrievePositionToCPU(VectorXR& x_cur, VectorXR& y_cur, VectorXR& z_cur) override;
    void RetrieveRotationToCPU(VectorXR& rot_x_cur, VectorXR& rot_y_cur, VectorXR& rot_z_cur);
    void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>&) override {}
    void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>&) override {}

    // ── ElementBase dispatch helpers ─────────────────────────────────────────

    ElementBase* GetDevicePtr() override {
        return d_data;
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    // Default constructor: mesh counts are set later by SetupFromMesh() or Setup().
    GPU_LDPMTet4_Data() : n_nodes(0), n_elem(0), n_edge(0), n_constraint(0), h_rho(0) {
        type = TYPE_LDPM_TET4;
    }

    // Parameterized constructor: use when calling Setup() directly.
    GPU_LDPMTet4_Data(int num_nodes, int num_elems)
        : n_nodes(num_nodes), n_elem(num_elems), n_edge(0), n_constraint(0), h_rho(0) {
        type = TYPE_LDPM_TET4;
    }

    // ── Setup / configuration ─────────────────────────────────────────────────

    void Setup(const VectorXR& h_x, const VectorXR& h_y, const VectorXR& h_z, const MatrixXi& tet_connectivity);

    // Convenience overload: initialize from a pre-parsed LDPMTet4Mesh.
    // Calls Setup() with the particle positions and TET connectivity from the
    // mesh, then stores all additional file-loaded data fields
    // (particle diameters, sub-facet geometry, face facets, facet vertices).
    // The mesh must have been populated by ReadLDPMTet4MeshFromFiles() or the
    // individual file readers.
    void SetupFromMesh(const LDPMTet4Mesh& mesh);

    // Set LDPM-TET4 material moduli.  Must be called after Setup().
    // E_N  – normal translational modulus (Pa)
    // E_T  – shear translational modulus (Pa)
    // E_kT – twist modulus (Pa·m²)
    // E_kM – bending modulus, m-direction (Pa·m²)
    // E_kL – bending modulus, l-direction (Pa·m²)
    void SetMaterial(Real E_N_val, Real E_T_val, Real E_kT_val, Real E_kM_val, Real E_kL_val);

    // Set the full LDPM parameter set.  Must be called after Setup().
    // Configures the full LDPM model with tensile fracture, compression, and friction.
    void SetLDPMParams(const LDPMParams& params);

    void SetDensity(Real rho_val);
    void SetExternalForce(const VectorXR& h_f_ext);
    void SetExternalForce(const VectorReal3& h_f_ext) {
        if (static_cast<int>(h_f_ext.size()) != n_nodes) {
            MOPHI_ERROR("GPU_LDPMTet4_Data::SetExternalForce: force vector size mismatch.");
            return;
        }
        VectorXR flat = FlattenVectorReal3(h_f_ext);
        SetExternalForce(flat);
    }
    void SetNodalFixed(const VectorXi& fixed_nodes_in);

    void RetrieveExternalForceToCPU(VectorXR& f_ext_out);
    void RetrieveExternalForceToCPU(VectorReal3& f_ext_out) {
        VectorXR flat;
        RetrieveExternalForceToCPU(flat);
        if (!UnflattenVectorReal3(flat, f_ext_out)) {
            MOPHI_ERROR(
                "GPU_LDPMTet4_Data::RetrieveExternalForceToCPU: external force size (%d) is not divisible by 3.",
                static_cast<int>(flat.size()));
            return;
        }
    }

    // Retrieve per-edge damage state from GPU to host.
    // omega_out : size n_edge, omega ∈ [0, 1] (0 = undamaged, 1 = fully failed)
    void RetrieveFacetDamageToCPU(VectorXR& omega_out);

    // Retrieve per-edge facet traction/moment components from GPU to host.
    // out size: LDPM_FACET_N_COMPONENTS * n_edge, layout per edge:
    // [t_N, t_M, t_L, m_T, m_M, m_L].
    void RetrieveFacetTractionToCPU(VectorXR& out);

    // Retrieve per-edge max effective strain history (κ) from GPU to host.
    // kappa_out : size n_edge, κ ≥ 0 (max damage-driving strain ever reached)
    void RetrieveFacetKappaToCPU(VectorXR& kappa_out);

    // Expose edge connectivity for visualization (e.g. VTK line-segment mesh).
    // Returns a reference to the host-side edge node array (size 2 * n_edge):
    //   h_edge_nodes_vec[2*e+0] = node i,  h_edge_nodes_vec[2*e+1] = node j.
    const std::vector<int>& GetEdgeNodes() const {
        return h_edge_nodes_vec;
    }

    // Project per-edge damage ω onto the sub-facet mesh.
    //
    // Each sub-facet belongs to a unique LDPM edge (built during SetupFromMesh).
    // This GPU kernel looks up omega[edge_idx] for each sub-facet and writes the
    // result into a flat output array of size n_subfacet.
    //
    // out  — resized to n_subfacet on return.  Values ∈ [0, 1]:
    //          0 = undamaged / elastic,  1 = fully fractured.
    //        Sub-facets whose owning edge could not be matched during setup
    //        receive a value of 0.
    // Must be called after SetupFromMesh() and at least one CalcP() step.
    void ProjectEdgeDamageToSubfacets(VectorXR& out);

    // Project per-edge crack opening displacement onto the sub-facet mesh.
    //
    // The crack opening for each edge is computed as ω × κ × l₀ [mm], where:
    //   ω  — damage variable (0 = elastic, 1 = fully fractured)
    //   κ  — max damage-driving strain history (irreversible, ≥ 0)
    //   l₀ — reference edge length [mm]
    //
    // This gives the inelastic (crack) displacement per facet in physical units.
    // For a fully cracked edge (ω ≈ 1) in the localization band all the boundary
    // displacement concentrates here, so values reach ~0.1 mm for the dogbone
    // test driven at 1 mm/s for 0.1 s.
    //
    // out — resized to n_subfacet on return; units match the mesh length unit (mm).
    // Must be called after SetupFromMesh() and at least one CalcP() step.
    void ProjectEdgeCrackDistanceToSubfacets(VectorXR& out);

    // Advance a set of particles' z-coordinate by a prescribed increment.
    //
    // Implements velocity-driven (kinematic) boundary conditions for the top
    // plate of a tensile test: the leapfrog solver has already zeroed the
    // velocity of the driven nodes (they are listed in SetNodalFixed so that
    // the solver applies zero-velocity at each step); this method then adds
    // dz = v_prescribed * dt to their z-position each step, producing the
    // prescribed displacement while keeping velocity nominally zero.
    //
    // Note: driven nodes differ from fixed (clamped) nodes — fixed nodes have
    // zero velocity AND zero displacement increment; driven nodes have zero
    // velocity but receive a prescribed nonzero displacement increment here.
    //
    // d_driven_idx — device array of driven node indices (length n_driven)
    // n_driven     — number of driven nodes
    // dz           — prescribed z-displacement increment this step [same units as positions]
    //
    // Must be called after solver.Solve() each time step.
    // Must be called after SetupFromMesh().
    void AdvanceDrivenNodesZ(const int* d_driven_idx, int n_driven, Real dz);

    const Real* GetXCurDevicePtr() const {
        return d_x_cur;
    }
    const Real* GetYCurDevicePtr() const {
        return d_y_cur;
    }
    const Real* GetZCurDevicePtr() const {
        return d_z_cur;
    }
    // Non-const accessors for externally driven (kinematic BC) position updates.
    Real* GetXCurWritableDevicePtr() {
        return d_x_cur;
    }
    Real* GetYCurWritableDevicePtr() {
        return d_y_cur;
    }
    Real* GetZCurWritableDevicePtr() {
        return d_z_cur;
    }
    Real* GetExternalForceDevicePtr() {
        return d_f_ext_t;
    }

    void Destroy();

    // ── Public data members ──────────────────────────────────────────────────

    GPU_LDPMTet4_Data* d_data;

    int n_nodes;       // Number of nodes (particles)
    int n_elem;        // Number of TET4 elements
    int n_edge;        // Number of unique edges (set by Setup)
    int n_constraint;  // 6 * number of fixed nodes (LDPMTet4 has 6 DOFs per node)

    // ── File-loaded mesh data (host-side; filled by SetupFromMesh) ────────────
    // These fields mirror the six LDPM Chrono Workbench output files.
    // They are stored for reference and future use; not all are required by
    // the current linear-elastic simulation kernel.

    // From particles.dat — aggregate diameter per particle (0 = boundary node).
    // Size: n_nodes.
    VectorXR h_particle_d;

    // From facets.dat — precomputed Voronoi sub-facet geometry.
    // n_subfacet == 12 * n_elem (12 sub-facets per TET).
    int n_subfacet = 0;
    VectorXi h_subfacet_tet;         // owning TET index, size n_subfacet
    VectorXi h_subfacet_vertex_ids;  // 3 vertex indices (flat, size 3*n_subfacet)
    VectorXR h_subfacet_vol;         // sub-cell volume, size n_subfacet
    VectorXR h_subfacet_parea;       // projected (triangle) area, size n_subfacet
    VectorXR h_subfacet_centroid;    // centroid (cx,cy,cz) flat, size 3*n_subfacet
    VectorXR h_subfacet_normal;      // unit normal p (px,py,pz) flat, size 3*n_subfacet
    VectorXR h_subfacet_tangent_q;   // first tangent q (qx,qy,qz) flat, size 3*n_subfacet
    VectorXR h_subfacet_tangent_s;   // second tangent s (sx,sy,sz) flat, size 3*n_subfacet
    VectorXi h_subfacet_matflag;     // material zone flag, size n_subfacet

    // From faceFacets.dat — surface boundary triangles for traction BCs.
    int n_face_facet = 0;
    VectorXi h_face_facet_node;      // particle node index, size n_face_facet
    VectorXR h_face_facet_vertices;  // 9 coords per triangle (flat, size 9*n_face_facet)

    // From facetsVertices.dat — exact triangle corner positions.
    // n_facet_vertex == 3 * n_subfacet.
    int n_facet_vertex = 0;
    VectorXR h_facet_vertex_x, h_facet_vertex_y, h_facet_vertex_z;  // size n_facet_vertex each

  private:
    // ── Translational position arrays ────────────────────────────────────────
    mophi::DualArray<Real> da_x_cur, da_y_cur, da_z_cur;
    Real *d_x_cur, *d_y_cur, *d_z_cur;

    // ── Rotational position arrays ───────────────────────────────────────────
    mophi::DualArray<Real> da_rot_x_cur, da_rot_y_cur, da_rot_z_cur;
    Real *d_rot_x_cur, *d_rot_y_cur, *d_rot_z_cur;

    // ── Translational force arrays ───────────────────────────────────────────
    mophi::DualArray<Real> da_f_int_t, da_f_ext_t;
    Real *d_f_int_t, *d_f_ext_t;

    // ── Rotational moment arrays ─────────────────────────────────────────────
    mophi::DualArray<Real> da_f_int_r, da_f_ext_r;
    Real *d_f_int_r, *d_f_ext_r;

    // ── TET connectivity (host copy for CalcMassMatrix) ───────────────────────
    std::vector<int> h_tet_conn_vec;

    // ── Edge geometry ────────────────────────────────────────────────────────
    mophi::DualArray<int> da_edge_nodes;
    mophi::DualArray<Real> da_l0;
    mophi::DualArray<Real> da_facet_area;
    mophi::DualArray<Real> da_edge_n;
    mophi::DualArray<Real> da_edge_m;
    mophi::DualArray<Real> da_edge_lv;
    int* d_edge_nodes;
    Real* d_l0;
    Real* d_facet_area;
    Real* d_edge_n;
    Real* d_edge_m;
    Real* d_edge_lv;

    // Host copies of edge data for CalcMassMatrix (per-node l_min computation)
    std::vector<int> h_edge_nodes_vec;
    std::vector<Real> h_l0_vec;

    // ── Per-step facet tractions and moments ──────────────────────────────────
    mophi::DualArray<Real> da_facet_t;  // [n_edge * LDPM_FACET_N_COMPONENTS]
    Real* d_facet_t;

    // ── Per-edge damage state (legacy) ───────────────────────────────────────────
    mophi::DualArray<Real> da_kappa;  // history max effective strain [n_edge]
    mophi::DualArray<Real> da_omega;  // damage variable              [n_edge]
    Real* d_kappa;
    Real* d_omega;

    // ── Per-edge full LDPM state vector [n_edge * LDPM_N_STATEV] ─────────────
    mophi::DualArray<Real> da_statev;
    Real* d_statev = nullptr;

    // ── Volumetric strain averaging data ─────────────────────────────────────
    // Tet connectivity on device [n_elem * 4]
    mophi::DualArray<int> da_tet_conn;
    int* d_tet_conn = nullptr;

    // Reference tet volumes [n_elem]
    mophi::DualArray<Real> da_tet_vol_ref;
    Real* d_tet_vol_ref = nullptr;

    // Per-tet current volumetric strain [n_elem] (scratch, computed each step)
    mophi::DualArray<Real> da_tet_vol_strain;
    Real* d_tet_vol_strain = nullptr;

    // CSR edge-to-tet mapping: for each edge, which tets share it.
    // d_edge_tet_offsets[n_edge+1], d_edge_tet_indices[nnz]
    mophi::DualArray<int> da_edge_tet_offsets;
    mophi::DualArray<int> da_edge_tet_indices;
    int* d_edge_tet_offsets = nullptr;
    int* d_edge_tet_indices = nullptr;

    // Per-edge averaged volumetric strain [n_edge]
    mophi::DualArray<Real> da_edge_vol_strain;
    Real* d_edge_vol_strain = nullptr;

    // ── Full LDPM material parameters (device) ───────────────────────────────
    LDPMParams* d_ldpm_params = nullptr;

    // ── Subfacet → global edge index (built by SetupFromMesh) ─────────────────
    // da_subfacet_edge_idx[sf] = global edge index for sub-facet sf.
    // Entries are -1 for sub-facets that could not be matched (fallback: 0 damage).
    mophi::DualArray<int> da_subfacet_edge_idx;  // [n_subfacet]
    int* d_subfacet_edge_idx = nullptr;

    // ── Boundary conditions ───────────────────────────────────────────────────
    mophi::DualArray<int> da_fixed_nodes;
    int* d_fixed_nodes;

    // ── Diagonal CSR mass matrix (translational) ──────────────────────────────
    int* d_csr_offsets;
    int* d_csr_columns;
    Real* d_csr_values;
    int* d_nnz;

    // ── Per-node rotational inertia ───────────────────────────────────────────
    mophi::DualArray<Real> da_I_lump;
    Real* d_I_lump;

    // ── Material parameters (device scalars) ──────────────────────────────────
    Real* d_E_N;
    Real* d_E_T;
    Real* d_E_kT;
    Real* d_E_kM;
    Real* d_E_kL;
    Real* d_rho;
    Real h_rho;

    bool is_setup = false;
    bool is_csr_setup = false;
    bool is_constraints_setup = false;
};

}  // namespace feris
