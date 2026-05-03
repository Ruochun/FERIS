/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    LDPMTet4Data.cu
 * Brief:   Implements GPU-side data management for the 6-DOF LDPM
 *          element (TYPE_LDPM_TET4).  Host methods handle:
 *            - Edge extraction from TET4 connectivity
 *            - Reference geometry (l0, n, m, l_vec, facet area)
 *            - Diagonal CSR translational mass matrix
 *            - Per-node rotational inertia  (I = alpha * m * l_min^2)
 *            - Fixed-node BC setup
 *          GPU kernels handle:
 *            - CalcP  (wrapper: compute 6-DOF facet response)
 *            - CalcInternalForce  (clear + assemble f_int from tractions)
 *==============================================================
 *==============================================================*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../types.h"
#include "../utils/cuda_utils.h"
#include "LDPMTet4Data.cuh"
#include "LDPMTet4DataFunc.cuh"
#include <MoPhiEssentials.h>

namespace feris {

// ─── GPU kernel wrappers ─────────────────────────────────────────────────────

__global__ void calc_p_ldpm_tet4_kernel(GPU_LDPMTet4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_data->gpu_n_edge())
        return;
    compute_p(tid, 0, d_data, nullptr, Real(0));
}

// ─── GPU kernel: project per-edge damage ω onto per-subfacet damage ──────────
//
// For each sub-facet sf, looks up its owning global edge index via
// subfacet_edge_idx[sf] and writes omega[edge_idx] into subfacet_damage[sf].
// Sub-facets with edge_idx == -1 (unmatched) receive a value of 0.
__global__ void project_edge_damage_to_subfacets_kernel(int n_subfacet,
                                                        const int* __restrict__ subfacet_edge_idx,
                                                        const Real* __restrict__ edge_omega,
                                                        Real* __restrict__ subfacet_damage) {
    const int sf = blockIdx.x * blockDim.x + threadIdx.x;
    if (sf >= n_subfacet)
        return;
    const int eidx = subfacet_edge_idx[sf];
    subfacet_damage[sf] = (eidx >= 0) ? edge_omega[eidx] : Real(0);
}

// ─── GPU kernel: project per-edge crack opening displacement onto subfacets ──
//
// crack_dist[sf] = omega[e] * kappa[e] * l0[e]  (same units as l0, e.g. mm)
// Sub-facets with edge_idx == -1 (unmatched) receive a value of 0.
__global__ void project_edge_crack_dist_to_subfacets_kernel(int n_subfacet,
                                                            const int* __restrict__ subfacet_edge_idx,
                                                            const Real* __restrict__ edge_omega,
                                                            const Real* __restrict__ edge_kappa,
                                                            const Real* __restrict__ edge_l0,
                                                            Real* __restrict__ subfacet_crack_dist) {
    const int sf = blockIdx.x * blockDim.x + threadIdx.x;
    if (sf >= n_subfacet)
        return;
    const int eidx = subfacet_edge_idx[sf];
    subfacet_crack_dist[sf] = (eidx >= 0) ? edge_omega[eidx] * edge_kappa[eidx] * edge_l0[eidx] : Real(0);
}

__global__ void clear_f_int_ldpm_tet4_kernel(GPU_LDPMTet4_Data* d_data) {
    clear_internal_force(d_data);
}

__global__ void compute_f_int_ldpm_tet4_kernel(GPU_LDPMTet4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = d_data->gpu_n_edge() * 2;
    if (tid >= total)
        return;
    const int edge_idx = tid / 2;
    const int node_local = tid % 2;
    compute_internal_force(edge_idx, node_local, d_data);
}

// ─── GPU kernel: advance a set of particles' z-position by a fixed increment ─
//
// Used to implement velocity-driven (kinematic) BCs: the solver has already
// zeroed the velocity at these nodes; this kernel adds the prescribed
// displacement dz = v_prescribed * dt to their z-coordinate each step.
__global__ void advance_driven_nodes_z_kernel(Real* __restrict__ d_z,
                                              const int* __restrict__ d_driven_idx,
                                              int n_driven,
                                              Real dz) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_driven)
        return;
    d_z[d_driven_idx[tid]] += dz;
}

// ─── Host-side helper ────────────────────────────────────────────────────────

static inline Real triangle_area(const Real* a, const Real* b, const Real* c) {
    Real v1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    Real v2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    Real cross[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
    return Real(0.5) * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
}

// ─── Setup ────────────────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::Setup(const VectorXR& h_x,
                              const VectorXR& h_y,
                              const VectorXR& h_z,
                              const MatrixXi& tet_connectivity) {
    if (is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data is already set up.");
        return;
    }
    if (h_x.size() != n_nodes || h_y.size() != n_nodes || h_z.size() != n_nodes) {
        MOPHI_ERROR("Position vector size mismatch in GPU_LDPMTet4_Data::Setup.");
        return;
    }
    if (tet_connectivity.rows() != n_elem || tet_connectivity.cols() != 4) {
        MOPHI_ERROR("TET connectivity matrix has wrong dimensions.");
        return;
    }

    // ── 1. Allocate and fill translational position arrays ───────────────────

    da_x12.resize(n_nodes);
    da_x12.BindDevicePointer(&d_x12);
    da_y12.resize(n_nodes);
    da_y12.BindDevicePointer(&d_y12);
    da_z12.resize(n_nodes);
    da_z12.BindDevicePointer(&d_z12);

    std::copy(h_x.data(), h_x.data() + n_nodes, da_x12.host());
    da_x12.ToDevice();
    std::copy(h_y.data(), h_y.data() + n_nodes, da_y12.host());
    da_y12.ToDevice();
    std::copy(h_z.data(), h_z.data() + n_nodes, da_z12.host());
    da_z12.ToDevice();

    // ── 2. Allocate rotation arrays (initialised to zero) ────────────────────

    da_rx12.resize(n_nodes);
    da_rx12.BindDevicePointer(&d_rx12);
    da_rx12.SetVal(Real(0));
    da_rx12.ToDevice();

    da_ry12.resize(n_nodes);
    da_ry12.BindDevicePointer(&d_ry12);
    da_ry12.SetVal(Real(0));
    da_ry12.ToDevice();

    da_rz12.resize(n_nodes);
    da_rz12.BindDevicePointer(&d_rz12);
    da_rz12.SetVal(Real(0));
    da_rz12.ToDevice();

    // ── 3. Allocate force / moment arrays ────────────────────────────────────

    da_f_int_t.resize(n_nodes * 3);
    da_f_int_t.BindDevicePointer(&d_f_int_t);
    da_f_int_t.SetVal(Real(0));
    da_f_int_t.ToDevice();

    da_f_ext_t.resize(n_nodes * 3);
    da_f_ext_t.BindDevicePointer(&d_f_ext_t);
    da_f_ext_t.SetVal(Real(0));
    da_f_ext_t.ToDevice();

    da_f_int_r.resize(n_nodes * 3);
    da_f_int_r.BindDevicePointer(&d_f_int_r);
    da_f_int_r.SetVal(Real(0));
    da_f_int_r.ToDevice();

    da_f_ext_r.resize(n_nodes * 3);
    da_f_ext_r.BindDevicePointer(&d_f_ext_r);
    da_f_ext_r.SetVal(Real(0));
    da_f_ext_r.ToDevice();

    // ── 4. Store TET connectivity on host for CalcMassMatrix ─────────────────

    h_tet_conn_vec.resize(n_elem * 4);
    for (int e = 0; e < n_elem; e++) {
        for (int k = 0; k < 4; k++) {
            h_tet_conn_vec[e * 4 + k] = tet_connectivity(e, k);
        }
    }

    // ── 5. Extract unique edges ───────────────────────────────────────────────

    static const int LOCAL_EDGES[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    std::map<std::pair<int, int>, std::vector<int>> edge_to_tets;

    for (int e = 0; e < n_elem; e++) {
        for (int le = 0; le < 6; le++) {
            int a = tet_connectivity(e, LOCAL_EDGES[le][0]);
            int b = tet_connectivity(e, LOCAL_EDGES[le][1]);
            if (a > b)
                std::swap(a, b);
            edge_to_tets[{a, b}].push_back(e);
        }
    }

    n_edge = static_cast<int>(edge_to_tets.size());

    // ── 6. Compute edge geometry and facet areas ──────────────────────────────

    std::vector<int> h_edge_nodes(n_edge * 2);
    std::vector<Real> h_l0(n_edge);
    std::vector<Real> h_facet_area(n_edge);
    std::vector<Real> h_edge_n(n_edge * 3);
    std::vector<Real> h_edge_m(n_edge * 3);
    std::vector<Real> h_edge_lv(n_edge * 3);

    int edge_idx = 0;
    for (auto& kv : edge_to_tets) {
        const int ni = kv.first.first;
        const int nj = kv.first.second;
        const std::vector<int>& tets = kv.second;

        h_edge_nodes[edge_idx * 2 + 0] = ni;
        h_edge_nodes[edge_idx * 2 + 1] = nj;

        Real dx = h_x(nj) - h_x(ni);
        Real dy = h_y(nj) - h_y(ni);
        Real dz = h_z(nj) - h_z(ni);
        const Real l0_val = std::sqrt(dx * dx + dy * dy + dz * dz);
        h_l0[edge_idx] = l0_val;

        const Real inv_l0 = (l0_val > Real(0)) ? Real(1) / l0_val : Real(0);
        const Real nx = dx * inv_l0;
        const Real ny = dy * inv_l0;
        const Real nz = dz * inv_l0;
        h_edge_n[edge_idx * 3 + 0] = nx;
        h_edge_n[edge_idx * 3 + 1] = ny;
        h_edge_n[edge_idx * 3 + 2] = nz;

        Real mx, my, mz;
        if (std::abs(nx) <= std::abs(ny) && std::abs(nx) <= std::abs(nz)) {
            mx = Real(0);
            my = nz;
            mz = -ny;
        } else if (std::abs(ny) <= std::abs(nz)) {
            mx = -nz;
            my = Real(0);
            mz = nx;
        } else {
            mx = ny;
            my = -nx;
            mz = Real(0);
        }
        const Real m_len = std::sqrt(mx * mx + my * my + mz * mz);
        const Real inv_m_len = (m_len > Real(0)) ? Real(1) / m_len : Real(0);
        mx *= inv_m_len;
        my *= inv_m_len;
        mz *= inv_m_len;
        h_edge_m[edge_idx * 3 + 0] = mx;
        h_edge_m[edge_idx * 3 + 1] = my;
        h_edge_m[edge_idx * 3 + 2] = mz;

        const Real lvx = ny * mz - nz * my;
        const Real lvy = nz * mx - nx * mz;
        const Real lvz = nx * my - ny * mx;
        h_edge_lv[edge_idx * 3 + 0] = lvx;
        h_edge_lv[edge_idx * 3 + 1] = lvy;
        h_edge_lv[edge_idx * 3 + 2] = lvz;

        Real area_sum = Real(0);
        const Real xi = h_x(ni), yi = h_y(ni), zi = h_z(ni);
        const Real xj = h_x(nj), yj = h_y(nj), zj = h_z(nj);
        const Real mid[3] = {Real(0.5) * (xi + xj), Real(0.5) * (yi + yj), Real(0.5) * (zi + zj)};

        for (int t : tets) {
            int others[4];
            int n_others = 0;
            for (int k = 0; k < 4; k++) {
                int node = h_tet_conn_vec[t * 4 + k];
                if (node != ni && node != nj) {
                    others[n_others++] = node;
                }
            }
            if (n_others != 2)
                continue;

            const int nk = others[0];
            const int nl = others[1];
            const Real xk = h_x(nk), yk = h_y(nk), zk = h_z(nk);
            const Real xl = h_x(nl), yl = h_y(nl), zl = h_z(nl);

            const Real c_ijk[3] = {(xi + xj + xk) / Real(3), (yi + yj + yk) / Real(3), (zi + zj + zk) / Real(3)};
            const Real c_ijl[3] = {(xi + xj + xl) / Real(3), (yi + yj + yl) / Real(3), (zi + zj + zl) / Real(3)};
            const Real c_tet[3] = {(xi + xj + xk + xl) / Real(4), (yi + yj + yk + yl) / Real(4),
                                   (zi + zj + zk + zl) / Real(4)};

            area_sum += triangle_area(mid, c_ijk, c_tet);
            area_sum += triangle_area(mid, c_ijl, c_tet);
        }
        h_facet_area[edge_idx] = area_sum;

        ++edge_idx;
    }

    // ── 7. Keep host copies of edge data for CalcMassMatrix ───────────────────

    h_edge_nodes_vec = h_edge_nodes;
    h_l0_vec = h_l0;

    // ── 8. Allocate and copy edge data to device ──────────────────────────────

    da_edge_nodes.resize(static_cast<size_t>(n_edge) * 2);
    da_edge_nodes.BindDevicePointer(&d_edge_nodes);
    std::copy(h_edge_nodes.begin(), h_edge_nodes.end(), da_edge_nodes.host());
    da_edge_nodes.ToDevice();

    da_l0.resize(n_edge);
    da_l0.BindDevicePointer(&d_l0);
    std::copy(h_l0.begin(), h_l0.end(), da_l0.host());
    da_l0.ToDevice();

    da_facet_area.resize(n_edge);
    da_facet_area.BindDevicePointer(&d_facet_area);
    std::copy(h_facet_area.begin(), h_facet_area.end(), da_facet_area.host());
    da_facet_area.ToDevice();

    da_edge_n.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_n.BindDevicePointer(&d_edge_n);
    std::copy(h_edge_n.begin(), h_edge_n.end(), da_edge_n.host());
    da_edge_n.ToDevice();

    da_edge_m.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_m.BindDevicePointer(&d_edge_m);
    std::copy(h_edge_m.begin(), h_edge_m.end(), da_edge_m.host());
    da_edge_m.ToDevice();

    da_edge_lv.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_lv.BindDevicePointer(&d_edge_lv);
    std::copy(h_edge_lv.begin(), h_edge_lv.end(), da_edge_lv.host());
    da_edge_lv.ToDevice();

    // 6 traction/moment components per edge
    da_facet_t.resize(static_cast<size_t>(n_edge) * 6);
    da_facet_t.BindDevicePointer(&d_facet_t);
    da_facet_t.SetVal(Real(0));
    da_facet_t.ToDevice();

    // ── Per-edge damage state (initialised to zero / undamaged) ──────────────

    da_kappa.resize(n_edge);
    da_kappa.BindDevicePointer(&d_kappa);
    da_kappa.SetVal(Real(0));
    da_kappa.ToDevice();

    da_omega.resize(n_edge);
    da_omega.BindDevicePointer(&d_omega);
    da_omega.SetVal(Real(0));
    da_omega.ToDevice();

    // ── 9. Allocate device material scalars with safe defaults ────────────────

    MOPHI_GPU_CALL(cudaMalloc(&d_E_N, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_T, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kT, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kM, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kL, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_rho, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_sigma_t, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_H_t, sizeof(Real)));

    const Real default_val = Real(0);
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kT, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kM, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kL, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_sigma_t, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_H_t, &default_val, sizeof(Real), cudaMemcpyHostToDevice));

    // ── 10. Allocate per-node rotational inertia (filled in CalcMassMatrix) ───

    da_I_lump.resize(n_nodes);
    da_I_lump.BindDevicePointer(&d_I_lump);
    da_I_lump.SetVal(Real(0));
    da_I_lump.ToDevice();

    // ── 11. Allocate device-side struct mirror ────────────────────────────────

    MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_LDPMTet4_Data)));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));

    is_setup = true;

    MOPHI_INFO("GPU_LDPMTet4_Data: %d nodes, %d TET4 elements, %d unique edges", n_nodes, n_elem, n_edge);
}

// ─── SetupFromMesh ────────────────────────────────────────────────────────────
//
// Calls Setup() with particle positions and TET connectivity from the mesh,
// stores all additional file-loaded data fields, and — when facets.dat data
// is present — overrides the tet4-derived edge geometry (facet areas and
// tangent frame vectors m / l) with the richer file-based values.
//
// Facets.dat override
// ───────────────────
// Each row in facets.dat carries a Voronoi sub-facet for one edge of one TET:
//   Tet  IDx IDy IDz  Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  mF
// where p is the unit normal (= edge direction), q the first tangent, and
// s = p × q the second tangent.  12 sub-facets appear per TET (6 edges × 2).
//
// For each global unique edge the override:
//   facet_area  ← sum of pArea over all its sub-facets
//   edge_m      ← q from the first sub-facet (sign-adjusted for canonical dir)
//   edge_lv     ← s from the first sub-facet  (sign is unchanged; see below)
//
// Sign adjustment: the canonical edge direction stored in edge_n[] points from
// the lower-index node to the higher-index one.  If a sub-facet's p points in
// the opposite direction the tangent q must be negated so that l = n × m = s
// still holds for the canonical frame.

void GPU_LDPMTet4_Data::SetupFromMesh(const LDPMTet4Mesh& mesh) {
    if (mesh.n_particles <= 0) {
        MOPHI_ERROR("SetupFromMesh: mesh has no particles (particles.dat not loaded?).");
        return;
    }
    if (mesh.n_tets <= 0) {
        MOPHI_ERROR("SetupFromMesh: mesh has no TET elements (tets.dat not loaded?).");
        return;
    }

    // Update element counts to match what the mesh actually provides.
    n_nodes = mesh.n_particles;
    n_elem = mesh.n_tets;

    Setup(mesh.particle_x, mesh.particle_y, mesh.particle_z, mesh.tet_connectivity);

    // ── Particle diameters ───────────────────────────────────────────────────
    if (mesh.particle_d.size() == mesh.n_particles) {
        h_particle_d = mesh.particle_d;
    }

    // ── Sub-facet geometry ───────────────────────────────────────────────────
    if (mesh.n_subfacets > 0) {
        n_subfacet = mesh.n_subfacets;
        h_subfacet_tet = mesh.subfacet_tet;
        h_subfacet_vertex_ids = mesh.subfacet_vertex_ids;
        h_subfacet_vol = mesh.subfacet_vol;
        h_subfacet_parea = mesh.subfacet_parea;
        h_subfacet_centroid = mesh.subfacet_centroid;
        h_subfacet_normal = mesh.subfacet_normal;
        h_subfacet_tangent_q = mesh.subfacet_tangent_q;
        h_subfacet_tangent_s = mesh.subfacet_tangent_s;
        h_subfacet_matflag = mesh.subfacet_matflag;
    }

    // ── Surface boundary triangles ───────────────────────────────────────────
    if (mesh.n_face_facets > 0) {
        n_face_facet = mesh.n_face_facets;
        h_face_facet_node = mesh.face_facet_node;
        h_face_facet_vertices = mesh.face_facet_vertices;
    }

    // ── Facet vertex coordinates ─────────────────────────────────────────────
    if (mesh.n_facet_vertices > 0) {
        n_facet_vertex = mesh.n_facet_vertices;
        h_facet_vertex_x = mesh.facet_vertex_x;
        h_facet_vertex_y = mesh.facet_vertex_y;
        h_facet_vertex_z = mesh.facet_vertex_z;
    }

    MOPHI_INFO("SetupFromMesh: stored %d sub-facets, %d face-facets, %d facet vertices", n_subfacet, n_face_facet,
               n_facet_vertex);

    // ── Override edge geometry from sub-facet file data ──────────────────────
    // Only performed when sub-facet data was actually loaded.
    if (n_subfacet <= 0) {
        return;
    }

    static const int LOCAL_EDGES_SF[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    // Build a fast lookup: sorted node pair → global edge index.
    std::map<std::pair<int, int>, int> edge_index_map;
    for (int e = 0; e < n_edge; e++) {
        const int ni = h_edge_nodes_vec[e * 2 + 0];
        const int nj = h_edge_nodes_vec[e * 2 + 1];
        edge_index_map[{std::min(ni, nj), std::max(ni, nj)}] = e;
    }

    // Per-edge accumulators.
    std::vector<Real> new_area(n_edge, Real(0));
    std::vector<Real> new_m(n_edge * 3, Real(0));
    std::vector<Real> new_l(n_edge * 3, Real(0));
    std::vector<bool> frame_set(n_edge, false);

    // Allocate and initialise the subfacet→edge mapping array.
    // Entries are set to -1 (unmatched) and filled during the loop below.
    da_subfacet_edge_idx.resize(n_subfacet);
    da_subfacet_edge_idx.BindDevicePointer(&d_subfacet_edge_idx);
    std::fill(da_subfacet_edge_idx.host(), da_subfacet_edge_idx.host() + n_subfacet, -1);

    const Real* hx = da_x12.host();
    const Real* hy = da_y12.host();
    const Real* hz = da_z12.host();

    int n_unmatched = 0;

    for (int sf = 0; sf < n_subfacet; sf++) {
        const int t = h_subfacet_tet(sf);

        const Real px = h_subfacet_normal(sf * 3 + 0);
        const Real py = h_subfacet_normal(sf * 3 + 1);
        const Real pz = h_subfacet_normal(sf * 3 + 2);
        const Real parea = h_subfacet_parea(sf);

        // Find the local edge of TET t whose unit direction best matches p.
        int best_le = -1;
        Real best_abs_dot = Real(0);
        Real best_sign = Real(1);

        for (int le = 0; le < 6; le++) {
            const int na = h_tet_conn_vec[t * 4 + LOCAL_EDGES_SF[le][0]];
            const int nb = h_tet_conn_vec[t * 4 + LOCAL_EDGES_SF[le][1]];

            const Real dx = hx[nb] - hx[na];
            const Real dy = hy[nb] - hy[na];
            const Real dz = hz[nb] - hz[na];
            const Real len = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (len <= Real(0))
                continue;
            const Real inv_len = Real(1) / len;

            const Real d = (px * dx + py * dy + pz * dz) * inv_len;
            const Real abs_d = std::abs(d);
            if (abs_d > best_abs_dot) {
                best_abs_dot = abs_d;
                best_le = le;
                // sign = dot(p, canonical direction na→nb)
                // canonical direction is ni→nj where ni = min(na,nb)
                // so sign relative to canonical = d if na < nb, else -d
                best_sign = (na < nb) ? d : -d;
            }
        }

        if (best_le < 0 || best_abs_dot < Real(0.99)) {
            ++n_unmatched;
            continue;
        }

        // Resolve global edge index.
        const int na = h_tet_conn_vec[t * 4 + LOCAL_EDGES_SF[best_le][0]];
        const int nb = h_tet_conn_vec[t * 4 + LOCAL_EDGES_SF[best_le][1]];
        const int ni = std::min(na, nb);
        const int nj = std::max(na, nb);

        auto it = edge_index_map.find({ni, nj});
        if (it == edge_index_map.end()) {
            ++n_unmatched;
            continue;
        }
        const int eidx = it->second;

        // Record which global edge this sub-facet belongs to.
        da_subfacet_edge_idx.host()[sf] = eidx;

        // Accumulate projected area.
        new_area[eidx] += parea;

        // Store the tangent frame from the first matching sub-facet.
        // Sign adjustment for m (q): if p is anti-parallel to the canonical
        // edge direction, negate q so that l = n_canonical × m = s still holds.
        //
        // Why l (= s from file) needs no sign adjustment:
        //   File convention: s = p × q
        //   sign > 0  (p ≈ +n_canonical):
        //     m = q,   l = n_canonical × q = p × q = s ✓
        //   sign < 0  (p ≈ −n_canonical):
        //     m = −q,  l = n_canonical × (−q) = −(n_canonical × q)
        //              = −((−p) × q) = p × q = s ✓
        //   In both cases l = s, so s is copied without negation.
        if (!frame_set[eidx]) {
            const Real q0 = h_subfacet_tangent_q(sf * 3 + 0);
            const Real q1 = h_subfacet_tangent_q(sf * 3 + 1);
            const Real q2 = h_subfacet_tangent_q(sf * 3 + 2);
            const Real s0 = h_subfacet_tangent_s(sf * 3 + 0);
            const Real s1 = h_subfacet_tangent_s(sf * 3 + 1);
            const Real s2 = h_subfacet_tangent_s(sf * 3 + 2);

            const Real sign_m = (best_sign > Real(0)) ? Real(1) : Real(-1);
            new_m[eidx * 3 + 0] = sign_m * q0;
            new_m[eidx * 3 + 1] = sign_m * q1;
            new_m[eidx * 3 + 2] = sign_m * q2;
            new_l[eidx * 3 + 0] = s0;
            new_l[eidx * 3 + 1] = s1;
            new_l[eidx * 3 + 2] = s2;
            frame_set[eidx] = true;
        }
    }

    if (n_unmatched > 0) {
        MOPHI_WARNING("SetupFromMesh: %d sub-facet(s) could not be matched to a global edge.", n_unmatched);
    }

    // Count edges whose frame was set; edges with no sub-facet match keep the
    // tet4-derived geometry (unlikely but handled gracefully).
    int n_frame_set = 0;
    for (int e = 0; e < n_edge; e++) {
        if (frame_set[e])
            ++n_frame_set;
    }

    // Write updated arrays back to device.
    std::copy(new_area.begin(), new_area.end(), da_facet_area.host());
    da_facet_area.ToDevice();

    std::copy(new_m.begin(), new_m.end(), da_edge_m.host());
    da_edge_m.ToDevice();

    std::copy(new_l.begin(), new_l.end(), da_edge_lv.host());
    da_edge_lv.ToDevice();

    // Upload subfacet→edge mapping to device.
    da_subfacet_edge_idx.ToDevice();

    // Sync device struct mirror so kernels see updated pointers.
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));

    MOPHI_INFO("SetupFromMesh: overrode facet area and tangent frame for %d/%d edges from sub-facet file data.",
               n_frame_set, n_edge);
}

// ─── Material / parameter setters ────────────────────────────────────────────

void GPU_LDPMTet4_Data::SetMaterial(Real E_N_val, Real E_T_val, Real E_kT_val, Real E_kM_val, Real E_kL_val) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting material.");
        return;
    }
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &E_N_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &E_T_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kT, &E_kT_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kM, &E_kM_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kL, &E_kL_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));
}

void GPU_LDPMTet4_Data::SetDensity(Real rho_val) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting density.");
        return;
    }
    h_rho = rho_val;
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &rho_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));
}

void GPU_LDPMTet4_Data::SetDamageParams(Real sigma_t_val, Real H_t_val) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting damage parameters.");
        return;
    }
    MOPHI_GPU_CALL(cudaMemcpy(d_sigma_t, &sigma_t_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_H_t, &H_t_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));
}

void GPU_LDPMTet4_Data::SetExternalForce(const VectorXR& h_f_ext) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting external force.");
        return;
    }
    if (h_f_ext.size() != n_nodes * 3) {
        MOPHI_ERROR("External force vector size mismatch (expected %d, got %d).", n_nodes * 3,
                    static_cast<int>(h_f_ext.size()));
        return;
    }
    std::copy(h_f_ext.data(), h_f_ext.data() + n_nodes * 3, da_f_ext_t.host());
    da_f_ext_t.ToDevice();
}

void GPU_LDPMTet4_Data::SetNodalFixed(const VectorXi& fixed_nodes_in) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting fixed nodes.");
        return;
    }
    n_constraint = static_cast<int>(fixed_nodes_in.size()) * 6;

    da_fixed_nodes.resize(static_cast<size_t>(fixed_nodes_in.size()));
    da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);
    std::copy(fixed_nodes_in.data(), fixed_nodes_in.data() + fixed_nodes_in.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    is_constraints_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));
}

// ─── CalcMassMatrix ──────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::CalcMassMatrix() {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before CalcMassMatrix.");
        return;
    }

    const Real* hx = da_x12.host();
    const Real* hy = da_y12.host();
    const Real* hz = da_z12.host();

    // ── Voronoi volumes → translational lumped mass ───────────────────────────
    std::vector<Real> node_volumes(n_nodes, Real(0));

    for (int e = 0; e < n_elem; e++) {
        const int n0 = h_tet_conn_vec[e * 4 + 0];
        const int n1 = h_tet_conn_vec[e * 4 + 1];
        const int n2 = h_tet_conn_vec[e * 4 + 2];
        const int n3 = h_tet_conn_vec[e * 4 + 3];

        const Real a[3] = {hx[n1] - hx[n0], hy[n1] - hy[n0], hz[n1] - hz[n0]};
        const Real b[3] = {hx[n2] - hx[n0], hy[n2] - hy[n0], hz[n2] - hz[n0]};
        const Real c[3] = {hx[n3] - hx[n0], hy[n3] - hy[n0], hz[n3] - hz[n0]};

        const Real cross_bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
        const Real vol = std::abs(a[0] * cross_bc[0] + a[1] * cross_bc[1] + a[2] * cross_bc[2]) / Real(6);

        node_volumes[n0] += vol / Real(4);
        node_volumes[n1] += vol / Real(4);
        node_volumes[n2] += vol / Real(4);
        node_volumes[n3] += vol / Real(4);
    }

    // ── Per-node minimum edge length → rotational inertia ────────────────────
    std::vector<Real> node_l_min(n_nodes, std::numeric_limits<Real>::max());

    for (int e = 0; e < n_edge; e++) {
        const int ni = h_edge_nodes_vec[e * 2 + 0];
        const int nj = h_edge_nodes_vec[e * 2 + 1];
        const Real l = h_l0_vec[e];
        if (l < node_l_min[ni])
            node_l_min[ni] = l;
        if (l < node_l_min[nj])
            node_l_min[nj] = l;
    }

    // Guard against isolated nodes (no edges connected)
    for (int i = 0; i < n_nodes; i++) {
        if (node_l_min[i] == std::numeric_limits<Real>::max())
            node_l_min[i] = Real(0);
    }

    // I_lump[i] = alpha * m_lump[i] * l_min[i]^2
    std::vector<Real> h_I_lump(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        const Real m_i = h_rho * node_volumes[i];
        h_I_lump[i] = LDPM_TET4_ALPHA_ROT * m_i * node_l_min[i] * node_l_min[i];
    }

    std::copy(h_I_lump.begin(), h_I_lump.end(), da_I_lump.host());
    da_I_lump.ToDevice();

    // ── Build diagonal CSR for translational mass ─────────────────────────────
    std::vector<int> h_offsets(n_nodes + 1);
    std::vector<int> h_columns(n_nodes);
    std::vector<Real> h_values(n_nodes);

    for (int i = 0; i < n_nodes; i++) {
        h_offsets[i] = i;
        h_columns[i] = i;
        h_values[i] = h_rho * node_volumes[i];
    }
    h_offsets[n_nodes] = n_nodes;

    if (!is_csr_setup) {
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_offsets, static_cast<size_t>(n_nodes + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_columns, static_cast<size_t>(n_nodes) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_values, static_cast<size_t>(n_nodes) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_nnz, sizeof(int)));
        is_csr_setup = true;
    }

    MOPHI_GPU_CALL(cudaMemcpy(d_csr_offsets, h_offsets.data(), static_cast<size_t>(n_nodes + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(
        cudaMemcpy(d_csr_columns, h_columns.data(), static_cast<size_t>(n_nodes) * sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(
        cudaMemcpy(d_csr_values, h_values.data(), static_cast<size_t>(n_nodes) * sizeof(Real), cudaMemcpyHostToDevice));

    const int nnz_val = n_nodes;
    MOPHI_GPU_CALL(cudaMemcpy(d_nnz, &nnz_val, sizeof(int), cudaMemcpyHostToDevice));

    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));
}

// ─── CalcP ────────────────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::CalcP() {
    if (!is_setup)
        return;
    if (n_edge == 0)
        return;
    constexpr int threads = 256;
    const int blocks = (n_edge + threads - 1) / threads;
    calc_p_ldpm_tet4_kernel<<<blocks, threads>>>(d_data);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── CalcInternalForce ────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::CalcInternalForce() {
    if (!is_setup)
        return;
    if (n_edge == 0)
        return;
    constexpr int threads = 256;

    // Clear both translational and rotational f_int (n_nodes * 3 threads)
    const int blocks_clear = (n_nodes * 3 + threads - 1) / threads;
    clear_f_int_ldpm_tet4_kernel<<<blocks_clear, threads>>>(d_data);

    // Assemble from edges (n_edge * 2 threads)
    const int blocks_assemble = (n_edge * 2 + threads - 1) / threads;
    compute_f_int_ldpm_tet4_kernel<<<blocks_assemble, threads>>>(d_data);

    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── Retrieve helpers ─────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::RetrievePositionToCPU(VectorXR& x_out, VectorXR& y_out, VectorXR& z_out) {
    x_out.resize(n_nodes);
    y_out.resize(n_nodes);
    z_out.resize(n_nodes);
    da_x12.ToHost();
    da_y12.ToHost();
    da_z12.ToHost();
    std::copy(da_x12.host(), da_x12.host() + n_nodes, x_out.data());
    std::copy(da_y12.host(), da_y12.host() + n_nodes, y_out.data());
    std::copy(da_z12.host(), da_z12.host() + n_nodes, z_out.data());
}

void GPU_LDPMTet4_Data::RetrieveRotationToCPU(VectorXR& rx_out, VectorXR& ry_out, VectorXR& rz_out) {
    rx_out.resize(n_nodes);
    ry_out.resize(n_nodes);
    rz_out.resize(n_nodes);
    da_rx12.ToHost();
    da_ry12.ToHost();
    da_rz12.ToHost();
    std::copy(da_rx12.host(), da_rx12.host() + n_nodes, rx_out.data());
    std::copy(da_ry12.host(), da_ry12.host() + n_nodes, ry_out.data());
    std::copy(da_rz12.host(), da_rz12.host() + n_nodes, rz_out.data());
}

void GPU_LDPMTet4_Data::RetrieveInternalForceToCPU(VectorXR& f_out) {
    f_out.resize(n_nodes * 3);
    da_f_int_t.ToHost();
    std::copy(da_f_int_t.host(), da_f_int_t.host() + n_nodes * 3, f_out.data());
}

void GPU_LDPMTet4_Data::RetrieveExternalForceToCPU(VectorXR& f_out) {
    f_out.resize(n_nodes * 3);
    da_f_ext_t.ToHost();
    std::copy(da_f_ext_t.host(), da_f_ext_t.host() + n_nodes * 3, f_out.data());
}

void GPU_LDPMTet4_Data::RetrieveFacetDamageToCPU(VectorXR& omega_out) {
    omega_out.resize(n_edge);
    da_omega.ToHost();
    std::copy(da_omega.host(), da_omega.host() + n_edge, omega_out.data());
}

void GPU_LDPMTet4_Data::RetrieveFacetKappaToCPU(VectorXR& kappa_out) {
    kappa_out.resize(n_edge);
    da_kappa.ToHost();
    std::copy(da_kappa.host(), da_kappa.host() + n_edge, kappa_out.data());
}

// ─── ProjectEdgeDamageToSubfacets ────────────────────────────────────────────
//
// For each sub-facet, looks up omega[subfacet_edge_idx[sf]] via a GPU scatter
// kernel and returns a flat array of size n_subfacet.
//
// Requires SetupFromMesh() to have been called so that d_subfacet_edge_idx and
// d_omega are valid device pointers.

void GPU_LDPMTet4_Data::ProjectEdgeDamageToSubfacets(VectorXR& out) {
    if (!is_setup || n_subfacet <= 0 || d_subfacet_edge_idx == nullptr) {
        out.resize(0);
        return;
    }

    // Temporary device output buffer.
    Real* d_sf_dmg = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_sf_dmg, static_cast<size_t>(n_subfacet) * sizeof(Real)));

    constexpr int threads = 256;
    const int blocks = (n_subfacet + threads - 1) / threads;
    project_edge_damage_to_subfacets_kernel<<<blocks, threads>>>(n_subfacet, d_subfacet_edge_idx, d_omega, d_sf_dmg);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    out.resize(n_subfacet);
    MOPHI_GPU_CALL(
        cudaMemcpy(out.data(), d_sf_dmg, static_cast<size_t>(n_subfacet) * sizeof(Real), cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaFree(d_sf_dmg));
}

// ─── ProjectEdgeCrackDistanceToSubfacets ─────────────────────────────────────
//
// Computes the crack opening displacement per sub-facet as ω × κ × l₀.
// Each entry gives the inelastic (damage) opening in the same length units as
// the mesh (mm for the Dogbone demos).

void GPU_LDPMTet4_Data::ProjectEdgeCrackDistanceToSubfacets(VectorXR& out) {
    if (!is_setup || n_subfacet <= 0 || d_subfacet_edge_idx == nullptr) {
        out.resize(0);
        return;
    }

    Real* d_sf_crack = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_sf_crack, static_cast<size_t>(n_subfacet) * sizeof(Real)));

    constexpr int threads = 256;
    const int blocks = (n_subfacet + threads - 1) / threads;
    project_edge_crack_dist_to_subfacets_kernel<<<blocks, threads>>>(n_subfacet, d_subfacet_edge_idx, d_omega, d_kappa,
                                                                     d_l0, d_sf_crack);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());

    out.resize(n_subfacet);
    MOPHI_GPU_CALL(
        cudaMemcpy(out.data(), d_sf_crack, static_cast<size_t>(n_subfacet) * sizeof(Real), cudaMemcpyDeviceToHost));
    MOPHI_GPU_CALL(cudaFree(d_sf_crack));
}

// ─── Velocity-driven (kinematic) BC helper ────────────────────────────────────

void GPU_LDPMTet4_Data::AdvanceDrivenNodesZ(const int* d_driven_idx, int n_driven, Real dz) {
    if (n_driven <= 0)
        return;
    constexpr int threads = 256;
    const int blocks = (n_driven + threads - 1) / threads;
    advance_driven_nodes_z_kernel<<<blocks, threads>>>(d_z12, d_driven_idx, n_driven, dz);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── Destroy ─────────────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::Destroy() {
    da_x12.free();
    da_y12.free();
    da_z12.free();
    da_rx12.free();
    da_ry12.free();
    da_rz12.free();
    da_f_int_t.free();
    da_f_ext_t.free();
    da_f_int_r.free();
    da_f_ext_r.free();
    da_edge_nodes.free();
    da_l0.free();
    da_facet_area.free();
    da_edge_n.free();
    da_edge_m.free();
    da_edge_lv.free();
    da_facet_t.free();
    da_kappa.free();
    da_omega.free();
    da_subfacet_edge_idx.free();
    da_fixed_nodes.free();
    da_I_lump.free();

    if (is_csr_setup) {
        MOPHI_GPU_CALL(cudaFree(d_csr_offsets));
        MOPHI_GPU_CALL(cudaFree(d_csr_columns));
        MOPHI_GPU_CALL(cudaFree(d_csr_values));
        MOPHI_GPU_CALL(cudaFree(d_nnz));
    }
    MOPHI_GPU_CALL(cudaFree(d_E_N));
    MOPHI_GPU_CALL(cudaFree(d_E_T));
    MOPHI_GPU_CALL(cudaFree(d_E_kT));
    MOPHI_GPU_CALL(cudaFree(d_E_kM));
    MOPHI_GPU_CALL(cudaFree(d_E_kL));
    MOPHI_GPU_CALL(cudaFree(d_rho));
    MOPHI_GPU_CALL(cudaFree(d_sigma_t));
    MOPHI_GPU_CALL(cudaFree(d_H_t));
    MOPHI_GPU_CALL(cudaFree(d_data));
}

}  // namespace feris
