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

namespace {

constexpr int LDPM_TET4_LOCAL_EDGES[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

__host__ __device__ __forceinline__ Real ldpm_tet_abs_volume(Real x0,
                                                             Real y0,
                                                             Real z0,
                                                             Real x1,
                                                             Real y1,
                                                             Real z1,
                                                             Real x2,
                                                             Real y2,
                                                             Real z2,
                                                             Real x3,
                                                             Real y3,
                                                             Real z3) {
    const Real a0 = x1 - x0;
    const Real a1 = y1 - y0;
    const Real a2 = z1 - z0;

    const Real b0 = x2 - x0;
    const Real b1 = y2 - y0;
    const Real b2 = z2 - z0;

    const Real c0 = x3 - x0;
    const Real c1 = y3 - y0;
    const Real c2 = z3 - z0;

    const Real bc0 = b1 * c2 - b2 * c1;
    const Real bc1 = b2 * c0 - b0 * c2;
    const Real bc2 = b0 * c1 - b1 * c0;

    const Real det = a0 * bc0 + a1 * bc1 + a2 * bc2;
    return ((det >= Real(0)) ? det : -det) / Real(6);
}

static inline Real
ldpm_host_tet_abs_volume(const Real* x, const Real* y, const Real* z, int n0, int n1, int n2, int n3) {
    return ldpm_tet_abs_volume(x[n0], y[n0], z[n0], x[n1], y[n1], z[n1], x[n2], y[n2], z[n2], x[n3], y[n3], z[n3]);
}

}  // namespace

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

// ─── GPU kernel: compute per-tet volumetric strain ───────────────────────────
//
// For each tet, computes current volume from the 4 node positions and
// returns eps_V = (V_cur - V_ref) / (3 * V_ref), matching the CPU reference
// in chrono-mechanics ChElementLDPM::ComputeVolume().
__global__ void compute_tet_volumetric_strain_kernel(int n_elem,
                                                     const int* __restrict__ tet_conn,
                                                     const Real* __restrict__ x_cur,
                                                     const Real* __restrict__ y_cur,
                                                     const Real* __restrict__ z_cur,
                                                     const Real* __restrict__ vol_ref,
                                                     Real* __restrict__ vol_strain) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elem)
        return;

    const int n0 = tet_conn[tid * 4 + 0];
    const int n1 = tet_conn[tid * 4 + 1];
    const int n2 = tet_conn[tid * 4 + 2];
    const int n3 = tet_conn[tid * 4 + 3];

    const Real V_cur = ldpm_tet_abs_volume(x_cur[n0], y_cur[n0], z_cur[n0], x_cur[n1], y_cur[n1], z_cur[n1], x_cur[n2],
                                           y_cur[n2], z_cur[n2], x_cur[n3], y_cur[n3], z_cur[n3]);
    const Real V_ref = vol_ref[tid];
    vol_strain[tid] = (V_ref > Real(0)) ? (V_cur - V_ref) / (Real(3) * V_ref) : Real(0);
}

// ─── GPU kernel: average tet volumetric strain onto edges ────────────────────
//
// For each edge, averages the volumetric strain of the tets sharing that edge
// using a CSR (compressed sparse row) mapping.
__global__ void average_tet_volumetric_strain_to_edges_kernel(int n_edge,
                                                              const int* __restrict__ edge_tet_offsets,
                                                              const int* __restrict__ edge_tet_indices,
                                                              const Real* __restrict__ tet_vol_strain,
                                                              Real* __restrict__ edge_vol_strain) {
    const int eidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (eidx >= n_edge)
        return;

    const int start = edge_tet_offsets[eidx];
    const int end = edge_tet_offsets[eidx + 1];
    const int count = end - start;

    if (count <= 0) {
        edge_vol_strain[eidx] = Real(0);
        return;
    }

    Real sum = Real(0);
    for (int i = start; i < end; ++i) {
        sum += tet_vol_strain[edge_tet_indices[i]];
    }
    edge_vol_strain[eidx] = sum / static_cast<Real>(count);
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

    da_x_cur.resize(n_nodes);
    da_x_cur.BindDevicePointer(&d_x_cur);
    da_y_cur.resize(n_nodes);
    da_y_cur.BindDevicePointer(&d_y_cur);
    da_z_cur.resize(n_nodes);
    da_z_cur.BindDevicePointer(&d_z_cur);

    std::copy(h_x.data(), h_x.data() + n_nodes, da_x_cur.host());
    da_x_cur.ToDevice();
    std::copy(h_y.data(), h_y.data() + n_nodes, da_y_cur.host());
    da_y_cur.ToDevice();
    std::copy(h_z.data(), h_z.data() + n_nodes, da_z_cur.host());
    da_z_cur.ToDevice();

    da_x_ref.resize(n_nodes);
    da_x_ref.BindDevicePointer(&d_x_ref);
    std::copy(h_x.data(), h_x.data() + n_nodes, da_x_ref.host());
    da_x_ref.ToDevice();

    da_y_ref.resize(n_nodes);
    da_y_ref.BindDevicePointer(&d_y_ref);
    std::copy(h_y.data(), h_y.data() + n_nodes, da_y_ref.host());
    da_y_ref.ToDevice();

    da_z_ref.resize(n_nodes);
    da_z_ref.BindDevicePointer(&d_z_ref);
    std::copy(h_z.data(), h_z.data() + n_nodes, da_z_ref.host());
    da_z_ref.ToDevice();

    // ── 2. Allocate rotation arrays (initialised to zero) ────────────────────

    da_rot_x_cur.resize(n_nodes);
    da_rot_x_cur.BindDevicePointer(&d_rot_x_cur);
    da_rot_x_cur.SetVal(Real(0));
    da_rot_x_cur.ToDevice();

    da_rot_y_cur.resize(n_nodes);
    da_rot_y_cur.BindDevicePointer(&d_rot_y_cur);
    da_rot_y_cur.SetVal(Real(0));
    da_rot_y_cur.ToDevice();

    da_rot_z_cur.resize(n_nodes);
    da_rot_z_cur.BindDevicePointer(&d_rot_z_cur);
    da_rot_z_cur.SetVal(Real(0));
    da_rot_z_cur.ToDevice();

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

    std::map<std::pair<int, int>, std::vector<int>> edge_to_tets;

    for (int e = 0; e < n_elem; e++) {
        for (int le = 0; le < 6; le++) {
            int a = tet_connectivity(e, LDPM_TET4_LOCAL_EDGES[le][0]);
            int b = tet_connectivity(e, LDPM_TET4_LOCAL_EDGES[le][1]);
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
    std::vector<Real> h_edge_center(n_edge * 3);

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
        h_edge_center[edge_idx * 3 + 0] = mid[0];
        h_edge_center[edge_idx * 3 + 1] = mid[1];
        h_edge_center[edge_idx * 3 + 2] = mid[2];

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

    da_edge_center.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_center.BindDevicePointer(&d_edge_center);
    std::copy(h_edge_center.begin(), h_edge_center.end(), da_edge_center.host());
    da_edge_center.ToDevice();

    // LDPM_FACET_N_COMPONENTS traction/moment components per edge
    da_facet_t.resize(static_cast<size_t>(n_edge) * LDPM_FACET_N_COMPONENTS);
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

    // ── Volumetric strain averaging: tet connectivity, reference volumes, ────
    //    edge-to-tet CSR mapping, and per-edge volumetric strain scratch.

    // Store tet connectivity on device [n_elem * 4]
    da_tet_conn.resize(static_cast<size_t>(n_elem) * 4);
    da_tet_conn.BindDevicePointer(&d_tet_conn);
    std::copy(h_tet_conn_vec.begin(), h_tet_conn_vec.end(), da_tet_conn.host());
    da_tet_conn.ToDevice();

    // Compute and store reference tet volumes [n_elem]
    da_tet_vol_ref.resize(n_elem);
    da_tet_vol_ref.BindDevicePointer(&d_tet_vol_ref);
    for (int e = 0; e < n_elem; e++) {
        da_tet_vol_ref.host()[e] =
            ldpm_host_tet_abs_volume(h_x.data(), h_y.data(), h_z.data(), h_tet_conn_vec[e * 4 + 0],
                                     h_tet_conn_vec[e * 4 + 1], h_tet_conn_vec[e * 4 + 2], h_tet_conn_vec[e * 4 + 3]);
    }
    da_tet_vol_ref.ToDevice();

    // Per-tet volumetric strain scratch [n_elem] (recomputed each step)
    da_tet_vol_strain.resize(n_elem);
    da_tet_vol_strain.BindDevicePointer(&d_tet_vol_strain);
    da_tet_vol_strain.SetVal(Real(0));
    da_tet_vol_strain.ToDevice();

    // Build CSR edge-to-tet mapping from edge_to_tets map (still in scope)
    {
        std::vector<int> h_offsets(n_edge + 1, 0);
        std::vector<int> h_indices;

        int ei = 0;
        for (const auto& kv : edge_to_tets) {
            h_offsets[ei + 1] = h_offsets[ei] + static_cast<int>(kv.second.size());
            for (int t : kv.second) {
                h_indices.push_back(t);
            }
            ++ei;
        }

        da_edge_tet_offsets.resize(static_cast<size_t>(n_edge) + 1);
        da_edge_tet_offsets.BindDevicePointer(&d_edge_tet_offsets);
        std::copy(h_offsets.begin(), h_offsets.end(), da_edge_tet_offsets.host());
        da_edge_tet_offsets.ToDevice();

        da_edge_tet_indices.resize(h_indices.size());
        da_edge_tet_indices.BindDevicePointer(&d_edge_tet_indices);
        std::copy(h_indices.begin(), h_indices.end(), da_edge_tet_indices.host());
        da_edge_tet_indices.ToDevice();
    }

    // Per-edge volumetric strain [n_edge] (averaged from tets each step)
    da_edge_vol_strain.resize(n_edge);
    da_edge_vol_strain.BindDevicePointer(&d_edge_vol_strain);
    da_edge_vol_strain.SetVal(Real(0));
    da_edge_vol_strain.ToDevice();

    // ── 9. Allocate device material scalars with safe defaults ────────────────

    MOPHI_GPU_CALL(cudaMalloc(&d_E_N, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_T, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kT, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kM, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_kL, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_rho, sizeof(Real)));

    const Real default_val = Real(0);
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kT, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kM, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kL, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &default_val, sizeof(Real), cudaMemcpyHostToDevice));

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
// is present — replaces the unique-edge fallback interactions with one
// interaction per sub-facet, matching ChElementLDPM's 12 sections per TET.
//
// Facets.dat override
// ───────────────────
// Each row in facets.dat carries a Voronoi sub-facet for one edge of one TET:
//   Tet  IDx IDy IDz  Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  mF
// where p is the unit normal (= edge direction), q the first tangent, and
// s = p × q the second tangent.  12 sub-facets appear per TET (6 edges × 2).
//
// Each sub-facet keeps its own pArea, center, and p/q/s frame. Its endpoint
// order follows ChElementLDPM::facetNodeNums (two sub-facets per local edge).

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

    // ── Replace unique-edge fallback with sub-facet interactions ─────────────
    // Only performed when sub-facet data was actually loaded.
    if (n_subfacet <= 0) {
        return;
    }

    n_edge = n_subfacet;
    h_edge_nodes_vec.resize(static_cast<size_t>(n_edge) * 2);
    h_l0_vec.resize(n_edge);

    da_edge_nodes.resize(static_cast<size_t>(n_edge) * 2);
    da_l0.resize(n_edge);
    da_facet_area.resize(n_edge);
    da_edge_n.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_m.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_lv.resize(static_cast<size_t>(n_edge) * 3);
    da_edge_center.resize(static_cast<size_t>(n_edge) * 3);
    da_facet_t.resize(static_cast<size_t>(n_edge) * LDPM_FACET_N_COMPONENTS);
    da_kappa.resize(n_edge);
    da_omega.resize(n_edge);
    da_edge_tet_offsets.resize(static_cast<size_t>(n_edge) + 1);
    da_edge_tet_indices.resize(n_edge);
    da_edge_vol_strain.resize(n_edge);

    da_subfacet_edge_idx.resize(n_subfacet);
    da_subfacet_edge_idx.BindDevicePointer(&d_subfacet_edge_idx);

    std::vector<int> tet_facet_count(static_cast<size_t>(n_elem), 0);
    for (int sf = 0; sf < n_subfacet; sf++) {
        const int t = h_subfacet_tet(sf);
        if (t < 0 || t >= n_elem) {
            MOPHI_ERROR("SetupFromMesh: sub-facet %d has invalid owning TET %d.", sf, t);
            return;
        }
        const int local_facet = tet_facet_count[static_cast<size_t>(t)]++;
        if (local_facet >= 12) {
            MOPHI_ERROR("SetupFromMesh: TET %d has more than 12 sub-facets.", t);
            return;
        }
        const int local_edge = local_facet / 2;
        const int ni = h_tet_conn_vec[t * 4 + LDPM_TET4_LOCAL_EDGES[local_edge][0]];
        const int nj = h_tet_conn_vec[t * 4 + LDPM_TET4_LOCAL_EDGES[local_edge][1]];

        h_edge_nodes_vec[sf * 2 + 0] = ni;
        h_edge_nodes_vec[sf * 2 + 1] = nj;
        da_edge_nodes.host()[sf * 2 + 0] = ni;
        da_edge_nodes.host()[sf * 2 + 1] = nj;

        const Real dx = mesh.particle_x(nj) - mesh.particle_x(ni);
        const Real dy = mesh.particle_y(nj) - mesh.particle_y(ni);
        const Real dz = mesh.particle_z(nj) - mesh.particle_z(ni);
        const Real length = std::sqrt(dx * dx + dy * dy + dz * dz);
        h_l0_vec[sf] = length;
        da_l0.host()[sf] = length;
        da_facet_area.host()[sf] = h_subfacet_parea(sf);

        for (int k = 0; k < 3; ++k) {
            da_edge_center.host()[sf * 3 + k] = h_subfacet_centroid(sf * 3 + k);
            da_edge_n.host()[sf * 3 + k] = h_subfacet_normal(sf * 3 + k);
            da_edge_m.host()[sf * 3 + k] = h_subfacet_tangent_q(sf * 3 + k);
            da_edge_lv.host()[sf * 3 + k] = h_subfacet_tangent_s(sf * 3 + k);
        }

        da_edge_tet_offsets.host()[sf] = sf;
        da_edge_tet_indices.host()[sf] = t;
        da_subfacet_edge_idx.host()[sf] = sf;
    }
    for (int t = 0; t < n_elem; ++t) {
        if (tet_facet_count[static_cast<size_t>(t)] != 12) {
            MOPHI_ERROR("SetupFromMesh: TET %d has %d sub-facets; Chrono-compatible LDPM requires 12.", t,
                        tet_facet_count[static_cast<size_t>(t)]);
            return;
        }
    }
    da_edge_tet_offsets.host()[n_edge] = n_edge;

    da_edge_nodes.ToDevice();
    da_l0.ToDevice();
    da_facet_area.ToDevice();
    da_edge_n.ToDevice();
    da_edge_m.ToDevice();
    da_edge_lv.ToDevice();
    da_edge_center.ToDevice();
    da_edge_tet_offsets.ToDevice();
    da_edge_tet_indices.ToDevice();
    da_subfacet_edge_idx.ToDevice();

    da_facet_t.SetVal(Real(0));
    da_facet_t.ToDevice();
    da_kappa.SetVal(Real(0));
    da_kappa.ToDevice();
    da_omega.SetVal(Real(0));
    da_omega.ToDevice();
    da_edge_vol_strain.SetVal(Real(0));
    da_edge_vol_strain.ToDevice();

    // Sync device struct mirror so kernels see resized arrays and interaction count.
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPMTet4_Data), cudaMemcpyHostToDevice));

    MOPHI_INFO("SetupFromMesh: configured %d LDPM sub-facet interactions.", n_edge);
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

void GPU_LDPMTet4_Data::SetLDPMParams(const LDPMParams& params) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPMTet4_Data must be set up before setting LDPM parameters.");
        return;
    }

    // Allocate per-interaction state vector if not already done
    if (d_statev == nullptr) {
        da_statev.resize(static_cast<size_t>(n_edge) * LDPM_N_STATEV);
        da_statev.BindDevicePointer(&d_statev);
        da_statev.SetVal(Real(0));
        da_statev.ToDevice();
    }

    // Allocate and copy params struct to device
    if (d_ldpm_params == nullptr) {
        MOPHI_GPU_CALL(cudaMalloc(&d_ldpm_params, sizeof(LDPMParams)));
    }
    MOPHI_GPU_CALL(cudaMemcpy(d_ldpm_params, &params, sizeof(LDPMParams), cudaMemcpyHostToDevice));

    // Also set the material parameters used by the force assembly
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &params.E0, sizeof(Real), cudaMemcpyHostToDevice));
    Real E_T_val = params.alpha * params.E0;
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &E_T_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kT, &params.E_kT, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kM, &params.E_kM, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_kL, &params.E_kL, sizeof(Real), cudaMemcpyHostToDevice));

    h_rho = params.rho;
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &params.rho, sizeof(Real), cudaMemcpyHostToDevice));

    // Update device mirror
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

    const Real* hx = da_x_cur.host();
    const Real* hy = da_y_cur.host();
    const Real* hz = da_z_cur.host();

    // ── Voronoi volumes → translational lumped mass ───────────────────────────
    std::vector<Real> node_volumes(n_nodes, Real(0));

    for (int e = 0; e < n_elem; e++) {
        const int n0 = h_tet_conn_vec[e * 4 + 0];
        const int n1 = h_tet_conn_vec[e * 4 + 1];
        const int n2 = h_tet_conn_vec[e * 4 + 2];
        const int n3 = h_tet_conn_vec[e * 4 + 3];

        const Real vol = ldpm_host_tet_abs_volume(hx, hy, hz, n0, n1, n2, n3);

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
    MOPHI_GPU_CALL(cudaMemcpy(d_csr_columns, h_columns.data(), static_cast<size_t>(n_nodes) * sizeof(int),
                              cudaMemcpyHostToDevice));
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
    if (d_ldpm_params == nullptr || d_statev == nullptr) {
        MOPHI_ERROR("GPU_LDPMTet4_Data::CalcP requires SetLDPMParams() before evaluating LDPM facet response.");
        return;
    }
    constexpr int threads = 256;

    ComputeVolumetricStrain();

    // Compute facet strains, constitutive response, and tractions.
    const int blocks = (n_edge + threads - 1) / threads;
    calc_p_ldpm_tet4_kernel<<<blocks, threads>>>(d_data);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── ComputeVolumetricStrain ──────────────────────────────────────────────────

void GPU_LDPMTet4_Data::ComputeVolumetricStrain() {
    if (!is_setup)
        return;
    if (n_elem <= 0 || d_tet_conn == nullptr)
        return;
    constexpr int threads = 256;

    const int blocks_tet = (n_elem + threads - 1) / threads;
    compute_tet_volumetric_strain_kernel<<<blocks_tet, threads>>>(n_elem, d_tet_conn, d_x_cur, d_y_cur, d_z_cur,
                                                                  d_tet_vol_ref, d_tet_vol_strain);

    const int blocks_edge = (n_edge + threads - 1) / threads;
    average_tet_volumetric_strain_to_edges_kernel<<<blocks_edge, threads>>>(
        n_edge, d_edge_tet_offsets, d_edge_tet_indices, d_tet_vol_strain, d_edge_vol_strain);
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
    da_x_cur.ToHost();
    da_y_cur.ToHost();
    da_z_cur.ToHost();
    std::copy(da_x_cur.host(), da_x_cur.host() + n_nodes, x_out.data());
    std::copy(da_y_cur.host(), da_y_cur.host() + n_nodes, y_out.data());
    std::copy(da_z_cur.host(), da_z_cur.host() + n_nodes, z_out.data());
}

void GPU_LDPMTet4_Data::RetrieveRotationToCPU(VectorXR& rx_out, VectorXR& ry_out, VectorXR& rz_out) {
    rx_out.resize(n_nodes);
    ry_out.resize(n_nodes);
    rz_out.resize(n_nodes);
    da_rot_x_cur.ToHost();
    da_rot_y_cur.ToHost();
    da_rot_z_cur.ToHost();
    std::copy(da_rot_x_cur.host(), da_rot_x_cur.host() + n_nodes, rx_out.data());
    std::copy(da_rot_y_cur.host(), da_rot_y_cur.host() + n_nodes, ry_out.data());
    std::copy(da_rot_z_cur.host(), da_rot_z_cur.host() + n_nodes, rz_out.data());
}

void GPU_LDPMTet4_Data::RetrieveInternalForceToCPU(VectorXR& f_out) {
    f_out.resize(n_nodes * 3);
    da_f_int_t.ToHost();
    std::copy(da_f_int_t.host(), da_f_int_t.host() + n_nodes * 3, f_out.data());
}

void GPU_LDPMTet4_Data::RetrieveInternalMomentToCPU(VectorXR& m_out) {
    m_out.resize(n_nodes * 3);
    da_f_int_r.ToHost();
    std::copy(da_f_int_r.host(), da_f_int_r.host() + n_nodes * 3, m_out.data());
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

void GPU_LDPMTet4_Data::RetrieveFacetTractionToCPU(VectorXR& out) {
    out.resize(n_edge * LDPM_FACET_N_COMPONENTS);
    da_facet_t.ToHost();
    std::copy(da_facet_t.host(), da_facet_t.host() + n_edge * LDPM_FACET_N_COMPONENTS, out.data());
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
    advance_driven_nodes_z_kernel<<<blocks, threads>>>(d_z_cur, d_driven_idx, n_driven, dz);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── Destroy ─────────────────────────────────────────────────────────────────

void GPU_LDPMTet4_Data::Destroy() {
    da_x_cur.free();
    da_y_cur.free();
    da_z_cur.free();
    da_x_ref.free();
    da_y_ref.free();
    da_z_ref.free();
    da_rot_x_cur.free();
    da_rot_y_cur.free();
    da_rot_z_cur.free();
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
    da_edge_center.free();
    da_facet_t.free();
    da_kappa.free();
    da_omega.free();
    da_statev.free();
    da_tet_conn.free();
    da_tet_vol_ref.free();
    da_tet_vol_strain.free();
    da_edge_tet_offsets.free();
    da_edge_tet_indices.free();
    da_edge_vol_strain.free();
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
    if (d_ldpm_params != nullptr) {
        MOPHI_GPU_CALL(cudaFree(d_ldpm_params));
    }
    MOPHI_GPU_CALL(cudaFree(d_data));
}

}  // namespace feris
