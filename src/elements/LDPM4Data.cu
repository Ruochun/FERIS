/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    LDPM4Data.cu
 * Brief:   Implements GPU-side data management for the 3-DOF translational
 *          LDPM element.  Host methods handle:
 *            - Edge extraction from TET4 connectivity
 *            - Reference geometry (l0, n, m, l_vec, facet area)
 *            - Diagonal CSR mass matrix construction
 *            - Fixed-node BC setup
 *          GPU kernels handle:
 *            - CalcP  (wrapper: compute facet tractions at current positions)
 *            - CalcInternalForce  (clear + assemble f_int from facet tractions)
 *==============================================================
 *==============================================================*/

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../types.h"
#include "../utils/cuda_utils.h"
#include "LDPM4Data.cuh"
#include "LDPM4DataFunc.cuh"
#include <MoPhiEssentials.h>

namespace tlfea {

// ─── GPU kernel wrappers ─────────────────────────────────────────────────────

__global__ void calc_p_ldpm4_kernel(GPU_LDPM4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_data->gpu_n_edge())
        return;
    compute_p(tid, 0, d_data, nullptr, Real(0));
}

__global__ void clear_f_int_ldpm4_kernel(GPU_LDPM4_Data* d_data) {
    clear_internal_force(d_data);
}

__global__ void compute_f_int_ldpm4_kernel(GPU_LDPM4_Data* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Two endpoint nodes per edge
    const int total = d_data->gpu_n_edge() * 2;
    if (tid >= total)
        return;
    const int edge_idx = tid / 2;
    const int node_local = tid % 2;
    compute_internal_force(edge_idx, node_local, d_data);
}

// ─── Host-side helper ────────────────────────────────────────────────────────

// Compute the area of the triangle spanned by three 3-D points.
static inline Real triangle_area(const Real* a, const Real* b, const Real* c) {
    Real v1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    Real v2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    Real cross[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
                     v1[0] * v2[1] - v1[1] * v2[0]};
    return Real(0.5) * std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
}

// ─── Setup ────────────────────────────────────────────────────────────────────

void GPU_LDPM4_Data::Setup(const VectorXR& h_x,
                           const VectorXR& h_y,
                           const VectorXR& h_z,
                           const MatrixXi& tet_connectivity) {
    if (is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data is already set up.");
        return;
    }
    if (h_x.size() != n_coef || h_y.size() != n_coef || h_z.size() != n_coef) {
        MOPHI_ERROR("Position vector size mismatch in GPU_LDPM4_Data::Setup.");
        return;
    }
    if (tet_connectivity.rows() != n_elem || tet_connectivity.cols() != 4) {
        MOPHI_ERROR("TET connectivity matrix has wrong dimensions.");
        return;
    }

    // ── 1. Allocate and fill nodal position arrays ───────────────────────────

    da_x12.resize(n_coef);
    da_x12.BindDevicePointer(&d_x12);
    da_y12.resize(n_coef);
    da_y12.BindDevicePointer(&d_y12);
    da_z12.resize(n_coef);
    da_z12.BindDevicePointer(&d_z12);

    std::copy(h_x.data(), h_x.data() + n_coef, da_x12.host());
    da_x12.ToDevice();
    std::copy(h_y.data(), h_y.data() + n_coef, da_y12.host());
    da_y12.ToDevice();
    std::copy(h_z.data(), h_z.data() + n_coef, da_z12.host());
    da_z12.ToDevice();

    da_f_int.resize(n_coef * 3);
    da_f_int.BindDevicePointer(&d_f_int);
    da_f_int.SetVal(Real(0));
    da_f_int.MakeReadyDevice();

    da_f_ext.resize(n_coef * 3);
    da_f_ext.BindDevicePointer(&d_f_ext);
    da_f_ext.SetVal(Real(0));
    da_f_ext.MakeReadyDevice();

    // ── 2. Store TET connectivity on host for CalcMassMatrix ─────────────────

    h_tet_conn_vec.resize(n_elem * 4);
    for (int e = 0; e < n_elem; e++) {
        for (int k = 0; k < 4; k++) {
            h_tet_conn_vec[e * 4 + k] = tet_connectivity(e, k);
        }
    }

    // ── 3. Extract unique edges ───────────────────────────────────────────────

    // The 6 local edge pairs for a TET4 element.
    static const int LOCAL_EDGES[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    // edge_to_tets[{i,j}] = list of tet indices containing edge (i,j), i<j
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

    // ── 4. Compute edge geometry and facet areas ──────────────────────────────

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

        // Reference edge vector and length
        Real dx = h_x(nj) - h_x(ni);
        Real dy = h_y(nj) - h_y(ni);
        Real dz = h_z(nj) - h_z(ni);
        const Real l0_val = std::sqrt(dx * dx + dy * dy + dz * dz);
        h_l0[edge_idx] = l0_val;

        // Unit normal along edge (reference)
        const Real inv_l0 = (l0_val > Real(0)) ? Real(1) / l0_val : Real(0);
        const Real nx = dx * inv_l0;
        const Real ny = dy * inv_l0;
        const Real nz = dz * inv_l0;
        h_edge_n[edge_idx * 3 + 0] = nx;
        h_edge_n[edge_idx * 3 + 1] = ny;
        h_edge_n[edge_idx * 3 + 2] = nz;

        // Orthonormal shear basis (Gram-Schmidt with a reference vector)
        // Choose reference vector most orthogonal to n to avoid near-zero cross.
        Real mx, my, mz;
        if (std::abs(nx) <= std::abs(ny) && std::abs(nx) <= std::abs(nz)) {
            // |nx| is smallest: cross(n, e_x) = (0, nz, -ny)
            mx = Real(0);
            my = nz;
            mz = -ny;
        } else if (std::abs(ny) <= std::abs(nz)) {
            // |ny| is smallest: cross(n, e_y) = (-nz, 0, nx)
            mx = -nz;
            my = Real(0);
            mz = nx;
        } else {
            // |nz| is smallest: cross(n, e_z) = (ny, -nx, 0)
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

        // l_vec = n × m  (already unit since n and m are orthogonal unit vectors)
        const Real lvx = ny * mz - nz * my;
        const Real lvy = nz * mx - nx * mz;
        const Real lvz = nx * my - ny * mx;
        h_edge_lv[edge_idx * 3 + 0] = lvx;
        h_edge_lv[edge_idx * 3 + 1] = lvy;
        h_edge_lv[edge_idx * 3 + 2] = lvz;

        // Voronoi facet area: for each tet sharing this edge, the two
        // triangular sub-facets connecting:
        //   midpoint(i,j) — centroid(face ijk) — centroid(tet)
        //   midpoint(i,j) — centroid(face ijl) — centroid(tet)
        Real area_sum = Real(0);
        const Real xi = h_x(ni), yi = h_y(ni), zi = h_z(ni);
        const Real xj = h_x(nj), yj = h_y(nj), zj = h_z(nj);
        const Real mid[3] = {Real(0.5) * (xi + xj), Real(0.5) * (yi + yj), Real(0.5) * (zi + zj)};

        for (int t : tets) {
            // Find the two "other" nodes in this tet (not ni or nj)
            int others[4];
            int n_others = 0;
            for (int k = 0; k < 4; k++) {
                int node = h_tet_conn_vec[t * 4 + k];
                if (node != ni && node != nj) {
                    others[n_others++] = node;
                }
            }
            // n_others should be exactly 2 for a valid TET4
            if (n_others != 2)
                continue;

            const int nk = others[0];
            const int nl = others[1];
            const Real xk = h_x(nk), yk = h_y(nk), zk = h_z(nk);
            const Real xl = h_x(nl), yl = h_y(nl), zl = h_z(nl);

            // Centroid of face (ni, nj, nk)
            const Real c_ijk[3] = {(xi + xj + xk) / Real(3), (yi + yj + yk) / Real(3),
                                   (zi + zj + zk) / Real(3)};

            // Centroid of face (ni, nj, nl)
            const Real c_ijl[3] = {(xi + xj + xl) / Real(3), (yi + yj + yl) / Real(3),
                                   (zi + zj + zl) / Real(3)};

            // Centroid of tetrahedron
            const Real c_tet[3] = {(xi + xj + xk + xl) / Real(4), (yi + yj + yk + yl) / Real(4),
                                   (zi + zj + zk + zl) / Real(4)};

            area_sum += triangle_area(mid, c_ijk, c_tet);
            area_sum += triangle_area(mid, c_ijl, c_tet);
        }
        h_facet_area[edge_idx] = area_sum;

        ++edge_idx;
    }

    // ── 5. Allocate and copy edge data to device ──────────────────────────────

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

    da_facet_t.resize(static_cast<size_t>(n_edge) * 3);
    da_facet_t.BindDevicePointer(&d_facet_t);
    da_facet_t.SetVal(Real(0));
    da_facet_t.MakeReadyDevice();

    // ── 6. Allocate device material scalars with safe defaults ────────────────

    MOPHI_GPU_CALL(cudaMalloc(&d_E_N, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_E_T, sizeof(Real)));
    MOPHI_GPU_CALL(cudaMalloc(&d_rho, sizeof(Real)));

    Real default_val = Real(0);
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &default_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &default_val, sizeof(Real), cudaMemcpyHostToDevice));

    // ── 7. Allocate device-side struct mirror ─────────────────────────────────

    MOPHI_GPU_CALL(cudaMalloc(&d_data, sizeof(GPU_LDPM4_Data)));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPM4_Data), cudaMemcpyHostToDevice));

    is_setup = true;

    MOPHI_INFO("GPU_LDPM4_Data: %d nodes, %d TET4 elements, %d unique edges", n_coef, n_elem, n_edge);
}

// ─── Material / parameter setters ────────────────────────────────────────────

void GPU_LDPM4_Data::SetMaterial(Real E_N_val, Real E_T_val) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data must be set up before setting material.");
        return;
    }
    MOPHI_GPU_CALL(cudaMemcpy(d_E_N, &E_N_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_E_T, &E_T_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPM4_Data), cudaMemcpyHostToDevice));
}

void GPU_LDPM4_Data::SetDensity(Real rho_val) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data must be set up before setting density.");
        return;
    }
    h_rho = rho_val;
    MOPHI_GPU_CALL(cudaMemcpy(d_rho, &rho_val, sizeof(Real), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPM4_Data), cudaMemcpyHostToDevice));
}

void GPU_LDPM4_Data::SetExternalForce(const VectorXR& h_f_ext) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data must be set up before setting external force.");
        return;
    }
    if (h_f_ext.size() != n_coef * 3) {
        MOPHI_ERROR("External force vector size mismatch.");
        return;
    }
    std::copy(h_f_ext.data(), h_f_ext.data() + n_coef * 3, da_f_ext.host());
    da_f_ext.ToDevice();
}

void GPU_LDPM4_Data::SetNodalFixed(const VectorXi& fixed_nodes_in) {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data must be set up before setting fixed nodes.");
        return;
    }
    n_constraint = fixed_nodes_in.size() * 3;

    da_fixed_nodes.resize(static_cast<size_t>(fixed_nodes_in.size()));
    da_fixed_nodes.BindDevicePointer(&d_fixed_nodes);
    std::copy(fixed_nodes_in.data(), fixed_nodes_in.data() + fixed_nodes_in.size(), da_fixed_nodes.host());
    da_fixed_nodes.ToDevice();

    is_constraints_setup = true;
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPM4_Data), cudaMemcpyHostToDevice));
}

// ─── CalcMassMatrix ──────────────────────────────────────────────────────────

void GPU_LDPM4_Data::CalcMassMatrix() {
    if (!is_setup) {
        MOPHI_ERROR("GPU_LDPM4_Data must be set up before CalcMassMatrix.");
        return;
    }

    // ── Compute Voronoi volumes on host ───────────────────────────────────────
    // Each TET4 contributes V_tet / 4 to each of its four nodes (equal share).
    const Real* hx = da_x12.host();
    const Real* hy = da_y12.host();
    const Real* hz = da_z12.host();

    std::vector<Real> node_volumes(n_coef, Real(0));

    for (int e = 0; e < n_elem; e++) {
        const int n0 = h_tet_conn_vec[e * 4 + 0];
        const int n1 = h_tet_conn_vec[e * 4 + 1];
        const int n2 = h_tet_conn_vec[e * 4 + 2];
        const int n3 = h_tet_conn_vec[e * 4 + 3];

        // Edge vectors from node 0
        const Real a[3] = {hx[n1] - hx[n0], hy[n1] - hy[n0], hz[n1] - hz[n0]};
        const Real b[3] = {hx[n2] - hx[n0], hy[n2] - hy[n0], hz[n2] - hz[n0]};
        const Real c[3] = {hx[n3] - hx[n0], hy[n3] - hy[n0], hz[n3] - hz[n0]};

        // V_tet = |a · (b × c)| / 6
        const Real cross_bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2],
                                   b[0] * c[1] - b[1] * c[0]};
        const Real vol = std::abs(a[0] * cross_bc[0] + a[1] * cross_bc[1] + a[2] * cross_bc[2]) / Real(6);

        node_volumes[n0] += vol / Real(4);
        node_volumes[n1] += vol / Real(4);
        node_volumes[n2] += vol / Real(4);
        node_volumes[n3] += vol / Real(4);
    }

    // ── Build diagonal CSR ────────────────────────────────────────────────────
    std::vector<int> h_offsets(n_coef + 1);
    std::vector<int> h_columns(n_coef);
    std::vector<Real> h_values(n_coef);

    for (int i = 0; i < n_coef; i++) {
        h_offsets[i] = i;
        h_columns[i] = i;
        h_values[i] = h_rho * node_volumes[i];
    }
    h_offsets[n_coef] = n_coef;

    // ── Allocate and copy to device ───────────────────────────────────────────
    if (!is_csr_setup) {
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_offsets, static_cast<size_t>(n_coef + 1) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_columns, static_cast<size_t>(n_coef) * sizeof(int)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_csr_values, static_cast<size_t>(n_coef) * sizeof(Real)));
        MOPHI_GPU_CALL(cudaMalloc((void**)&d_nnz, sizeof(int)));
        is_csr_setup = true;
    }

    MOPHI_GPU_CALL(
        cudaMemcpy(d_csr_offsets, h_offsets.data(), static_cast<size_t>(n_coef + 1) * sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(
        cudaMemcpy(d_csr_columns, h_columns.data(), static_cast<size_t>(n_coef) * sizeof(int), cudaMemcpyHostToDevice));
    MOPHI_GPU_CALL(
        cudaMemcpy(d_csr_values, h_values.data(), static_cast<size_t>(n_coef) * sizeof(Real), cudaMemcpyHostToDevice));

    const int nnz_val = n_coef;
    MOPHI_GPU_CALL(cudaMemcpy(d_nnz, &nnz_val, sizeof(int), cudaMemcpyHostToDevice));

    // Refresh device-side struct mirror to expose the new CSR pointers
    MOPHI_GPU_CALL(cudaMemcpy(d_data, this, sizeof(GPU_LDPM4_Data), cudaMemcpyHostToDevice));
}

// ─── CalcP ────────────────────────────────────────────────────────────────────

void GPU_LDPM4_Data::CalcP() {
    if (!is_setup)
        return;
    if (n_edge == 0)
        return;
    constexpr int threads = 256;
    const int blocks = (n_edge + threads - 1) / threads;
    calc_p_ldpm4_kernel<<<blocks, threads>>>(d_data);
    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── CalcInternalForce ────────────────────────────────────────────────────────

void GPU_LDPM4_Data::CalcInternalForce() {
    if (!is_setup)
        return;
    if (n_edge == 0)
        return;
    constexpr int threads = 256;

    // Clear f_int (n_coef * 3 DOFs)
    const int blocks_clear = (n_coef * 3 + threads - 1) / threads;
    clear_f_int_ldpm4_kernel<<<blocks_clear, threads>>>(d_data);

    // Assemble from edges (n_edge * 2 threads)
    const int blocks_assemble = (n_edge * 2 + threads - 1) / threads;
    compute_f_int_ldpm4_kernel<<<blocks_assemble, threads>>>(d_data);

    MOPHI_GPU_CALL(cudaDeviceSynchronize());
}

// ─── Retrieve helpers ─────────────────────────────────────────────────────────

void GPU_LDPM4_Data::RetrievePositionToCPU(VectorXR& x_out, VectorXR& y_out, VectorXR& z_out) {
    x_out.resize(n_coef);
    y_out.resize(n_coef);
    z_out.resize(n_coef);
    da_x12.ToHost();
    da_y12.ToHost();
    da_z12.ToHost();
    std::copy(da_x12.host(), da_x12.host() + n_coef, x_out.data());
    std::copy(da_y12.host(), da_y12.host() + n_coef, y_out.data());
    std::copy(da_z12.host(), da_z12.host() + n_coef, z_out.data());
}

void GPU_LDPM4_Data::RetrieveInternalForceToCPU(VectorXR& f_out) {
    f_out.resize(n_coef * 3);
    da_f_int.ToHost();
    std::copy(da_f_int.host(), da_f_int.host() + n_coef * 3, f_out.data());
}

void GPU_LDPM4_Data::RetrieveExternalForceToCPU(VectorXR& f_out) {
    f_out.resize(n_coef * 3);
    da_f_ext.ToHost();
    std::copy(da_f_ext.host(), da_f_ext.host() + n_coef * 3, f_out.data());
}

// ─── Destroy ─────────────────────────────────────────────────────────────────

void GPU_LDPM4_Data::Destroy() {
    da_x12.free();
    da_y12.free();
    da_z12.free();
    da_f_int.free();
    da_f_ext.free();

    if (n_edge > 0) {
        da_edge_nodes.free();
        da_l0.free();
        da_facet_area.free();
        da_edge_n.free();
        da_edge_m.free();
        da_edge_lv.free();
        da_facet_t.free();
    }

    if (is_constraints_setup) {
        da_fixed_nodes.free();
    }

    if (is_csr_setup) {
        MOPHI_GPU_CALL(cudaFree(d_csr_offsets));
        MOPHI_GPU_CALL(cudaFree(d_csr_columns));
        MOPHI_GPU_CALL(cudaFree(d_csr_values));
        MOPHI_GPU_CALL(cudaFree(d_nnz));
    }

    MOPHI_GPU_CALL(cudaFree(d_E_N));
    MOPHI_GPU_CALL(cudaFree(d_E_T));
    MOPHI_GPU_CALL(cudaFree(d_rho));
    MOPHI_GPU_CALL(cudaFree(d_data));

    h_tet_conn_vec.clear();
    is_setup = false;
    is_csr_setup = false;
    is_constraints_setup = false;
}

}  // namespace tlfea
