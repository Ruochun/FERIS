/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    ldpm_mesh_utils.cc
 * Brief:   Implements the six LDPM mesh-file readers declared in
 *          ldpm_mesh_utils.h.
 *
 *  All six Chrono Workbench output files use the same comment
 *  convention: lines whose first non-whitespace characters are
 *  "//" are ignored, as are blank lines.  No row-count header
 *  is present; each reader accumulates rows until EOF.
 *==============================================================
 *==============================================================*/

#include "ldpm_mesh_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace feris {

// ============================================================
// Internal helpers
// ============================================================

namespace {

// Returns true if the (already-trimmed) line is a comment or blank.
static bool IsSkippableLine(const std::string& line) {
    if (line.empty())
        return true;
    // Comment lines start with "//"
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/')
        return true;
    return false;
}

// Trim leading/trailing whitespace in-place.
static void TrimLDPM(std::string& s) {
    const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// Read the next non-blank, non-comment line from file.
// Returns false when EOF is reached.
static bool NextDataLine(std::ifstream& file, std::string& out) {
    std::string line;
    while (std::getline(file, line)) {
        TrimLDPM(line);
        if (!IsSkippableLine(line)) {
            out = line;
            return true;
        }
    }
    return false;
}

// Split a whitespace-delimited string into tokens.
static std::vector<std::string> TokenizeLDPM(const std::string& s) {
    std::vector<std::string> toks;
    std::istringstream iss(s);
    std::string w;
    while (iss >> w)
        toks.push_back(w);
    return toks;
}

static void SetErr(std::string* err, const std::string& msg) {
    if (err)
        *err = msg;
}

// Parse a double from a string; return false on failure.
static bool ParseReal(const std::string& s, Real& out) {
    try {
        size_t idx = 0;
        out = std::stod(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

// Parse an int from a string; return false on failure.
static bool ParseInt(const std::string& s, int& out) {
    try {
        size_t idx = 0;
        out = std::stoi(s, &idx);
        return idx == s.size();
    } catch (...) {
        return false;
    }
}

}  // namespace

// ============================================================
// ReadLDPMTet4NodesFile
//
// Format (one row per node, no index column):
//   X  Y  Z
// ============================================================
bool ReadLDPMTet4NodesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_nodes = 0;
    out.node_x = VectorXR{};
    out.node_y = VectorXR{};
    out.node_z = VectorXR{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4NodesFile: cannot open " + path);
        return false;
    }

    std::vector<Real> xs, ys, zs;
    std::string line;
    int line_num = 0;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 3) {
            SetErr(error, "ReadLDPMTet4NodesFile: expected 3 tokens at data line " + std::to_string(line_num) + ": \"" +
                              line + "\"");
            return false;
        }
        Real x, y, z;
        if (!ParseReal(t[0], x) || !ParseReal(t[1], y) || !ParseReal(t[2], z)) {
            SetErr(error, "ReadLDPMTet4NodesFile: parse error at data line " + std::to_string(line_num));
            return false;
        }
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
    }

    const int n = static_cast<int>(xs.size());
    out.n_nodes = n;
    out.node_x.resize(n);
    out.node_y.resize(n);
    out.node_z.resize(n);
    for (int i = 0; i < n; ++i) {
        out.node_x(i) = xs[i];
        out.node_y(i) = ys[i];
        out.node_z(i) = zs[i];
    }
    return true;
}

// ============================================================
// ReadLDPMTet4ParticlesFile
//
// Format (one row per particle):
//   n  x  y  z  d
//   n is the zero-indexed particle ID; d is the aggregate diameter.
// ============================================================
bool ReadLDPMTet4ParticlesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_particles = 0;
    out.particle_x = VectorXR{};
    out.particle_y = VectorXR{};
    out.particle_z = VectorXR{};
    out.particle_d = VectorXR{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4ParticlesFile: cannot open " + path);
        return false;
    }

    // First pass: collect rows. Each row has 5 tokens: n x y z d.
    // Rows arrive in order of particle index but we track the max index
    // to size the output arrays correctly.
    struct Row {
        int n;
        Real x, y, z, d;
    };
    std::vector<Row> rows;

    std::string line;
    int line_num = 0;
    int max_id = -1;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 5) {
            SetErr(error, "ReadLDPMTet4ParticlesFile: expected 5 tokens at data line " + std::to_string(line_num) +
                              ": \"" + line + "\"");
            return false;
        }
        Row r{};
        if (!ParseInt(t[0], r.n) || r.n < 0) {
            SetErr(error, "ReadLDPMTet4ParticlesFile: invalid particle index at data line " + std::to_string(line_num));
            return false;
        }
        if (!ParseReal(t[1], r.x) || !ParseReal(t[2], r.y) || !ParseReal(t[3], r.z) || !ParseReal(t[4], r.d)) {
            SetErr(error, "ReadLDPMTet4ParticlesFile: parse error at data line " + std::to_string(line_num));
            return false;
        }
        max_id = std::max(max_id, r.n);
        rows.push_back(r);
    }

    if (rows.empty()) {
        SetErr(error, "ReadLDPMTet4ParticlesFile: no data rows found in " + path);
        return false;
    }

    const int n = max_id + 1;
    out.n_particles = n;
    out.particle_x = VectorXR::Zero(n);
    out.particle_y = VectorXR::Zero(n);
    out.particle_z = VectorXR::Zero(n);
    out.particle_d = VectorXR::Zero(n);

    for (const auto& r : rows) {
        out.particle_x(r.n) = r.x;
        out.particle_y(r.n) = r.y;
        out.particle_z(r.n) = r.z;
        out.particle_d(r.n) = r.d;
    }
    return true;
}

// ============================================================
// ReadLDPMTet4TetsFile
//
// Format (one row per TET element, zero-indexed):
//   Node1  Node2  Node3  Node4
// ============================================================
bool ReadLDPMTet4TetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_tets = 0;
    out.tet_connectivity = MatrixXi{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4TetsFile: cannot open " + path);
        return false;
    }

    std::vector<std::array<int, 4>> rows;
    std::string line;
    int line_num = 0;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 4) {
            SetErr(error, "ReadLDPMTet4TetsFile: expected 4 tokens at data line " + std::to_string(line_num) + ": \"" +
                              line + "\"");
            return false;
        }
        std::array<int, 4> r{};
        for (int k = 0; k < 4; ++k) {
            if (!ParseInt(t[k], r[k]) || r[k] < 0) {
                SetErr(error, "ReadLDPMTet4TetsFile: invalid node index at data line " + std::to_string(line_num));
                return false;
            }
        }
        rows.push_back(r);
    }

    const int n = static_cast<int>(rows.size());
    out.n_tets = n;
    out.tet_connectivity.resize(n, 4);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k)
            out.tet_connectivity(i, k) = rows[i][k];
    }
    return true;
}

// ============================================================
// ReadLDPMTet4FacetsFile
//
// Format (one row per Voronoi sub-facet, 19 tokens):
//   Tet  IDx  IDy  IDz  Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  mF
// ============================================================
bool ReadLDPMTet4FacetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_subfacets = 0;
    out.subfacet_tet = VectorXi{};
    out.subfacet_vertex_ids = VectorXi{};
    out.subfacet_vol = VectorXR{};
    out.subfacet_parea = VectorXR{};
    out.subfacet_centroid = VectorXR{};
    out.subfacet_normal = VectorXR{};
    out.subfacet_tangent_q = VectorXR{};
    out.subfacet_tangent_s = VectorXR{};
    out.subfacet_matflag = VectorXi{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4FacetsFile: cannot open " + path);
        return false;
    }

    // Temporary storage — reserve a generous initial capacity for performance.
    std::vector<int> v_tet, v_vid;  // v_vid stores 3 ints per row
    std::vector<Real> v_vol, v_area;
    std::vector<Real> v_c;            // 3 per row
    std::vector<Real> v_p, v_q, v_s;  // 3 per row each
    std::vector<int> v_mf;

    constexpr size_t kReserve = 70000;
    v_tet.reserve(kReserve);
    v_vid.reserve(kReserve * 3);
    v_vol.reserve(kReserve);
    v_area.reserve(kReserve);
    v_c.reserve(kReserve * 3);
    v_p.reserve(kReserve * 3);
    v_q.reserve(kReserve * 3);
    v_s.reserve(kReserve * 3);
    v_mf.reserve(kReserve);

    std::string line;
    int line_num = 0;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 19) {
            SetErr(error, "ReadLDPMTet4FacetsFile: expected 19 tokens at data line " + std::to_string(line_num) +
                              " (got " + std::to_string(t.size()) + ")");
            return false;
        }

        // Tet  IDx  IDy  IDz  (ints, cols 0-3)
        int tet_idx, idx, idy, idz;
        if (!ParseInt(t[0], tet_idx) || tet_idx < 0 || !ParseInt(t[1], idx) || idx < 0 || !ParseInt(t[2], idy) ||
            idy < 0 || !ParseInt(t[3], idz) || idz < 0) {
            SetErr(error, "ReadLDPMTet4FacetsFile: invalid integer at data line " + std::to_string(line_num));
            return false;
        }

        // Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  (Reals, cols 4-17)
        Real vals[14];
        for (int k = 0; k < 14; ++k) {
            if (!ParseReal(t[4 + k], vals[k])) {
                SetErr(error, "ReadLDPMTet4FacetsFile: parse error (col " + std::to_string(4 + k) + ") at data line " +
                                  std::to_string(line_num));
                return false;
            }
        }

        // mF  (int, col 18)
        int mf;
        if (!ParseInt(t[18], mf)) {
            SetErr(error, "ReadLDPMTet4FacetsFile: invalid mF at data line " + std::to_string(line_num));
            return false;
        }

        v_tet.push_back(tet_idx);
        v_vid.push_back(idx);
        v_vid.push_back(idy);
        v_vid.push_back(idz);
        v_vol.push_back(vals[0]);   // Vol
        v_area.push_back(vals[1]);  // pArea
        v_c.push_back(vals[2]);     // cx
        v_c.push_back(vals[3]);     // cy
        v_c.push_back(vals[4]);     // cz
        v_p.push_back(vals[5]);     // px
        v_p.push_back(vals[6]);     // py
        v_p.push_back(vals[7]);     // pz
        v_q.push_back(vals[8]);     // qx
        v_q.push_back(vals[9]);     // qy
        v_q.push_back(vals[10]);    // qz
        v_s.push_back(vals[11]);    // sx
        v_s.push_back(vals[12]);    // sy
        v_s.push_back(vals[13]);    // sz
        v_mf.push_back(mf);
    }

    const int n = static_cast<int>(v_tet.size());
    out.n_subfacets = n;

    out.subfacet_tet.resize(n);
    out.subfacet_vertex_ids.resize(3 * n);
    out.subfacet_vol.resize(n);
    out.subfacet_parea.resize(n);
    out.subfacet_centroid.resize(3 * n);
    out.subfacet_normal.resize(3 * n);
    out.subfacet_tangent_q.resize(3 * n);
    out.subfacet_tangent_s.resize(3 * n);
    out.subfacet_matflag.resize(n);

    for (int i = 0; i < n; ++i) {
        out.subfacet_tet(i) = v_tet[i];
        out.subfacet_vertex_ids(3 * i + 0) = v_vid[3 * i + 0];
        out.subfacet_vertex_ids(3 * i + 1) = v_vid[3 * i + 1];
        out.subfacet_vertex_ids(3 * i + 2) = v_vid[3 * i + 2];
        out.subfacet_vol(i) = v_vol[i];
        out.subfacet_parea(i) = v_area[i];
        out.subfacet_centroid(3 * i + 0) = v_c[3 * i + 0];
        out.subfacet_centroid(3 * i + 1) = v_c[3 * i + 1];
        out.subfacet_centroid(3 * i + 2) = v_c[3 * i + 2];
        out.subfacet_normal(3 * i + 0) = v_p[3 * i + 0];
        out.subfacet_normal(3 * i + 1) = v_p[3 * i + 1];
        out.subfacet_normal(3 * i + 2) = v_p[3 * i + 2];
        out.subfacet_tangent_q(3 * i + 0) = v_q[3 * i + 0];
        out.subfacet_tangent_q(3 * i + 1) = v_q[3 * i + 1];
        out.subfacet_tangent_q(3 * i + 2) = v_q[3 * i + 2];
        out.subfacet_tangent_s(3 * i + 0) = v_s[3 * i + 0];
        out.subfacet_tangent_s(3 * i + 1) = v_s[3 * i + 1];
        out.subfacet_tangent_s(3 * i + 2) = v_s[3 * i + 2];
        out.subfacet_matflag(i) = v_mf[i];
    }
    return true;
}

// ============================================================
// ReadLDPMTet4FaceFacetsFile
//
// Format (one row per surface triangle, 10 tokens):
//   n  x1 y1 z1  x2 y2 z2  x3 y3 z3
//   n — associated particle node index
// ============================================================
bool ReadLDPMTet4FaceFacetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_face_facets = 0;
    out.face_facet_node = VectorXi{};
    out.face_facet_vertices = VectorXR{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4FaceFacetsFile: cannot open " + path);
        return false;
    }

    std::vector<int> v_node;
    std::vector<Real> v_verts;  // 9 per row

    std::string line;
    int line_num = 0;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 10) {
            SetErr(error, "ReadLDPMTet4FaceFacetsFile: expected 10 tokens at data line " + std::to_string(line_num) +
                              " (got " + std::to_string(t.size()) + ")");
            return false;
        }
        int node_idx;
        if (!ParseInt(t[0], node_idx) || node_idx < 0) {
            SetErr(error, "ReadLDPMTet4FaceFacetsFile: invalid node index at data line " + std::to_string(line_num));
            return false;
        }
        Real coords[9];
        for (int k = 0; k < 9; ++k) {
            if (!ParseReal(t[1 + k], coords[k])) {
                SetErr(error, "ReadLDPMTet4FaceFacetsFile: parse error at data line " + std::to_string(line_num));
                return false;
            }
        }
        v_node.push_back(node_idx);
        for (int k = 0; k < 9; ++k)
            v_verts.push_back(coords[k]);
    }

    const int n = static_cast<int>(v_node.size());
    out.n_face_facets = n;
    out.face_facet_node.resize(n);
    out.face_facet_vertices.resize(9 * n);
    for (int i = 0; i < n; ++i) {
        out.face_facet_node(i) = v_node[i];
        for (int k = 0; k < 9; ++k)
            out.face_facet_vertices(9 * i + k) = v_verts[9 * i + k];
    }
    return true;
}

// ============================================================
// ReadLDPMTet4FacetVerticesFile
//
// Format (one row per vertex, 3 tokens):
//   X  Y  Z
// Rows arrive in groups of 3 per sub-facet (IDx, IDy, IDz from facets.dat).
// ============================================================
bool ReadLDPMTet4FacetVerticesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error) {
    out.n_facet_vertices = 0;
    out.facet_vertex_x = VectorXR{};
    out.facet_vertex_y = VectorXR{};
    out.facet_vertex_z = VectorXR{};

    std::ifstream file(path);
    if (!file.is_open()) {
        SetErr(error, "ReadLDPMTet4FacetVerticesFile: cannot open " + path);
        return false;
    }

    std::vector<Real> xs, ys, zs;
    xs.reserve(200000);
    ys.reserve(200000);
    zs.reserve(200000);

    std::string line;
    int line_num = 0;
    while (NextDataLine(file, line)) {
        ++line_num;
        const auto t = TokenizeLDPM(line);
        if (t.size() != 3) {
            SetErr(error, "ReadLDPMTet4FacetVerticesFile: expected 3 tokens at data line " + std::to_string(line_num) +
                              ": \"" + line + "\"");
            return false;
        }
        Real x, y, z;
        if (!ParseReal(t[0], x) || !ParseReal(t[1], y) || !ParseReal(t[2], z)) {
            SetErr(error, "ReadLDPMTet4FacetVerticesFile: parse error at data line " + std::to_string(line_num));
            return false;
        }
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
    }

    const int n = static_cast<int>(xs.size());
    out.n_facet_vertices = n;
    out.facet_vertex_x.resize(n);
    out.facet_vertex_y.resize(n);
    out.facet_vertex_z.resize(n);
    for (int i = 0; i < n; ++i) {
        out.facet_vertex_x(i) = xs[i];
        out.facet_vertex_y(i) = ys[i];
        out.facet_vertex_z(i) = zs[i];
    }
    return true;
}

// ============================================================
// ReadLDPMTet4MeshFromFiles — read all six files using a common
// filename prefix.
// ============================================================
bool ReadLDPMTet4MeshFromFiles(const std::string& prefix, LDPMTet4Mesh& out, std::string* error) {
    out = LDPMTet4Mesh{};

    if (!ReadLDPMTet4NodesFile(prefix + "-data-nodes.dat", out, error))
        return false;
    if (!ReadLDPMTet4ParticlesFile(prefix + "-data-particles.dat", out, error))
        return false;
    if (!ReadLDPMTet4TetsFile(prefix + "-data-tets.dat", out, error))
        return false;
    if (!ReadLDPMTet4FacetsFile(prefix + "-data-facets.dat", out, error))
        return false;
    if (!ReadLDPMTet4FaceFacetsFile(prefix + "-data-faceFacets.dat", out, error))
        return false;
    if (!ReadLDPMTet4FacetVerticesFile(prefix + "-data-facetsVertices.dat", out, error))
        return false;

    return true;
}

// ============================================================
// WriteLDPMTet4TetMeshToVTK
//
// POINTS   — particle positions (n_particles)
// CELLS    — TET4 elements (n_tets), VTK_TETRA (type 10)
// POINT_DATA:
//   "diameter" — aggregate diameter per particle (0 = boundary node)
// ============================================================
bool WriteLDPMTet4TetMeshToVTK(const std::string& filename, const LDPMTet4Mesh& mesh) {
    if (mesh.n_particles <= 0 || mesh.n_tets <= 0) {
        std::cerr << "WriteLDPMTet4TetMeshToVTK: mesh has no particles or tets\n";
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WriteLDPMTet4TetMeshToVTK: cannot open " << filename << "\n";
        return false;
    }

    const int np = mesh.n_particles;
    const int nt = mesh.n_tets;

    // ── Header ───────────────────────────────────────────────────────────────
    file << "# vtk DataFile Version 3.0\n";
    file << "LDPM TET4 Mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // ── Points ───────────────────────────────────────────────────────────────
    file << "\nPOINTS " << np << " double\n";
    for (int i = 0; i < np; ++i) {
        file << mesh.particle_x(i) << " " << mesh.particle_y(i) << " " << mesh.particle_z(i) << "\n";
    }

    // ── Cells (VTK_TETRA = 10, 4 nodes each, 5 integers per cell row) ────────
    file << "\nCELLS " << nt << " " << (nt * 5) << "\n";
    for (int e = 0; e < nt; ++e) {
        file << "4 " << mesh.tet_connectivity(e, 0) << " " << mesh.tet_connectivity(e, 1) << " "
             << mesh.tet_connectivity(e, 2) << " " << mesh.tet_connectivity(e, 3) << "\n";
    }

    file << "\nCELL_TYPES " << nt << "\n";
    for (int e = 0; e < nt; ++e) {
        file << "10\n";  // VTK_TETRA
    }

    // ── Point data: aggregate diameter ───────────────────────────────────────
    file << "\nPOINT_DATA " << np << "\n";
    file << "SCALARS diameter double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < np; ++i) {
        file << (mesh.particle_d.size() > 0 ? mesh.particle_d(i) : 0.0) << "\n";
    }

    file.close();
    return true;
}

// ============================================================
// WriteLDPMTet4TetMeshToVTK  (deformed-position overload)
//
// Same structure as the reference overload, but uses the
// caller-supplied current positions and also writes a
// "displacement" magnitude scalar in POINT_DATA.
// ============================================================
bool WriteLDPMTet4TetMeshToVTK(const std::string& filename,
                               const LDPMTet4Mesh& mesh,
                               const VectorXR& x_cur,
                               const VectorXR& y_cur,
                               const VectorXR& z_cur) {
    if (mesh.n_particles <= 0 || mesh.n_tets <= 0) {
        std::cerr << "WriteLDPMTet4TetMeshToVTK: mesh has no particles or tets\n";
        return false;
    }
    if (static_cast<int>(x_cur.size()) != mesh.n_particles || static_cast<int>(y_cur.size()) != mesh.n_particles ||
        static_cast<int>(z_cur.size()) != mesh.n_particles) {
        std::cerr << "WriteLDPMTet4TetMeshToVTK: current position vector size mismatch\n";
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WriteLDPMTet4TetMeshToVTK: cannot open " << filename << "\n";
        return false;
    }

    const int np = mesh.n_particles;
    const int nt = mesh.n_tets;

    // ── Header ───────────────────────────────────────────────────────────────
    file << "# vtk DataFile Version 3.0\n";
    file << "LDPM TET4 Mesh (deformed)\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // ── Points (current / deformed positions) ────────────────────────────────
    file << "\nPOINTS " << np << " double\n";
    for (int i = 0; i < np; ++i) {
        file << x_cur(i) << " " << y_cur(i) << " " << z_cur(i) << "\n";
    }

    // ── Cells (VTK_TETRA = 10, 4 nodes each, 5 integers per cell row) ────────
    file << "\nCELLS " << nt << " " << (nt * 5) << "\n";
    for (int e = 0; e < nt; ++e) {
        file << "4 " << mesh.tet_connectivity(e, 0) << " " << mesh.tet_connectivity(e, 1) << " "
             << mesh.tet_connectivity(e, 2) << " " << mesh.tet_connectivity(e, 3) << "\n";
    }

    file << "\nCELL_TYPES " << nt << "\n";
    for (int e = 0; e < nt; ++e) {
        file << "10\n";  // VTK_TETRA
    }

    // ── Point data ───────────────────────────────────────────────────────────
    file << "\nPOINT_DATA " << np << "\n";

    // Aggregate diameter (unchanged from reference)
    file << "SCALARS diameter double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < np; ++i) {
        file << (mesh.particle_d.size() > 0 ? mesh.particle_d(i) : 0.0) << "\n";
    }

    // Displacement magnitude |u| = sqrt((x-X)² + (y-Y)² + (z-Z)²)
    file << "\nSCALARS displacement double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < np; ++i) {
        const double dx = x_cur(i) - mesh.particle_x(i);
        const double dy = y_cur(i) - mesh.particle_y(i);
        const double dz = z_cur(i) - mesh.particle_z(i);
        file << std::sqrt(dx * dx + dy * dy + dz * dz) << "\n";
    }

    file.close();
    return true;
}

// ============================================================
// WriteLDPMTet4SubfacetMeshToVTK
//
// POINTS   — exact facet vertex positions (n_facet_vertices)
// CELLS    — sub-facet triangles (n_subfacets), VTK_TRIANGLE (type 5)
// CELL_DATA:
//   "pArea"   — projected (triangle) area
//   "matflag" — material zone flag (0 = mortar; >0 = aggregate/ITZ)
// ============================================================
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename, const LDPMTet4Mesh& mesh) {
    if (mesh.n_subfacets <= 0 || mesh.n_facet_vertices <= 0) {
        std::cerr << "WriteLDPMTet4SubfacetMeshToVTK: mesh has no sub-facets or facet vertices\n";
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WriteLDPMTet4SubfacetMeshToVTK: cannot open " << filename << "\n";
        return false;
    }

    const int nv = mesh.n_facet_vertices;
    const int nf = mesh.n_subfacets;

    // ── Header ───────────────────────────────────────────────────────────────
    file << "# vtk DataFile Version 3.0\n";
    file << "LDPM Sub-facet Mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // ── Points ───────────────────────────────────────────────────────────────
    file << "\nPOINTS " << nv << " double\n";
    for (int i = 0; i < nv; ++i) {
        file << mesh.facet_vertex_x(i) << " " << mesh.facet_vertex_y(i) << " " << mesh.facet_vertex_z(i) << "\n";
    }

    // ── Cells (VTK_TRIANGLE = 5, 3 nodes each, 4 integers per cell row) ──────
    file << "\nCELLS " << nf << " " << (nf * 4) << "\n";
    for (int i = 0; i < nf; ++i) {
        file << "3 " << mesh.subfacet_vertex_ids(3 * i + 0) << " " << mesh.subfacet_vertex_ids(3 * i + 1) << " "
             << mesh.subfacet_vertex_ids(3 * i + 2) << "\n";
    }

    file << "\nCELL_TYPES " << nf << "\n";
    for (int i = 0; i < nf; ++i) {
        file << "5\n";  // VTK_TRIANGLE
    }

    // ── Cell data ────────────────────────────────────────────────────────────
    file << "\nCELL_DATA " << nf << "\n";

    // pArea
    file << "SCALARS pArea double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nf; ++i) {
        file << mesh.subfacet_parea(i) << "\n";
    }

    // matflag
    file << "\nSCALARS matflag int 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nf; ++i) {
        file << mesh.subfacet_matflag(i) << "\n";
    }

    file.close();
    return true;
}

// ============================================================
// WriteLDPMTet4SubfacetMeshToVTK  (failure-ratio overload)
//
// Same POINTS / CELLS layout as the reference overload but
// writes the per-subfacet damage variable ω as "damage" cell
// data instead of pArea and matflag.
// ============================================================
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename,
                                    const LDPMTet4Mesh& mesh,
                                    const VectorXR& subfacet_damage) {
    return WriteLDPMTet4SubfacetMeshToVTK(filename, mesh, "damage", subfacet_damage);
}

// ============================================================
// WriteLDPMTet4SubfacetMeshToVTK  (generic named-field overload)
//
// Writes any per-subfacet scalar under the caller-supplied field_name.
// Supports "crack_distance" [mm], "damage" [0,1], or any other quantity.
// ============================================================
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename,
                                    const LDPMTet4Mesh& mesh,
                                    const std::string& field_name,
                                    const VectorXR& subfacet_values) {
    if (mesh.n_subfacets <= 0 || mesh.n_facet_vertices <= 0) {
        std::cerr << "WriteLDPMTet4SubfacetMeshToVTK: mesh has no sub-facets or facet vertices\n";
        return false;
    }
    if (static_cast<int>(subfacet_values.size()) != mesh.n_subfacets) {
        std::cerr << "WriteLDPMTet4SubfacetMeshToVTK: subfacet_values size mismatch (expected " << mesh.n_subfacets
                  << ", got " << subfacet_values.size() << ")\n";
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WriteLDPMTet4SubfacetMeshToVTK: cannot open " << filename << "\n";
        return false;
    }

    const int nv = mesh.n_facet_vertices;
    const int nf = mesh.n_subfacets;

    // ── Header ───────────────────────────────────────────────────────────────
    file << "# vtk DataFile Version 3.0\n";
    file << "LDPM Sub-facet Mesh (" << field_name << ")\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // ── Points ───────────────────────────────────────────────────────────────
    file << "\nPOINTS " << nv << " double\n";
    for (int i = 0; i < nv; ++i) {
        file << mesh.facet_vertex_x(i) << " " << mesh.facet_vertex_y(i) << " " << mesh.facet_vertex_z(i) << "\n";
    }

    // ── Cells (VTK_TRIANGLE = 5, 3 nodes each, 4 integers per cell row) ──────
    file << "\nCELLS " << nf << " " << (nf * 4) << "\n";
    for (int i = 0; i < nf; ++i) {
        file << "3 " << mesh.subfacet_vertex_ids(3 * i + 0) << " " << mesh.subfacet_vertex_ids(3 * i + 1) << " "
             << mesh.subfacet_vertex_ids(3 * i + 2) << "\n";
    }

    file << "\nCELL_TYPES " << nf << "\n";
    for (int i = 0; i < nf; ++i) {
        file << "5\n";  // VTK_TRIANGLE
    }

    // ── Cell data: named scalar field ────────────────────────────────────────
    file << "\nCELL_DATA " << nf << "\n";
    file << "SCALARS " << field_name << " double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nf; ++i) {
        file << subfacet_values(i) << "\n";
    }

    file.close();
    return true;
}

// ============================================================
// WriteLDPMTet4EdgeDamageToVTK
//
// POINTS   — current particle positions (n_particles)
// CELLS    — one VTK_LINE per unique edge (n_edges), 3 ints per row
// CELL_DATA:
//   "damage" — per-edge Cusatis damage variable ω ∈ [0, 1]
// ============================================================
bool WriteLDPMTet4EdgeDamageToVTK(const std::string& filename,
                                  int n_particles,
                                  int n_edges,
                                  const std::vector<int>& edge_nodes,
                                  const VectorXR& x_cur,
                                  const VectorXR& y_cur,
                                  const VectorXR& z_cur,
                                  const VectorXR& edge_damage) {
    if (n_particles <= 0 || n_edges <= 0) {
        std::cerr << "WriteLDPMTet4EdgeDamageToVTK: no particles or edges\n";
        return false;
    }
    if (static_cast<int>(edge_nodes.size()) != 2 * n_edges) {
        std::cerr << "WriteLDPMTet4EdgeDamageToVTK: edge_nodes size mismatch\n";
        return false;
    }
    if (static_cast<int>(edge_damage.size()) != n_edges) {
        std::cerr << "WriteLDPMTet4EdgeDamageToVTK: edge_damage size mismatch\n";
        return false;
    }
    if (static_cast<int>(x_cur.size()) != n_particles || static_cast<int>(y_cur.size()) != n_particles ||
        static_cast<int>(z_cur.size()) != n_particles) {
        std::cerr << "WriteLDPMTet4EdgeDamageToVTK: position vector size mismatch\n";
        return false;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "WriteLDPMTet4EdgeDamageToVTK: cannot open " << filename << "\n";
        return false;
    }

    // ── Header ───────────────────────────────────────────────────────────────
    file << "# vtk DataFile Version 3.0\n";
    file << "LDPM Edge Damage (omega)\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // ── Points (current / deformed particle positions) ────────────────────────
    file << "\nPOINTS " << n_particles << " double\n";
    for (int i = 0; i < n_particles; ++i) {
        file << x_cur(i) << " " << y_cur(i) << " " << z_cur(i) << "\n";
    }

    // ── Cells (VTK_LINE = 3, 2 nodes each, 3 integers per row) ───────────────
    file << "\nCELLS " << n_edges << " " << (n_edges * 3) << "\n";
    for (int e = 0; e < n_edges; ++e) {
        file << "2 " << edge_nodes[2 * e + 0] << " " << edge_nodes[2 * e + 1] << "\n";
    }

    file << "\nCELL_TYPES " << n_edges << "\n";
    for (int e = 0; e < n_edges; ++e) {
        file << "3\n";  // VTK_LINE
    }

    // ── Cell data: damage variable ω ─────────────────────────────────────────
    file << "\nCELL_DATA " << n_edges << "\n";
    file << "SCALARS damage double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int e = 0; e < n_edges; ++e) {
        file << edge_damage(e) << "\n";
    }

    file.close();
    return true;
}

}  // namespace feris
