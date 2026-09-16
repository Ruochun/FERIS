#include "tetrahedral_mesh.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
namespace feris {
namespace {
void Require(bool c, const std::string& m) {
    if (!c)
        throw std::runtime_error(m);
}
}  // namespace
TetrahedralMesh ReadAbaqusTetrahedralMesh(const std::string& inp) {
    std::ifstream input(inp);
    Require(input.good(), "Cannot open elastic-arm mesh: " + inp);
    VectorReal3 nodes;
    std::vector<std::vector<int>> tets;
    int width = 0;
    std::map<int, int> ids;
    std::string line, section;
    while (std::getline(input, line)) {
        if (line.empty() || line.rfind("**", 0) == 0)
            continue;
        if (line[0] == '*') {
            if (line.rfind("*End Part", 0) == 0)
                break;
            section = line;
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int id;
        if (section == "*Node") {
            Real3 p;
            Require(bool(row >> id >> p(0) >> p(1) >> p(2)), "Malformed Abaqus node");
            Require(!ids.count(id), "Duplicate Abaqus node ID");
            ids[id] = nodes.size();
            nodes.push_back(p);
        } else if (section.rfind("*Element", 0) == 0) {
            int count = section.find("type=C3D10") != std::string::npos  ? 10
                        : section.find("type=C3D4") != std::string::npos ? 4
                                                                         : 0;
            Require(count && (!width || width == count), "Expected uniform C3D4 or C3D10 mesh");
            width = count;
            std::vector<int> tet(count);
            Require(bool(row >> id), "Malformed element ID");
            for (int& n : tet)
                Require(bool(row >> n), "Malformed Abaqus tetrahedron");
            for (int& n : tet)
                n = ids.at(n);
            tets.push_back(tet);
        }
    }
    Require(!nodes.empty() && !tets.empty(), "Empty Abaqus elastic-arm mesh");

    TetrahedralMesh mesh;
    mesh.positions = std::move(nodes);
    mesh.connectivity.resize(tets.size(), width);
    for (int t = 0; t < static_cast<int>(tets.size()); ++t) {
        auto tet = tets[t];
        Matrix3R j;
        for (int a = 0; a < 3; ++a)
            j.col(a) = mesh.positions[tet[a + 1]] - mesh.positions[tet[0]];
        Require(std::abs(j.determinant()) > 1e-12, "Degenerate tetrahedron");
        if (j.determinant() < 0) {
            std::swap(tet[1], tet[2]);
            if (width == 10) {
                std::swap(tet[4], tet[6]);
                std::swap(tet[8], tet[9]);
            }
        }
        for (int a = 0; a < width; ++a)
            mesh.connectivity(t, a) = tet[a];
    }
    return mesh;
}
std::vector<TetrahedralFace> ExtractBoundaryFaces(const TetrahedralMesh& mesh) {
    Require(mesh.connectivity.cols() == 4 || mesh.connectivity.cols() == 10, "Expected T4/T10 connectivity");
    const int face_nodes[4][6] = {{0, 1, 2, 4, 5, 6}, {0, 1, 3, 4, 8, 7}, {0, 2, 3, 6, 9, 7}, {1, 2, 3, 5, 9, 8}};
    std::map<std::array<int, 3>, std::pair<TetrahedralFace, int>> faces;
    for (int t = 0; t < mesh.connectivity.rows(); ++t)
        for (int f = 0; f < 4; ++f) {
            TetrahedralFace face;
            for (int a = 0; a < (mesh.connectivity.cols() == 4 ? 3 : 6); ++a)
                face.nodes.push_back(mesh.connectivity(t, face_nodes[f][a]));
            std::array<int, 3> key{face.nodes[0], face.nodes[1], face.nodes[2]};
            std::sort(key.begin(), key.end());
            auto& entry = faces[key];
            if (entry.second) {
                auto old = entry.first.nodes, current = face.nodes;
                std::sort(old.begin(), old.end());
                std::sort(current.begin(), current.end());
                Require(old == current, "Inconsistent nodes on shared tetrahedral face");
            }
            entry.first = face;
            ++entry.second;
            Require(entry.second <= 2, "Non-manifold tetrahedral face");
        }
    std::vector<TetrahedralFace> out;
    for (const auto& entry : faces)
        if (entry.second.second == 1)
            out.push_back(entry.second.first);
    return out;
}
bool InterpolateFace(const TetrahedralMesh& mesh,
                     const TetrahedralFace& face,
                     const Real3& p,
                     Real tolerance,
                     VectorXR& weights) {
    Require(std::isfinite(tolerance) && tolerance > 0 && p.allFinite(), "Invalid face query");
    const int count = face.nodes.size();
    Require(count == 3 || count == 6, "Expected three- or six-node face");
    Matrix<Real, 3, 2> jac;
    jac.col(0) = mesh.positions.at(face.nodes[1]) - mesh.positions.at(face.nodes[0]);
    jac.col(1) = mesh.positions.at(face.nodes[2]) - mesh.positions.at(face.nodes[0]);
    if (jac.col(0).cross(jac.col(1)).norm() < 1e-12)
        return false;
    Matrix<Real, 2, 1> q = jac.colPivHouseholderQr().solve(p - mesh.positions.at(face.nodes[0]));
    weights.resize(count);
    for (int iter = 0; iter < 20; ++iter) {
        Real l[3] = {1 - q(0) - q(1), q(0), q(1)};
        const Real dl[3][2] = {{-1, -1}, {1, 0}, {0, 1}};
        MatrixXR grad(count, 2);
        for (int a = 0; a < 3; ++a) {
            weights(a) = count == 3 ? l[a] : l[a] * (2 * l[a] - 1);
            for (int d = 0; d < 2; ++d)
                grad(a, d) = dl[a][d] * (count == 3 ? 1 : 4 * l[a] - 1);
        }
        if (count == 6)
            for (int a = 0; a < 3; ++a) {
                int b = (a + 1) % 3;
                weights(3 + a) = 4 * l[a] * l[b];
                for (int d = 0; d < 2; ++d)
                    grad(3 + a, d) = 4 * (dl[a][d] * l[b] + l[a] * dl[b][d]);
            }
        Real3 mapped = Real3::Zero();
        jac.setZero();
        for (int a = 0; a < count; ++a) {
            mapped += weights(a) * mesh.positions.at(face.nodes[a]);
            for (int d = 0; d < 2; ++d)
                jac.col(d) += grad(a, d) * mesh.positions.at(face.nodes[a]);
        }
        if ((mapped - p).norm() <= tolerance)
            return l[0] >= -1e-7 && l[1] >= -1e-7 && l[2] >= -1e-7;
        q -= jac.colPivHouseholderQr().solve(mapped - p);
    }
    return false;
}
}  // namespace feris
