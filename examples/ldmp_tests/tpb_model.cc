/* Project: FERIS
 * File: tpb_model.cc
 * Brief: Assemble the reference TPB elastic arms and translation-only interface
 *        ties, rigid support plates, and x/z loading constraints.
 */
#include "tpb_model.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace feris {
namespace tpb {
namespace {
using Sparse = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Entry = Eigen::Triplet<Real>;
using Row = std::vector<std::pair<int, Real>>;
constexpr Real rho = 2.338e-9;
void Require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
Csr ToCsr(const Sparse& matrix) {
    Csr out;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (Sparse::InnerIterator it(matrix, i); it; ++it) {
            out.columns.push_back(it.col());
            out.values.push_back(it.value());
        }
        out.offsets.push_back(out.columns.size());
    }
    return out;
}
// Blocks have disjoint DOFs, so their mass-weighted projections commute.
void AddBlock(Model& model, const std::vector<Row>& rows, Real drive = 0) {
    Require(!rows.empty(), "Empty TPB constraint group");
    const int start = model.constraints.Rows();
    std::map<int, int> local;
    for (const auto& row : rows)
        for (auto [dof, value] : row)
            if (!local.count(dof))
                local[dof] = local.size();
    MatrixXR c = MatrixXR::Zero(rows.size(), local.size());
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        for (auto [dof, value] : rows[i]) {
            c(i, local.at(dof)) += value * std::sqrt(model.inv_mass(dof));
            model.constraints.columns.push_back(dof);
            model.constraints.values.push_back(value);
        }
        model.constraints.offsets.push_back(model.constraints.columns.size());
        model.drive.push_back(drive);
    }
    MatrixXR gram = c * c.transpose();
    // Scale the Gram matrix to unit diagonal before factorization.
    VectorXR scale = gram.diagonal().array().sqrt().inverse();
    MatrixXR normalized = scale.asDiagonal() * gram * scale.asDiagonal();
    Eigen::LDLT<MatrixXR> factor(normalized);
    Require(factor.info() == Eigen::Success && factor.isPositive(), "Singular TPB constraints");
    MatrixXR inverse =
        scale.asDiagonal() * factor.solve(MatrixXR::Identity(rows.size(), rows.size())) * scale.asDiagonal();
    Require((gram * inverse - MatrixXR::Identity(rows.size(), rows.size())).cwiseAbs().maxCoeff() < 1e-7,
            "Inaccurate TPB constraint inverse");
    for (int i = 0; i < inverse.rows(); ++i) {
        for (int j = 0; j < inverse.cols(); ++j) {
            model.gram_inverse.columns.push_back(start + j);
            model.gram_inverse.values.push_back(inverse(i, j));
        }
        model.gram_inverse.offsets.push_back(model.gram_inverse.columns.size());
    }
}
}  // namespace

Model BuildModel(const LDPMTet4Mesh& core, const std::string& inp, const VectorXR& core_mass) {
    Model model;
    model.core_dofs = 3 * core.n_particles;
    std::ifstream input(inp);
    Require(input.good(), "Cannot open elastic-arm mesh: " + inp);
    VectorReal3 nodes;
    std::vector<std::array<int, 4>> tets;
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
            Require(section.find("type=C3D4") != std::string::npos, "Expected C3D4 elastic arm");
            std::array<int, 4> tet;
            Require(bool(row >> id >> tet[0] >> tet[1] >> tet[2] >> tet[3]), "Malformed Abaqus tetrahedron");
            for (int& n : tet)
                n = ids.at(n);
            tets.push_back(tet);
        }
    }
    Require(!nodes.empty() && !tets.empty(), "Empty Abaqus elastic-arm mesh");
    for (Real shift : {Real(-525.5), Real(100)}) {
        const int offset = model.arm_pos_ref.size();
        for (auto p : nodes) {
            p(0) += shift;
            model.arm_pos_ref.push_back(p);
        }
        for (auto tet : tets) {
            for (int& n : tet)
                n += offset;
            model.arm_tets.push_back(tet);
        }
    }
    const int arm_dofs = 3 * model.arm_pos_ref.size();
    model.plate_offset = model.core_dofs + arm_dofs;
    VectorXR mass = VectorXR::Zero(model.plate_offset + 12);
    for (int i = 0; i < core.n_particles; ++i)
        mass.segment<3>(3 * i).setConstant(core_mass(i));
    const Real young = 39000, nu = 0.2;
    const Real lambda = young * nu / ((1 + nu) * (1 - 2 * nu));
    const Real mu = young / (2 * (1 + nu));
    Matrix<Real, 6, 6> d = Matrix<Real, 6, 6>::Zero();
    d.topLeftCorner<3, 3>().setConstant(lambda);
    d.diagonal().head<3>().array() += 2 * mu;
    d.diagonal().tail<3>().setConstant(mu);
    std::vector<Entry> entries;
    entries.reserve(model.arm_tets.size() * 144);
    for (const auto& tet : model.arm_tets) {
        Matrix<Real, 4, 4> coords;
        for (int a = 0; a < 4; ++a) {
            coords(a, 0) = 1;
            coords.block<1, 3>(a, 1) = model.arm_pos_ref[tet[a]].transpose();
        }
        Real volume = std::abs(coords.determinant()) / 6;
        Require(volume > 1e-12, "Degenerate elastic tetrahedron");
        Matrix<Real, 3, 4> grad = coords.inverse().bottomRows<3>();
        Matrix<Real, 6, 12> b = Matrix<Real, 6, 12>::Zero();
        for (int a = 0; a < 4; ++a) {
            Real gx = grad(0, a), gy = grad(1, a), gz = grad(2, a);
            b(0, 3 * a) = gx;
            b(1, 3 * a + 1) = gy;
            b(2, 3 * a + 2) = gz;
            b(3, 3 * a) = gy;
            b(3, 3 * a + 1) = gx;
            b(4, 3 * a + 1) = gz;
            b(4, 3 * a + 2) = gy;
            b(5, 3 * a) = gz;
            b(5, 3 * a + 2) = gx;
            mass.segment<3>(model.core_dofs + 3 * tet[a]).array() += rho * volume / 4;
        }
        Matrix<Real, 12, 12> k = volume * b.transpose() * d * b;
        for (int a = 0; a < 12; ++a)
            for (int b = 0; b < 12; ++b)
                entries.emplace_back(3 * tet[a / 3] + a % 3, 3 * tet[b / 3] + b % 3, k(a, b));
    }
    for (int side = 0; side < 2; ++side) {
        const int p = model.plate_offset + 6 * side;
        const Real m = 25 * 100 * 10 * 7.8e-9;
        mass.segment<3>(p).setConstant(m);
        mass(p + 3) = m * (100 * 100 + 10 * 10) / 12;
        mass(p + 4) = m * (25 * 25 + 10 * 10) / 12;
        mass(p + 5) = m * (25 * 25 + 100 * 100) / 12;
    }
    Require(mass.allFinite() && mass.minCoeff() > 0, "Non-positive TPB nodal mass");
    model.inv_mass = mass.cwiseInverse();
    Sparse stiffness(arm_dofs, arm_dofs);
    stiffness.setFromTriplets(entries.begin(), entries.end());
    model.stiffness = ToCsr(stiffness);
    Real spectral_bound = 0;
    for (int i = 0; i < arm_dofs; ++i) {
        Real sum = 0;
        for (Sparse::InnerIterator it(stiffness, i); it; ++it)
            sum += std::abs(it.value()) / std::sqrt(mass(model.core_dofs + i) * mass(model.core_dofs + it.col()));
        spectral_bound = std::max(spectral_bound, sum);
    }
    model.elastic_dt = 2 / std::sqrt(spectral_bound);
    model.min_edge = std::numeric_limits<Real>::max();
    for (int t = 0; t < core.n_tets; ++t)
        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b) {
                int i = core.tet_connectivity(t, a), j = core.tet_connectivity(t, b);
                model.min_edge = std::min(model.min_edge, Real3(core.particle_x(i) - core.particle_x(j),
                                                                core.particle_y(i) - core.particle_y(j),
                                                                core.particle_z(i) - core.particle_z(j))
                                                              .norm());
            }
    // Reference interface triangles lie on x=0 and x=100. The reference
    // ChLinkNodeRotFace imposes three translations, with rotations left free.
    for (Real x : {Real(0), Real(100)}) {
        std::vector<std::array<int, 3>> faces;
        for (auto tet : model.arm_tets) {
            std::vector<int> face;
            for (int n : tet)
                if (std::abs(model.arm_pos_ref[n](0) - x) < 1e-3)
                    face.push_back(n);
            if (face.size() == 3)
                faces.push_back({face[0], face[1], face[2]});
        }
        std::array<std::vector<Row>, 3> rows;
        for (int n = 0; n < core.n_particles; ++n) {
            if (std::abs(core.particle_x(n) - x) >= 1e-3)
                continue;
            bool found = false;
            for (auto face : faces) {
                Matrix3R yz;
                for (int a = 0; a < 3; ++a)
                    yz.col(a) = Real3(1, model.arm_pos_ref[face[a]](1), model.arm_pos_ref[face[a]](2));
                if (std::abs(yz.determinant()) < 1e-12)
                    continue;
                Real3 weights = yz.inverse() * Real3(1, core.particle_y(n), core.particle_z(n));
                if (weights.minCoeff() < -1e-7 || weights.maxCoeff() > 1 + 1e-7)
                    continue;
                for (int c = 0; c < 3; ++c) {
                    Row row{{3 * n + c, 1}};
                    for (int a = 0; a < 3; ++a)
                        row.emplace_back(model.core_dofs + 3 * face[a] + c, -weights(a));
                    rows[c].push_back(row);
                }
                found = true;
                break;
            }
            Require(found, "Unmatched LDPM interface particle " + std::to_string(n));
        }
        std::cout << "Interface x=" << x << ": " << rows[0].size() << " particles\n";
        for (auto& block : rows)
            AddBlock(model, block);
    }
    // Rigid bottom plates: only their center z translation is restrained.
    // The other five DOFs are dynamic, as in the Chrono reference.
    for (int side = 0; side < 2; ++side) {
        const Real cx = side == 0 ? -513 : 613;
        const int p = model.plate_offset + 6 * side;
        std::vector<Row> rows;
        for (int n = 0; n < static_cast<int>(model.arm_pos_ref.size()); ++n) {
            const auto& pos = model.arm_pos_ref[n];
            if (std::abs(pos(2)) > 1e-3 || std::abs(pos(0) - cx) > 12.5 + 1e-3)
                continue;
            Real3 r = pos - Real3(cx, 50, -5);
            Matrix3R skew;
            skew << 0, -r(2), r(1), r(2), 0, -r(0), -r(1), r(0), 0;
            for (int c = 0; c < 3; ++c) {
                Row row{{model.core_dofs + 3 * n + c, 1}, {p + c, -1}};
                for (int a = 0; a < 3; ++a)
                    if (skew(c, a) != 0)
                        row.emplace_back(p + 3 + a, skew(c, a));
                rows.push_back(row);
            }
        }
        Require(!rows.empty(), "Missing TPB support nodes");
        std::cout << "Support " << side << ": " << rows.size() / 3 << " nodes\n";
        model.support_rows[side] = model.constraints.Rows() + rows.size();
        rows.push_back({{p + 2, 1}});
        AddBlock(model, rows);
    }
    std::vector<Row> top_x, top_z;
    std::vector<int> gauges;
    for (int n = 0; n < core.n_particles; ++n) {
        Real x = core.particle_x(n), y = core.particle_y(n), z = core.particle_z(n);
        if (std::abs(z - 200) < 1e-3 && x >= 37.5 - 1e-3 && x <= 62.5 + 1e-3) {
            model.top_nodes.push_back(n);
            top_x.push_back({{3 * n, 1}});
            top_z.push_back({{3 * n + 2, 1}});
        }
        if (y < 1e-3 && z <= 1e-3 && x >= 47.5 - 1e-3 && x <= 52.5 + 1e-3)
            gauges.push_back(n);
    }
    Require(gauges.size() == 2, "Expected exactly two reference CMOD gauge nodes");
    std::sort(gauges.begin(), gauges.end(), [&](int a, int b) { return core.particle_x(a) < core.particle_x(b); });
    model.cmod_nodes = {gauges[0], gauges[1]};
    AddBlock(model, top_x);
    AddBlock(model, top_z, 1);
    std::cout << "Loading particles: " << model.top_nodes.size() << "; CMOD nodes: " << gauges[0] << ", " << gauges[1]
              << "; constraints: " << model.constraints.Rows() << '\n';
    return model;
}

void WriteArms(const std::string& path, const Model& model, const VectorXR& disp) {
    std::ofstream out(path);
    Require(out.good(), "Cannot write " + path);
    out.precision(12);
    out << "# vtk DataFile Version 3.0\nTPB elastic arms\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS "
        << model.arm_pos_ref.size() << " double\n";
    for (int n = 0; n < static_cast<int>(model.arm_pos_ref.size()); ++n)
        out << (model.arm_pos_ref[n] + disp.segment<3>(model.core_dofs + 3 * n)).transpose() << '\n';
    out << "CELLS " << model.arm_tets.size() << ' ' << 5 * model.arm_tets.size() << '\n';
    for (auto tet : model.arm_tets)
        out << "4 " << tet[0] << ' ' << tet[1] << ' ' << tet[2] << ' ' << tet[3] << '\n';
    out << "CELL_TYPES " << model.arm_tets.size() << '\n';
    for (size_t i = 0; i < model.arm_tets.size(); ++i)
        out << "10\n";
    out << "POINT_DATA " << model.arm_pos_ref.size() << "\nVECTORS displacement double\n";
    for (int n = 0; n < static_cast<int>(model.arm_pos_ref.size()); ++n)
        out << disp.segment<3>(model.core_dofs + 3 * n).transpose() << '\n';
    Require(out.good(), "Failed writing " + path);
}
}  // namespace tpb
}  // namespace feris
