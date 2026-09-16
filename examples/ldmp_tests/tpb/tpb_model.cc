/* Project: FERIS
 * File: tpb_model.cc
 * Brief: TPB-specific placement, support selection, loading, and visualization.
 */
#include "tpb_model.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace feris {
namespace tpb {
namespace {
void Require(bool c, const std::string& m) {
    if (!c)
        throw std::runtime_error(m);
}
}  // namespace
TetrahedralMesh LoadArms(const std::string& inp) {
    auto original = ReadAbaqusTetrahedralMesh(inp);
    Require(original.connectivity.cols() == 4, "TPB reference requires C3D4 arms");
    TetrahedralMesh mesh;
    mesh.connectivity.resize(2 * original.connectivity.rows(), 4);
    for (int side = 0; side < 2; ++side) {
        Real shift = side ? 100 : -525.5;
        int offset = mesh.positions.size();
        for (auto p : original.positions) {
            p(0) += shift;
            mesh.positions.push_back(p);
        }
        mesh.connectivity.middleRows(side * original.connectivity.rows(), original.connectivity.rows()) =
            original.connectivity.array() + offset;
    }
    return mesh;
}
Model BuildModel(const LDPMTet4Mesh& core,
                 const TetrahedralMesh& arms,
                 const HostCsrMatrix& stiffness,
                 DynamicsSystem& system,
                 int arm_offset) {
    Model model;
    model.core_dofs = arm_offset;
    model.arm_pos_ref = arms.positions;
    model.stiffness = stiffness;
    for (int t = 0; t < arms.connectivity.rows(); ++t)
        model.arm_tets.push_back(
            {arms.connectivity(t, 0), arms.connectivity(t, 1), arms.connectivity(t, 2), arms.connectivity(t, 3)});
    model.plate_offset = system.AddRigidBody(RigidBody::Box(Real3(25, 100, 10), 7.8e-9));
    system.AddRigidBody(RigidBody::Box(Real3(25, 100, 10), 7.8e-9));
    model.inv_mass = system.InverseMass();
    model.elastic_dt =
        stiffness.CriticalTimeStep(model.inv_mass.segment(arm_offset, 3 * arms.positions.size()).cwiseInverse());
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

    auto& constraints = system.Constraints();
    auto faces = ExtractBoundaryFaces(arms);
    for (Real x : {Real(0), Real(100)}) {
        std::vector<TetrahedralFace> surface;
        for (const auto& f : faces) {
            bool match = true;
            for (int n : f.nodes)
                if (std::abs(arms.positions[n](0) - x) > 1e-3)
                    match = false;
            if (match)
                surface.push_back(f);
        }
        int count = 0;
        for (int n = 0; n < core.n_particles; ++n) {
            if (std::abs(core.particle_x(n) - x) >= 1e-3)
                continue;
            bool found = false;
            for (const auto& face : surface) {
                VectorXR weights;
                if (!InterpolateFace(arms, face, Real3(core.particle_x(n), core.particle_y(n), core.particle_z(n)),
                                     1e-7, weights))
                    continue;
                std::vector<int> dofs;
                for (int node : face.nodes)
                    dofs.push_back(arm_offset + 3 * node);
                constraints.TieNodeToFace(3 * n, dofs, weights);
                found = true;
                ++count;
                break;
            }
            Require(found, "Unmatched TPB interface particle " + std::to_string(n));
        }
        Require(count > 0, "Missing TPB interface");
        std::cout << "Interface x=" << x << ": " << count << " particles\n";
    }
    for (int side = 0; side < 2; ++side) {
        Real cx = side ? 613 : -513;
        int p = model.plate_offset + 6 * side, count = 0;
        for (int n = 0; n < static_cast<int>(arms.positions.size()); ++n) {
            const auto& pos = arms.positions[n];
            if (std::abs(pos(2)) > 1e-3 || std::abs(pos(0) - cx) > 12.5 + 1e-3)
                continue;
            constraints.AttachNodeToRigidBody(arm_offset + 3 * n, p, pos - Real3(cx, 50, -5));
            ++count;
        }
        Require(count > 0, "Missing TPB support nodes");
        model.support_rows[side] = constraints.Prescribe(p + 2);
        std::cout << "Support " << side << ": " << count << " nodes\n";
    }
    std::vector<int> gauges;
    for (int n = 0; n < core.n_particles; ++n) {
        Real x = core.particle_x(n), y = core.particle_y(n), z = core.particle_z(n);
        if (std::abs(z - 200) < 1e-3 && x >= 37.5 - 1e-3 && x <= 62.5 + 1e-3) {
            model.top_nodes.push_back(n);
            constraints.Prescribe(3 * n);
            constraints.Prescribe(3 * n + 2, 1);
        }
        if (y < 1e-3 && z <= 1e-3 && x >= 47.5 - 1e-3 && x <= 52.5 + 1e-3)
            gauges.push_back(n);
    }
    Require(!model.top_nodes.empty(), "Missing TPB loading strip");
    Require(gauges.size() == 2, "Expected two TPB CMOD nodes");
    std::sort(gauges.begin(), gauges.end(), [&](int a, int b) { return core.particle_x(a) < core.particle_x(b); });
    model.cmod_nodes = {gauges[0], gauges[1]};
    model.drive = constraints.TargetScales();
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
