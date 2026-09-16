/* Project: FERIS
 * File: tpb_model_test.cc
 * Brief: Host checks for shared surface interpolation and bilateral constraints.
 */
#include "tpb_model.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace feris;
namespace {
void Require(bool c, const char* m) {
    if (!c)
        throw std::runtime_error(m);
}
VectorXR Multiply(const HostCsrMatrix& c, const VectorXR& x) {
    VectorXR out = VectorXR::Zero(c.Rows());
    for (int i = 0; i < c.Rows(); ++i)
        for (int k = c.offsets[i]; k < c.offsets[i + 1]; ++k)
            out(i) += c.values[k] * x(c.columns[k]);
    return out;
}
VectorXR Transpose(const HostCsrMatrix& c, const VectorXR& x, int size) {
    VectorXR out = VectorXR::Zero(size);
    for (int i = 0; i < c.Rows(); ++i)
        for (int k = c.offsets[i]; k < c.offsets[i + 1]; ++k)
            out(c.columns[k]) += c.values[k] * x(i);
    return out;
}
void ProjectionTest(int count) {
    LinearConstraints c;
    // Overlapping chain cannot be projected as independent hand-picked blocks.
    for (int i = 0; i < count - 1; ++i)
        c.AddRow({{i, 1}, {i + 1, -1}});
    VectorXR inverse_mass = VectorXR::LinSpaced(count, 0.5, 2);
    c.Prepare(inverse_mass);
    VectorXR v(count);
    for (int i = 0; i < count; ++i)
        v(i) = std::sin(i * 0.17);
    VectorXR impulse = c.SolveImpulse(-Multiply(c.Matrix(), v));
    VectorXR dv = inverse_mass.asDiagonal() * Transpose(c.Matrix(), impulse, count);
    VectorXR result = v + dv;
    Require(Multiply(c.Matrix(), result).cwiseAbs().maxCoeff() < 1e-9, "Overlapping projection residual");
    Require(std::abs((dv.array() / inverse_mass.array()).sum()) < 1e-9, "Internal constraint momentum conservation");
    Require(std::abs(result.dot(dv.cwiseQuotient(inverse_mass))) < 1e-8, "Mass orthogonality");
    Require((result.array().square() / inverse_mass.array()).sum() <=
                (v.array().square() / inverse_mass.array()).sum() + 1e-9,
            "Projection energy");
    Require(c.HasLargeBlocks() == (count > 513), "Sparse block selection");
}
}  // namespace
int main() {
    try {
        ProjectionTest(8);
        ProjectionTest(520);
        bool rejected = false;
        try {
            LinearConstraints c;
            c.Prescribe(0);
            c.Prescribe(0, 1);
            c.Prepare(VectorXR::Ones(1));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        Require(rejected, "Duplicate prescriptions accepted");
        LinearConstraints rigid;
        rigid.AttachNodeToRigidBody(0, 3, Real3(2, 3, 4));
        rigid.Prescribe(5);
        rigid.Prepare(VectorXR::Ones(9));
        VectorXR q = VectorXR::Zero(9);
        q.segment<3>(3) = Real3(1, 2, 0);
        q.segment<3>(6) = Real3(0.1, 0.2, 0.3);
        q.head<3>() = q.segment<3>(3) + q.segment<3>(6).cross(Real3(2, 3, 4));
        Require(rigid.PositionError(q, 0) < 1e-12, "Rigid attachment sign");
        TetrahedralMesh mesh;
        mesh.positions = {Real3(0, 0, 0),     Real3(1, 0, 0),       Real3(0, 1, 0),
                          Real3(0.5, 0, 0.1), Real3(0.5, 0.5, 0.2), Real3(0, 0.5, 0.1)};
        TetrahedralFace face{{0, 1, 2, 3, 4, 5}};
        VectorXR expected(6);
        Real a = 0.2, b = 0.3, d = 0.5;
        expected << a * (2 * a - 1), b * (2 * b - 1), d * (2 * d - 1), 4 * a * b, 4 * b * d, 4 * d * a;
        Real3 point = Real3::Zero();
        for (int i = 0; i < 6; ++i)
            point += expected(i) * mesh.positions[i];
        VectorXR weights;
        Require(InterpolateFace(mesh, face, point, 1e-10, weights), "Curved quadratic face inverse map");
        Require((weights - expected).norm() < 1e-8, "Quadratic interpolation weights");
        auto arms = tpb::LoadArms(std::string(TPB_DEFAULT_MESH_DIR) + "/TPBT_ElasticPart.inp");
        Require(arms.positions.size() == 3878 && arms.connectivity.rows() == 17658, "Reference arm import");
        Require(!ExtractBoundaryFaces(arms).empty(), "Boundary face extraction");
        Require(std::abs(tpb::PlateDisplacement(0.1) + 1.485) < 1e-14, "Reference final motion");
        Require(std::abs(tpb::PlateDisplacement(0.002) + 0.015) < 1e-14, "Reference ramp transition");
        std::cout << "Host checks passed: overlapping/sparse constraints, momentum, energy, rigid ties, T10 faces, TPB "
                     "mesh and ramp.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
