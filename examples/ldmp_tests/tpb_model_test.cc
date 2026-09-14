/* Project: FERIS
 * File: tpb_model_test.cc
 * Brief: CPU checks of TPB arm elasticity, constraint projection, and ramp.
 */
#include "tpb_model.h"
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace feris;
using namespace feris::tpb;
namespace {
void Require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
Eigen::SparseMatrix<Real> MatrixFromCsr(const Csr& csr, int columns) {
    std::vector<Eigen::Triplet<Real>> entries;
    for (int i = 0; i < csr.Rows(); ++i)
        for (int j = csr.offsets[i]; j < csr.offsets[i + 1]; ++j)
            entries.emplace_back(i, csr.columns[j], csr.values[j]);
    Eigen::SparseMatrix<Real> result(csr.Rows(), columns);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}
}  // namespace
int main() {
    try {
        LDPMTet4Mesh core;
        std::string error;
        if (!ReadLDPMTet4MeshFromFiles(std::string(TPB_DEFAULT_MESH_DIR) + "/LDPMgeo000", core, &error))
            throw std::runtime_error(error);
        VectorXR mass = VectorXR::Zero(core.n_particles);
        for (int t = 0; t < core.n_tets; ++t) {
            Matrix<Real, 4, 4> p;
            for (int a = 0; a < 4; ++a) {
                int n = core.tet_connectivity(t, a);
                p.row(a) << 1, core.particle_x(n), core.particle_y(n), core.particle_z(n);
            }
            Real nodal_mass = std::abs(p.determinant()) * 2.338e-9 / 24;
            for (int a = 0; a < 4; ++a)
                mass(core.tet_connectivity(t, a)) += nodal_mass;
        }
        auto model = BuildModel(core, std::string(TPB_DEFAULT_MESH_DIR) + "/TPBT_ElasticPart.inp", mass);
        auto k = MatrixFromCsr(model.stiffness, model.stiffness.Rows());
        Eigen::SparseMatrix<Real> asymmetry = k - Eigen::SparseMatrix<Real>(k.transpose());
        Require(asymmetry.norm() / k.norm() < 1e-13, "Elastic stiffness is not symmetric");
        // Each disconnected elastic arm has six zero-energy rigid modes.
        for (int side = 0; side < 2; ++side) {
            for (int mode = 0; mode < 6; ++mode) {
                VectorXR u = VectorXR::Zero(k.rows());
                Real3 axis = Real3::Zero();
                axis(mode % 3) = 1;
                int arm_nodes = model.arm_pos_ref.size() / 2;
                for (int i = side * arm_nodes; i < (side + 1) * arm_nodes; ++i) {
                    if (mode < 3)
                        u.segment<3>(3 * i) = axis;
                    else
                        u.segment<3>(3 * i) = axis.cross(model.arm_pos_ref[i]);
                }
                Require((k * u).norm() / (k.norm() * u.norm()) < 1e-13, "Elastic rigid mode has nonzero force");
            }
        }
        VectorXR trial = VectorXR::LinSpaced(k.rows(), -1, 1);
        Require(trial.dot(k * trial) > 0, "Elastic strain energy is not positive");
        auto c = MatrixFromCsr(model.constraints, model.inv_mass.size());
        auto inverse = MatrixFromCsr(model.gram_inverse, c.rows());
        VectorXR velocity(model.inv_mass.size());
        for (int i = 0; i < velocity.size(); ++i)
            velocity(i) = std::sin(0.23 * i);
        VectorXR target = Eigen::Map<const VectorXR>(model.drive.data(), model.drive.size()) * Real(-15);
        VectorXR impulse = inverse * (target - c * velocity);
        VectorXR correction = model.inv_mass.asDiagonal() * (c.transpose() * impulse);
        VectorXR projected = velocity + correction;
        Require((c * projected - target).cwiseAbs().maxCoeff() < 1e-8, "Projection fails velocity constraints");
        // Orthogonality of the correction to admissible velocities is the
        // mass-metric conservation check; zero targets cannot add kinetic energy.
        VectorXR impulse_zero = inverse * (-c * velocity);
        VectorXR projected_zero = velocity + model.inv_mass.asDiagonal() * (c.transpose() * impulse_zero);
        VectorXR momentum_correction = (projected_zero - velocity).cwiseQuotient(model.inv_mass);
        Require(std::abs(projected_zero.dot(momentum_correction)) < 1e-10, "Projection violates mass orthogonality");
        Require((projected_zero.array().square() / model.inv_mass.array()).sum() <=
                    (velocity.array().square() / model.inv_mass.array()).sum() + 1e-12,
                "Projection adds kinetic energy");
        // An admissible displacement stays admissible after the corrected drift.
        Real dt = 1e-6;
        VectorXR disp = dt * projected;
        Require((c * disp - dt * target).cwiseAbs().maxCoeff() < 1e-12, "Projected drift violates positions");
        Require(std::abs(PlateDisplacement(0.002) + 0.015) < 1e-14, "Ramp transition displacement");
        Require(std::abs(PlateDisplacement(0.1) + 1.485) < 1e-14, "Final reference displacement");
        Real h = 1e-8;
        Require(std::abs((PlateDisplacement(0.002 + h) - PlateDisplacement(0.002 - h)) / (2 * h) + 15) < 1e-4,
                "Ramp transition velocity");
        std::cout << "TPB model checks passed (elastic rigid modes, projection, energy, loading ramp).\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TPB model check failed: " << e.what() << '\n';
        return 1;
    }
}
