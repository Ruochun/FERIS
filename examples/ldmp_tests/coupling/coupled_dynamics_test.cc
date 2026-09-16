/* Project: FERIS
 * File: coupled_dynamics_test.cc
 * Brief: GPU element linearization, positive mass, and coupled integration checks.
 */
#include "elements/FEAT4Data.cuh"
#include "elements/FEAT10Data.cuh"
#include "solvers/CoupledLeapfrogSolver.h"
#include "solvers/LeapfrogSolver.cuh"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace feris;
namespace {
void Require(bool c, const char* m) {
    if (!c)
        throw std::runtime_error(m);
}
MatrixXR Dense(const HostCsrMatrix& csr) {
    MatrixXR k = MatrixXR::Zero(csr.Rows(), csr.Rows());
    for (int i = 0; i < csr.Rows(); ++i)
        for (int j = csr.offsets[i]; j < csr.offsets[i + 1]; ++j)
            k(i, csr.columns[j]) += csr.values[j];
    return k;
}
VectorReal3 Nodes(int count) {
    VectorReal3 p = {Real3(0, 0, 0), Real3(1, 0, 0), Real3(0, 1, 0), Real3(0, 0, 1)};
    if (count == 10) {
        const int edges[6][2] = {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}};
        for (auto& e : edges)
            p.push_back((p[e[0]] + p[e[1]]) / 2);
    }
    return p;
}
template <class Element>
void ElementCheck(int count) {
    auto pos = Nodes(count);
    MatrixXi conn(1, count);
    for (int i = 0; i < count; ++i)
        conn(0, i) = i;
    Element data(1, count);
    data.SetupFromMesh(pos, conn);
    struct Cleanup {
        Element& data;
        ~Cleanup() { data.Destroy(); }
    } cleanup{data};
    data.SetLinearizedSVK(1200, 0.25);
    data.SetDensity(2);
    data.SetDamping(0, 0);
    data.CalcMassMatrix();
    VectorXR mass;
    data.RetrieveLumpedMassToCPU(mass);
    Require(mass.minCoeff() > 0, "Non-positive explicit mass");
    Require(std::abs(mass.sum() - Real(1) / 3) < 1e-10, "Total mass conservation");
    MatrixXR k = Dense(data.AssembleTangentStiffnessCSR());
    Require((k - k.transpose()).norm() < 1e-10 * k.norm(), "Stiffness symmetry");
    for (int mode = 0; mode < 6; ++mode) {
        Real3 axis = Real3::Zero();
        axis(mode % 3) = 1;
        VectorXR u(3 * count);
        for (int i = 0; i < count; ++i) {
            if (mode < 3)
                u.segment<3>(3 * i) = axis;
            else
                u.segment<3>(3 * i) = axis.cross(pos[i]);
        }
        Require((k * u).norm() < 1e-10 * k.norm(), "Rigid mode stiffness");
    }
    VectorXR u(3 * count), x(count), y(count), z(count);
    for (int i = 0; i < count; ++i) {
        u.segment<3>(3 * i) = Real3(0.01 * pos[i](0), 0.02 * pos[i](1), -0.005 * pos[i](2));
        Real3 p = pos[i] + u.segment<3>(3 * i);
        x(i) = p(0);
        y(i) = p(1);
        z(i) = p(2);
    }
    data.UpdatePositions(x, y, z);
    data.CalcP();
    data.CalcInternalForce();
    VectorXR force;
    data.RetrieveInternalForceToCPU(force);
    Require((force - k * u).norm() < 1e-8 * (k * u).norm(), "Linearized force differs from K*u");
    Require((Dense(data.AssembleTangentStiffnessCSR()) - k).norm() < 1e-10 * k.norm(),
            "Linearized tangent changes with displacement");
    data.SetSVK(1200, 0.25);
    Require(!data.UsesLinearizedSVK(), "SVK fails to restore finite strain");
    data.CalcP();
    data.CalcInternalForce();
    data.RetrieveInternalForceToCPU(force);
    Require((force - k * u).norm() > 1e-5, "Finite strain path unchanged by setter");
    // Return to rest and use this element in a coupled solve with prescribed motion.
    for (int i = 0; i < count; ++i) {
        x(i) = pos[i](0);
        y(i) = pos[i](1);
        z(i) = pos[i](2);
    }
    data.UpdatePositions(x, y, z);
    data.SetLinearizedSVK(1200, 0.25);
    DynamicsSystem system;
    int offset = system.AddElement(data);
    system.Constraints().Prescribe(offset, 1);
    CoupledLeapfrogSolver solver(system);
    CoupledLeapfrogParams params{1e-5};
    solver.SetParameters(&params);
    for (int step = 0; step < 10; ++step) {
        solver.SetMotionIncrement(1e-6);
        solver.Solve();
    }
    VectorXR disp, vel, reaction;
    solver.RetrieveGeneralizedState(disp, vel, reaction);
    Require(std::abs(disp(0) - 1e-5) < 1e-12, "Prescribed motion drift");
    Require(vel.allFinite() && reaction.allFinite(), "Non-finite coupled element state");
}
}  // namespace
void StandaloneEquivalence() {
    auto pos = Nodes(4);
    MatrixXi conn(1, 4);
    conn << 0, 1, 2, 3;
    GPU_FEAT4_Data a(1, 4), b(1, 4);
    a.SetupFromMesh(pos, conn);
    b.SetupFromMesh(pos, conn);
    struct Cleanup {
        GPU_FEAT4_Data& a;
        GPU_FEAT4_Data& b;
        ~Cleanup() {
            a.Destroy();
            b.Destroy();
        }
    } cleanup{a, b};
    for (auto* data : {&a, &b}) {
        data->SetLinearizedSVK(1200, 0.25);
        data->SetDensity(2);
        data->SetDamping(0, 0);
        data->CalcMassMatrix();
        VectorReal3 load(4, Real3::Zero());
        load[1](0) = 0.1;
        data->SetExternalForce(load);
    }
    LeapfrogSolver standalone(&a);
    LeapfrogParams p{1e-5};
    standalone.SetParameters(&p);
    standalone.Setup();
    standalone.InitialHalfKick();
    DynamicsSystem system;
    system.AddElement(b);
    CoupledLeapfrogSolver coupled(system);
    CoupledLeapfrogParams q{1e-5};
    coupled.SetParameters(&q);
    for (int step = 0; step < 20; ++step) {
        standalone.Solve();
        coupled.Solve();
    }
    VectorXR ax, ay, az, bx, by, bz;
    a.RetrievePositionToCPU(ax, ay, az);
    b.RetrievePositionToCPU(bx, by, bz);
    Require((ax - bx).norm() + (ay - by).norm() + (az - bz).norm() < 1e-12, "Standalone leapfrog behavior changed");
}
void SparseCouplingCheck() {
    DynamicsSystem system;
    std::vector<int> offsets;
    for (int i = 0; i < 521; ++i)
        offsets.push_back(system.AddRigidBody(RigidBody::Box(Real3(1, 1, 1), 1)));
    for (int i = 1; i < 521; ++i)
        system.Constraints().AddRow({{offsets[i], 1}, {offsets[i - 1], -1}});
    int driven = system.Constraints().Prescribe(offsets[0], 1);
    CoupledLeapfrogSolver solver(system);
    Require(system.Constraints().HasLargeBlocks(), "Sparse GPU integration path not selected");
    CoupledLeapfrogParams params{0.01};
    solver.SetParameters(&params);
    VectorXR increments = VectorXR::Zero(521);
    increments(driven) = 1e-6;
    solver.SetConstraintMotionIncrements(increments);
    solver.Solve();
    // Switching back to scalar motion must restore the original row scales.
    solver.SetMotionIncrement(2e-6);
    solver.Solve();
    VectorXR u, v, reaction;
    solver.RetrieveGeneralizedState(u, v, reaction);
    for (int offset : offsets)
        Require(std::abs(u(offset) - 3e-6) < 1e-12, "Sparse coupled projection drift");
}
int main() {
    try {
        StandaloneEquivalence();
        ElementCheck<GPU_FEAT4_Data>(4);
        ElementCheck<GPU_FEAT10_Data>(10);
        SparseCouplingCheck();
        DynamicsSystem system;
        auto body = RigidBody::Box(Real3(2, 3, 4), 1);
        body.force = Real3(3, 0, 0);
        system.AddRigidBody(body);
        CoupledLeapfrogSolver solver(system);
        CoupledLeapfrogParams params{0.01};
        solver.SetParameters(&params);
        for (int i = 0; i < 10; ++i)
            solver.Solve();
        VectorXR u, v, r;
        solver.RetrieveGeneralizedState(u, v, r);
        Require(std::abs(u(0) - 0.5 * (3 / body.mass) * 0.1 * 0.1) < 1e-12, "Rigid body rest-start integration");
        DynamicsSystem pair;
        auto a = RigidBody::Box(Real3(1, 1, 1), 2);
        auto b = RigidBody::Box(Real3(1, 1, 1), 3);
        a.force = Real3(5, 0, 0);
        int ia = pair.AddRigidBody(a), ib = pair.AddRigidBody(b);
        pair.Constraints().TieNodes(ia, ib);
        CoupledLeapfrogSolver paired(pair);
        paired.SetParameters(&params);
        for (int i = 0; i < 10; ++i)
            paired.Solve();
        paired.RetrieveGeneralizedState(u, v, r);
        Require(std::abs(u(ia) - 0.005) < 1e-12 && std::abs(u(ib) - u(ia)) < 1e-12,
                "Coupled rigid-body force transfer");
        std::cout << "GPU checks passed: T4/T10 stiffness, force, mass, coupled motion, and rigid-body acceleration.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
