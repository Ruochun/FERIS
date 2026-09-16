/* Project: FERIS
 * File: DynamicsSystem.h
 * Brief: Element groups, linearized rigid bodies, and bilateral connections.
 */
#pragma once
#include <memory>
#include "../constraints/LinearConstraints.h"
#include "../elements/ElementBase.h"
namespace feris {
struct RigidBody {
    Real mass;
    Real3 inertia;
    Real3 force = Real3::Zero(), moment = Real3::Zero();
    static RigidBody Box(const Real3& dimensions, Real density);
};
class CoupledLeapfrogSolver;
class DynamicsSystem {
  public:
    DynamicsSystem();
    ~DynamicsSystem();
    DynamicsSystem(const DynamicsSystem&) = delete;
    DynamicsSystem& operator=(const DynamicsSystem&) = delete;
    // Borrow element storage; it must outlive the system and its solver.
    // Returns the first generalized translational DOF (three per node).
    int AddElement(ElementBase& element);
    // Linearized body coordinates: translation followed by rotation, six DOFs.
    int AddRigidBody(const RigidBody& body);
    LinearConstraints& Constraints();
    const VectorXR& InverseMass() const;

  private:
    friend class CoupledLeapfrogSolver;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace feris
