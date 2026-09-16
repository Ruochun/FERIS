/* Project: FERIS
 * File: CoupledLeapfrogSolver.h
 * Brief: Common-step leapfrog integration of LDPM/T4/T10 and linear rigid bodies.
 */
#pragma once
#include <memory>
#include "SolverBase.h"
#include "../dynamics/DynamicsSystem.h"
namespace feris {
struct CoupledLeapfrogParams {
    Real time_step;
};
class CoupledLeapfrogSolver : public SolverBase {
  public:
    explicit CoupledLeapfrogSolver(DynamicsSystem& system);
    ~CoupledLeapfrogSolver() override;
    CoupledLeapfrogSolver(const CoupledLeapfrogSolver&) = delete;
    CoupledLeapfrogSolver& operator=(const CoupledLeapfrogSolver&) = delete;
    void SetParameters(void* parameters) override;
    // Motion increment over the next time step. Row-specific scaling belongs
    // to the constraint set. Zero is the default for homogeneous systems.
    void SetMotionIncrement(Real displacement_increment);
    // Independent generalized motion increments, one per constraint row.
    void SetConstraintMotionIncrements(const VectorXR& increments);
    void Solve() override;
    // Generalized buffers include both translations and body angles. User
    // node-position/force collections remain available on the element APIs.
    void RetrieveGeneralizedState(VectorXR& displacement, VectorXR& velocity, VectorXR& reactions);
    Real KineticEnergy();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace feris
