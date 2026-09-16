/* Project: FERIS
 * File: LinearConstraints.h
 * Brief: Constant bilateral constraints and mass-weighted projection.
 */
#pragma once
#include <memory>
#include <utility>
#include "../utils/SparseMatrix.h"
namespace feris {
using ConstraintRow = std::vector<std::pair<int, Real>>;
class LinearConstraints {
  public:
    LinearConstraints();
    ~LinearConstraints();
    LinearConstraints(const LinearConstraints&) = delete;
    LinearConstraints& operator=(const LinearConstraints&) = delete;
    LinearConstraints(LinearConstraints&&) noexcept;
    LinearConstraints& operator=(LinearConstraints&&) noexcept;
    // Returns a stable reaction row index. target_scale multiplies the supplied
    // scalar motion history; zero gives a homogeneous constraint.
    int AddRow(const ConstraintRow& row, Real target_scale = 0);
    int Prescribe(int dof, Real target_scale = 0);
    void TieNodes(int first, int second);
    void TieNodeToFace(int node_dof, const std::vector<int>& face_node_dofs, const VectorXR& weights);
    void AttachNodeToRigidBody(int node_dof, int body_dof, const Real3& offset_ref);
    // Discovers connected components; rejects dependent/conflicting rows.
    // Small blocks cache inverses; larger blocks retain sparse factorizations.
    void Prepare(const VectorXR& inverse_mass);
    VectorXR SolveImpulse(const VectorXR& residual) const;
    void SolveLargeBlocks(const VectorXR& residual, VectorXR& impulse) const;
    bool HasLargeBlocks() const;
    Real PositionError(const VectorXR& displacement, Real target) const;
    const HostCsrMatrix& Matrix() const;
    const HostCsrMatrix& SmallBlockInverse() const;
    const std::vector<Real>& TargetScales() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace feris
