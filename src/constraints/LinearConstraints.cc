/* Project: FERIS
 * File: LinearConstraints.cc
 * Brief: Connected constraint components and mass-weighted projection factors.
 */
#include "LinearConstraints.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
namespace feris {
struct LinearConstraints::Impl {
    HostCsrMatrix matrix, inverse;
    std::vector<Real> drive;
    struct LargeBlock {
        std::vector<int> rows;
        VectorXR scale;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<Real>> factor;
    };
    std::vector<std::unique_ptr<LargeBlock>> large;
    bool prepared = false;
};
LinearConstraints::LinearConstraints() : impl_(new Impl) {}
LinearConstraints::~LinearConstraints() = default;
LinearConstraints::LinearConstraints(LinearConstraints&&) noexcept = default;
LinearConstraints& LinearConstraints::operator=(LinearConstraints&&) noexcept = default;
int LinearConstraints::AddRow(const ConstraintRow& row, Real scale) {
    if (impl_->prepared)
        throw std::logic_error("Cannot change prepared constraints");
    if (!std::isfinite(scale))
        throw std::invalid_argument("Non-finite constraint target");
    std::map<int, Real> combined;
    for (auto [dof, value] : row) {
        if (dof < 0 || !std::isfinite(value))
            throw std::invalid_argument("Invalid constraint coefficient");
        combined[dof] += value;
    }
    auto& c = impl_->matrix;
    int index = c.Rows(), start = c.values.size();
    for (auto [dof, value] : combined)
        if (value != 0) {
            c.columns.push_back(dof);
            c.values.push_back(value);
        }
    if (start == static_cast<int>(c.values.size()))
        throw std::invalid_argument("Empty constraint row");
    c.offsets.push_back(c.values.size());
    impl_->drive.push_back(scale);
    return index;
}
int LinearConstraints::Prescribe(int dof, Real scale) {
    return AddRow({{dof, 1}}, scale);
}
void LinearConstraints::TieNodes(int a, int b) {
    for (int d = 0; d < 3; ++d)
        AddRow({{a + d, 1}, {b + d, -1}});
}
void LinearConstraints::TieNodeToFace(int node, const std::vector<int>& face, const VectorXR& weights) {
    if (face.empty() || face.size() != static_cast<size_t>(weights.size()) || !weights.allFinite() ||
        std::abs(weights.sum() - 1) > 1e-8)
        throw std::invalid_argument("Invalid face interpolation");
    for (int d = 0; d < 3; ++d) {
        ConstraintRow row{{node + d, 1}};
        for (int a = 0; a < weights.size(); ++a)
            row.emplace_back(face[a] + d, -weights(a));
        AddRow(row);
    }
}
void LinearConstraints::AttachNodeToRigidBody(int node, int body, const Real3& r) {
    Matrix3R skew;
    skew << 0, -r(2), r(1), r(2), 0, -r(0), -r(1), r(0), 0;
    for (int d = 0; d < 3; ++d) {
        ConstraintRow row{{node + d, 1}, {body + d, -1}};
        for (int a = 0; a < 3; ++a)
            if (skew(d, a) != 0)
                row.emplace_back(body + 3 + a, skew(d, a));
        AddRow(row);
    }
}
void LinearConstraints::Prepare(const VectorXR& mass) {
    if (impl_->prepared)
        throw std::logic_error("Constraints already prepared");
    if (mass.size() == 0 || !mass.allFinite() || mass.minCoeff() <= 0)
        throw std::invalid_argument("Positive inverse mass required");
    const auto& c = impl_->matrix;
    int n = c.Rows();
    std::vector<int> parent(n), owner(mass.size(), -1);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int a) {
        while (parent[a] != a) {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        return a;
    };
    for (int i = 0; i < n; ++i)
        for (int k = c.offsets[i]; k < c.offsets[i + 1]; ++k) {
            int dof = c.columns[k];
            if (dof >= mass.size())
                throw std::out_of_range("Constraint DOF out of range");
            if (owner[dof] >= 0)
                parent[root(i)] = root(owner[dof]);
            owner[dof] = i;
        }
    std::map<int, std::vector<int>> components;
    for (int i = 0; i < n; ++i)
        components[root(i)].push_back(i);
    std::vector<ConstraintRow> inverse_rows(n);
    for (const auto& component : components) {
        const auto& rows = component.second;
        std::map<int, int> local;
        for (int row : rows)
            for (int k = c.offsets[row]; k < c.offsets[row + 1]; ++k)
                if (!local.count(c.columns[k]))
                    local[c.columns[k]] = local.size();
        std::vector<Eigen::Triplet<Real>> entries;
        for (int i = 0; i < static_cast<int>(rows.size()); ++i)
            for (int k = c.offsets[rows[i]]; k < c.offsets[rows[i] + 1]; ++k)
                entries.emplace_back(i, local.at(c.columns[k]), c.values[k] * std::sqrt(mass(c.columns[k])));
        Eigen::SparseMatrix<Real> cm(rows.size(), local.size());
        cm.setFromTriplets(entries.begin(), entries.end());
        Eigen::SparseMatrix<Real> gram = cm * cm.transpose();
        VectorXR scale = VectorXR(gram.diagonal()).array().sqrt().inverse();
        Eigen::SparseMatrix<Real> normalized = scale.asDiagonal() * gram * scale.asDiagonal();
        auto block = std::make_unique<Impl::LargeBlock>();
        block->rows = rows;
        block->scale = scale;
        block->factor.compute(normalized);
        if (block->factor.info() != Eigen::Success || block->factor.vectorD().minCoeff() < 1e-12)
            throw std::invalid_argument("Dependent or conflicting constraint rows");
        if (rows.size() > 512) {
            impl_->large.push_back(std::move(block));
            continue;
        }
        MatrixXR inverse =
            scale.asDiagonal() * block->factor.solve(MatrixXR::Identity(rows.size(), rows.size())) * scale.asDiagonal();
        if ((gram * inverse - MatrixXR::Identity(rows.size(), rows.size())).cwiseAbs().maxCoeff() > 1e-7)
            throw std::runtime_error("Inaccurate constraint factorization");
        for (int i = 0; i < inverse.rows(); ++i)
            for (int j = 0; j < inverse.cols(); ++j)
                inverse_rows[rows[i]].emplace_back(rows[j], inverse(i, j));
    }
    impl_->inverse = HostCsrMatrix{};
    for (const auto& row : inverse_rows) {
        for (auto [j, v] : row) {
            impl_->inverse.columns.push_back(j);
            impl_->inverse.values.push_back(v);
        }
        impl_->inverse.offsets.push_back(impl_->inverse.values.size());
    }
    impl_->prepared = true;
}
void LinearConstraints::SolveLargeBlocks(const VectorXR& rhs, VectorXR& out) const {
    if (!impl_->prepared || rhs.size() != impl_->matrix.Rows() || out.size() != rhs.size())
        throw std::invalid_argument("Constraint solve dimensions");
    for (const auto& block : impl_->large) {
        VectorXR local(block->rows.size());
        for (int i = 0; i < local.size(); ++i)
            local(i) = block->scale(i) * rhs(block->rows[i]);
        VectorXR solution = block->scale.asDiagonal() * block->factor.solve(local);
        for (int i = 0; i < local.size(); ++i)
            out(block->rows[i]) = solution(i);
    }
}
VectorXR LinearConstraints::SolveImpulse(const VectorXR& rhs) const {
    if (!impl_->prepared || rhs.size() != impl_->matrix.Rows())
        throw std::invalid_argument("Constraint solve dimensions");
    VectorXR out = VectorXR::Zero(rhs.size());
    const auto& a = impl_->inverse;
    for (int i = 0; i < a.Rows(); ++i)
        for (int k = a.offsets[i]; k < a.offsets[i + 1]; ++k)
            out(i) += a.values[k] * rhs(a.columns[k]);
    SolveLargeBlocks(rhs, out);
    return out;
}
bool LinearConstraints::HasLargeBlocks() const {
    return !impl_->large.empty();
}
Real LinearConstraints::PositionError(const VectorXR& u, Real target) const {
    Real error = 0;
    const auto& c = impl_->matrix;
    for (int i = 0; i < c.Rows(); ++i) {
        Real r = -target * impl_->drive[i];
        for (int k = c.offsets[i]; k < c.offsets[i + 1]; ++k) {
            if (c.columns[k] >= u.size())
                throw std::out_of_range("Constraint position dimensions");
            r += c.values[k] * u(c.columns[k]);
        }
        error = std::max(error, std::abs(r));
    }
    return error;
}
const HostCsrMatrix& LinearConstraints::Matrix() const {
    return impl_->matrix;
}
const HostCsrMatrix& LinearConstraints::SmallBlockInverse() const {
    return impl_->inverse;
}
const std::vector<Real>& LinearConstraints::TargetScales() const {
    return impl_->drive;
}
}  // namespace feris
