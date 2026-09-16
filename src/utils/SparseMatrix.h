/* Project: FERIS
 * File: SparseMatrix.h
 * Brief: Host CSR storage shared by element operators and constraints.
 */
#pragma once
#include "../types.h"
namespace feris {
struct HostCsrMatrix {
    std::vector<int> offsets{0}, columns;
    std::vector<Real> values;
    int Rows() const { return static_cast<int>(offsets.size()) - 1; }
    Real QuadraticEnergy(const VectorXR& displacement) const;
    Real CriticalTimeStep(const VectorXR& mass) const;
};
}  // namespace feris
