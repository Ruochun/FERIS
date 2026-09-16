/* Project: FERIS
 * File: SparseMatrix.cc
 * Brief: Host CSR quadratic forms and conservative elastic time-step bounds.
 */
#include "SparseMatrix.h"
#include <cmath>
#include <stdexcept>
namespace feris {
Real HostCsrMatrix::QuadraticEnergy(const VectorXR& u) const {
    if (u.size() != Rows())
        throw std::invalid_argument("CSR displacement size mismatch");
    Real energy = 0;
    for (int i = 0; i < Rows(); ++i)
        for (int j = offsets[i]; j < offsets[i + 1]; ++j)
            energy += Real(0.5) * u(i) * values[j] * u(columns[j]);
    return energy;
}
Real HostCsrMatrix::CriticalTimeStep(const VectorXR& mass) const {
    if (mass.size() != Rows() || !mass.allFinite() || mass.minCoeff() <= 0)
        throw std::invalid_argument("Positive mass required");
    Real bound = 0;
    for (int i = 0; i < Rows(); ++i) {
        Real sum = 0;
        for (int j = offsets[i]; j < offsets[i + 1]; ++j)
            sum += std::abs(values[j]) / std::sqrt(mass(i) * mass(columns[j]));
        bound = std::max(bound, sum);
    }
    if (bound <= 0)
        throw std::invalid_argument("Zero stiffness bound");
    return 2 / std::sqrt(bound);
}
}  // namespace feris
