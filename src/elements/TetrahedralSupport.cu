/* Project: FERIS
 * File: TetrahedralSupport.cu
 * Brief: Mesh setup and explicit lumped mass policies for T4/T10.
 */
#include "FEAT4Data.cuh"
#include "FEAT10Data.cuh"
#include <stdexcept>
namespace feris {
void GPU_FEAT4_Data::SetupFromMesh(const VectorReal3& positions, const MatrixXi& connectivity) {
    if (positions.size() != static_cast<size_t>(n_coef) || connectivity.rows() != n_elem || connectivity.cols() != 4)
        throw std::invalid_argument("FEAT4 mesh dimensions");
    if (is_setup)
        throw std::logic_error("Element is already set up");
    if (n_coef <= 0 || n_elem <= 0 || connectivity.minCoeff() < 0 || connectivity.maxCoeff() >= n_coef)
        throw std::invalid_argument("Invalid tetrahedral connectivity");
    for (const auto& p : positions)
        if (!p.allFinite())
            throw std::invalid_argument("Non-finite mesh coordinate");
    Initialize();
    VectorXR x(n_coef), y(n_coef), z(n_coef);
    for (int i = 0; i < n_coef; ++i) {
        x(i) = positions[i](0);
        y(i) = positions[i](1);
        z(i) = positions[i](2);
    }
    Setup(Quadrature::tet1pt_x, Quadrature::tet1pt_y, Quadrature::tet1pt_z, Quadrature::tet1pt_weights, x, y, z,
          connectivity);
    CalcDnDuPre();
}
void GPU_FEAT4_Data::RetrieveLumpedMassToCPU(VectorXR& mass) {
    std::vector<int> offsets, columns;
    std::vector<Real> values;
    RetrieveMassCSRToCPU(offsets, columns, values);
    mass = VectorXR::Zero(n_coef);
    Real total = 0;
    for (int i = 0; i < n_coef; ++i)
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            total += values[j];
            mass(i) += values[j];
        }
}
void GPU_FEAT10_Data::SetupFromMesh(const VectorReal3& positions, const MatrixXi& connectivity) {
    if (positions.size() != static_cast<size_t>(n_coef) || connectivity.rows() != n_elem || connectivity.cols() != 10)
        throw std::invalid_argument("FEAT10 mesh dimensions");
    if (is_setup)
        throw std::logic_error("Element is already set up");
    if (n_coef <= 0 || n_elem <= 0 || connectivity.minCoeff() < 0 || connectivity.maxCoeff() >= n_coef)
        throw std::invalid_argument("Invalid tetrahedral connectivity");
    for (const auto& p : positions)
        if (!p.allFinite())
            throw std::invalid_argument("Non-finite mesh coordinate");
    Initialize();
    VectorXR x(n_coef), y(n_coef), z(n_coef);
    for (int i = 0; i < n_coef; ++i) {
        x(i) = positions[i](0);
        y(i) = positions[i](1);
        z(i) = positions[i](2);
    }
    Setup(Quadrature::tet5pt_x, Quadrature::tet5pt_y, Quadrature::tet5pt_z, Quadrature::tet5pt_weights, x, y, z,
          connectivity);
    CalcDnDuPre();
}
void GPU_FEAT10_Data::RetrieveLumpedMassToCPU(VectorXR& mass) {
    std::vector<int> offsets, columns;
    std::vector<Real> values;
    RetrieveMassCSRToCPU(offsets, columns, values);
    mass = VectorXR::Zero(n_coef);
    Real total = 0;
    for (int i = 0; i < n_coef; ++i)
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            total += values[j];
            if (columns[j] == i)
                mass(i) += values[j];
        }
    // Positive diagonal scaling avoids the negative vertex row sums of T10.
    if (!mass.allFinite() || mass.minCoeff() <= 0 || total <= 0)
        throw std::runtime_error("Invalid T10 consistent mass diagonal");
    mass *= total / mass.sum();
}
}  // namespace feris
