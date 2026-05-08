/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    types.h
 * Brief:   Defines the Real type as an alias for double and provides
 *          type aliases for Eigen matrix and vector types using Real.
 *          Also provides a user-facing 3D-vector container alias
 *          (`Real3` + `VectorReal3`) and helpers to convert
 *          between VectorReal3 and flattened 3*n VectorXR arrays.
 *          This allows easy switching between different floating-point
 *          precisions throughout the codebase.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>

namespace feris {

// Define Real as the primary floating-point type for the project
typedef double Real;

// Forward declaration for DynamicMatrix size and storage options
static constexpr int DynamicMatrix = Eigen::Dynamic;
static constexpr int RowMajorMatrix = Eigen::RowMajor;
static constexpr int ColMajorMatrix = Eigen::ColMajor;

// Wrap Eigen::Matrix under feris namespace
// This allows future flexibility to change the underlying implementation
template <typename Scalar, int Rows, int Cols, int Options = 0>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options>;

// Wrap Eigen::Map under feris namespace
// This allows future flexibility to change the underlying implementation
template <typename PlainObjectType, int MapOptions = Eigen::Unaligned, typename StrideType = Eigen::Stride<0, 0>>
using Map = Eigen::Map<PlainObjectType, MapOptions, StrideType>;

// Type aliases using Real and our Matrix template
typedef Matrix<Real, DynamicMatrix, DynamicMatrix> MatrixXR;
typedef Matrix<Real, DynamicMatrix, 1> VectorXR;
typedef Matrix<Real, 3, 3> Matrix3R;
typedef Matrix<Real, 3, 1> Vector3R;
typedef Vector3R Real3;
typedef Matrix<int, DynamicMatrix, DynamicMatrix> MatrixXi;
typedef Matrix<int, DynamicMatrix, 1> VectorXi;
typedef std::vector<Real3, Eigen::aligned_allocator<Real3>> VectorReal3;

inline VectorXR FlattenVectorReal3(const VectorReal3& v3) {
    VectorXR flat(static_cast<int>(v3.size()) * 3);
    for (int i = 0; i < static_cast<int>(v3.size()); ++i) {
        flat(i * 3 + 0) = v3[static_cast<size_t>(i)](0);
        flat(i * 3 + 1) = v3[static_cast<size_t>(i)](1);
        flat(i * 3 + 2) = v3[static_cast<size_t>(i)](2);
    }
    return flat;
}

inline bool UnflattenVectorReal3(const VectorXR& flat, VectorReal3& v3) {
    if (flat.size() % 3 != 0) {
        return false;
    }
    const int n = static_cast<int>(flat.size()) / 3;
    v3.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        v3[static_cast<size_t>(i)](0) = flat(i * 3 + 0);
        v3[static_cast<size_t>(i)](1) = flat(i * 3 + 1);
        v3[static_cast<size_t>(i)](2) = flat(i * 3 + 2);
    }
    return true;
}

}  // namespace feris
