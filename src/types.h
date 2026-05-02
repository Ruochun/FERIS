/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    types.h
 * Brief:   Defines the Real type as an alias for double and provides
 *          type aliases for Eigen matrix and vector types using Real.
 *          This allows easy switching between different floating-point
 *          precisions throughout the codebase.
 *==============================================================
 *==============================================================*/

#pragma once

#include <Eigen/Dense>

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
typedef Matrix<int, DynamicMatrix, DynamicMatrix> MatrixXi;
typedef Matrix<int, DynamicMatrix, 1> VectorXi;

}  // namespace feris
