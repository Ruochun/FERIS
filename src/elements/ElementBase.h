/*==============================================================
 *==============================================================
 * Project: RoboDyna
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ElementBase.h
 * Brief:   Declares the ElementBase class and common interfaces shared by
 *          concrete GPU element data types (ANCF3243, ANCF3443, FEAT10).
 *          Provides virtual accessors for coefficients, beams/elements,
 *          mass/force/constraint storage, and type tagging used by the
 *          synchronized solvers and collision modules.
 *==============================================================
 *==============================================================*/

#pragma once

#include <vector>
#include "../types.h"

namespace tlfea {

enum ElementType { TYPE_3243, TYPE_3443, TYPE_T10, TYPE_T4, TYPE_LDPM4, TYPE_LDPM_TET4 };

class ElementBase {
  public:
    ElementType type;

    virtual ~ElementBase() {}

    ElementBase* d_data;

    // Do not use virtual function in solver class
    // CUDA cannot use virtual function
    virtual __host__ __device__ int get_n_beam() const = 0;
    virtual __host__ __device__ int get_n_coef() const = 0;

    // Core computation functions (actually implemented and used)
    virtual void CalcMassMatrix() = 0;
    virtual void CalcInternalForce() = 0;
    virtual void CalcConstraintData() = 0;
    virtual void CalcP() = 0;
    virtual void RetrieveInternalForceToCPU(VectorXR& internal_force) = 0;
    virtual void RetrieveConstraintDataToCPU(VectorXR& constraint) = 0;
    virtual void RetrieveConstraintJacobianToCPU(MatrixXR& constraint_jac) = 0;
    virtual void RetrievePositionToCPU(VectorXR& x12, VectorXR& y12, VectorXR& z12) = 0;
    virtual void RetrieveDeformationGradientToCPU(std::vector<std::vector<MatrixXR>>& deformation_gradient) = 0;
    virtual void RetrievePFromFToCPU(std::vector<std::vector<MatrixXR>>& p_from_F) = 0;
};

}  // namespace tlfea