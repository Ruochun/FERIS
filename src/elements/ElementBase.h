/*==============================================================
 *==============================================================
 * Project: FERIS
 * Author:  Json Zhou
 * Email:   zzhou292@wisc.edu
 * File:    ElementBase.h
 * Brief:   Declares the ElementBase class and common interfaces shared by
 *          all concrete GPU element data types (ANCF3243, ANCF3443, FEAT4,
 *          FEAT10, LDPMTet4). Provides virtual dispatch helpers, size
 *          accessors, and computation interfaces used by solvers.
 *==============================================================
 *==============================================================*/

#pragma once

#include <vector>
#include <MoPhiEssentials.h>
#include "../types.h"

namespace feris {

enum ElementType { TYPE_3243, TYPE_3443, TYPE_T10, TYPE_T4, TYPE_LDPM_TET4 };

// ── Per-element-type dispatch metadata ───────────────────────────────────────
// Used by solver constructors to look up quadrature and shape-function counts
// without repeating per-type if-else chains.

struct ElementDispatchInfo {
    int n_total_qp;  // Total number of quadrature / integration points per element
    int n_shape;     // Number of shape (basis) functions per element
};

// Returns the human-readable name of an ElementType enum value.
inline const char* ElementTypeToString(ElementType type) {
    switch (type) {
        case TYPE_3243:      return "TYPE_3243";
        case TYPE_3443:      return "TYPE_3443";
        case TYPE_T10:       return "TYPE_T10";
        case TYPE_T4:        return "TYPE_T4";
        case TYPE_LDPM_TET4: return "TYPE_LDPM_TET4";
        default:             return "UNKNOWN_TYPE";
    }
}

// Returns {n_total_qp, n_shape} for the given element type.
// Values correspond to the Quadrature namespace constants defined in
// utils/quadrature_utils.h:
//   TYPE_3243:       N_TOTAL_QP_3_2_2 = 12,  N_SHAPE_3243    = 8
//   TYPE_3443:       N_TOTAL_QP_4_4_3 = 48,  N_SHAPE_3443    = 16
//   TYPE_T10:        N_QP_T10_5       = 5,   N_NODE_T10_10   = 10
//   TYPE_T4:         N_QP_T4_1        = 1,   N_NODE_T4_4     = 4
//   TYPE_LDPM_TET4:  1 facet per edge,        2 endpoint nodes per edge
inline ElementDispatchInfo GetElementDispatchInfo(ElementType type) {
    switch (type) {
        case TYPE_3243:      return {12, 8};
        case TYPE_3443:      return {48, 16};
        case TYPE_T10:       return {5,  10};
        case TYPE_T4:        return {1,  4};
        case TYPE_LDPM_TET4: return {1,  2};
        default:
            MOPHI_ERROR("GetElementDispatchInfo: unknown element type %s",
                        ElementTypeToString(type));
            return {0, 0};  // unreachable; MOPHI_ERROR is fatal
    }
}

// ── ElementBase ───────────────────────────────────────────────────────────────

class ElementBase {
  public:
    ElementType type;

    virtual ~ElementBase() {}

    // ── Dispatch helpers ─────────────────────────────────────────────────────
    // These host-only virtual methods let solver constructors avoid per-type
    // if-else chains.  Concrete element classes override the ones they need.

    // Returns the device-side copy of this element struct (cast to ElementBase*).
    // Override in every concrete class to return its own typed d_data pointer.
    virtual ElementBase* GetDevicePtr() = 0;

    // Called by solver constructors to trigger any pre-computation that must
    // happen before the first Solve() (e.g., CalcDsDuPre() for ANCF elements).
    // The default is a no-op; ANCF concrete classes override this.
    virtual void PrepareSolverData() {}

    // Returns true if holonomic constraints have been set up on this element.
    // The default is false; override in elements that support constraints.
    virtual bool IsConstraintSetup() { return false; }

    // Returns a raw device pointer to the pre-assembled constraint data block,
    // or nullptr if constraints have not been set up.
    // Override in elements that support constraints.
    virtual Real* GetConstraintDevicePtr() { return nullptr; }

    // ── Size accessors ───────────────────────────────────────────────────────
    // Do not use virtual functions in GPU kernels — CUDA cannot dispatch virtual
    // calls through device-side pointers safely.  These are host-only.
    virtual __host__ __device__ int get_n_beam() const = 0;
    virtual __host__ __device__ int get_n_coef() const = 0;

    // ── Core computation functions ───────────────────────────────────────────
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

}  // namespace feris