/*==============================================================
 *==============================================================
 * Project: FERIS
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 * File:    LeapfrogSolver.cuh
 * Brief:   Declares the LeapfrogSolver, a second-order explicit
 *          central-difference (leapfrog) time integrator for solid
 *          mechanics simulations. Owns GPU buffers for half-step nodal
 *          velocities and a lumped (row-summed) mass vector. Supports
 *          ANCF3243, ANCF3443, FEAT10, FEAT4, and LDPM_TET4 element
 *          types, with primary focus on LDPM_TET4 for
 *          particle-scale simulations with 6 DOFs per particle.
 *
 *          Supported element types: TYPE_3243, TYPE_3443, TYPE_T10,
 *          TYPE_T4, TYPE_LDPM_TET4.
 *          See docs/SOLVER_ELEMENT_COMPATIBILITY.md for the full matrix.
 *==============================================================
 *==============================================================*/

#pragma once

#include <MoPhiEssentials.h>

#include "../elements/ANCF3243Data.cuh"
#include "../elements/ANCF3443Data.cuh"
#include "../elements/ElementBase.h"
#include "../elements/FEAT10Data.cuh"
#include "../elements/FEAT4Data.cuh"
#include "../elements/LDPMTet4Data.cuh"
#include "../types.h"
#include "../utils/cuda_utils.h"
#include "../utils/quadrature_utils.h"
#include "SolverBase.h"

namespace feris {

// Parameters for the leapfrog (central-difference) explicit integrator.
struct LeapfrogParams {
    // Physical simulation time step (seconds).
    Real time_step;
};

// LeapfrogSolver implements the explicit central-difference (leapfrog) time
// integration scheme for nonlinear solid mechanics:
//
//   Given x_n (full-step positions) and v_{n-1/2} (half-step velocities):
//     1. Compute P = P(x_n, v_{n-1/2})  [stress, including viscous damping]
//     2. Assemble f_int(x_n) from P
//     3. a_n = M_lump^{-1} * (f_ext - f_int)
//     4. v_{n+1/2} = v_{n-1/2} + dt * a_n
//     5. Enforce BCs: zero velocity at fixed nodes;
//                    set prescribed velocity at driven nodes (SetPrescribedVelocityBC)
//     6. x_{n+1} = x_n + dt * v_{n+1/2}
//
// The lumped (row-sum) mass matrix is computed once in Setup() from the
// consistent CSR mass matrix that the element has already assembled.
// Each call to Solve() advances the simulation by one time step.
class LeapfrogSolver : public SolverBase {
  public:
    LeapfrogSolver(ElementBase* data) : n_coef_(data->get_n_coef()), n_beam_(data->get_n_beam()) {
        // Validate: all element types are supported; guard against unknown future types.
        switch (data->type) {
            case TYPE_3243:
            case TYPE_3443:
            case TYPE_T10:
            case TYPE_T4:
            case TYPE_LDPM_TET4:
                break;
            default:
                MOPHI_ERROR("LeapfrogSolver: unknown element type %s.", ElementTypeToString(data->type));
                return;  // unreachable; MOPHI_ERROR is fatal
        }

        type_ = data->type;
        data->PrepareSolverData();       // no-op for FEAT/LDPM; calls CalcDsDuPre for ANCF
        d_data_ = data->GetDevicePtr();  // device-side copy pointer
        auto info = GetElementDispatchInfo(type_);
        n_total_qp_ = info.n_total_qp;
        n_shape_ = info.n_shape;

        cudaMalloc(&d_time_step_, sizeof(Real));
        cudaMalloc(&d_leapfrog_solver_, sizeof(LeapfrogSolver));

        // For TYPE_LDPM_TET4 the velocity array holds both translational and
        // rotational half-step velocities:
        //   [v_trans (3*n_coef_) | v_rot (3*n_coef_)]
        // All other types use the standard 3*n_coef_ translational layout.
        const size_t v_size =
            (type_ == TYPE_LDPM_TET4) ? static_cast<size_t>(n_coef_) * 6 : static_cast<size_t>(n_coef_) * 3;
        da_v_.resize(v_size);
        da_v_.BindDevicePointer(&d_v_);

        // For TYPE_LDPM_TET4 the mass array holds both translational lumped
        // mass and rotational inertia:
        //   [m_trans (n_coef_) | I_rot (n_coef_)]
        const size_t m_size =
            (type_ == TYPE_LDPM_TET4) ? static_cast<size_t>(n_coef_) * 2 : static_cast<size_t>(n_coef_);
        da_mass_lump_.resize(m_size);
        da_mass_lump_.BindDevicePointer(&d_mass_lump_);
    }

    ~LeapfrogSolver() {
        da_v_.free();
        da_mass_lump_.free();
        da_pvel_nodes_.free();
        da_pvel_values_.free();

        cudaFree(d_time_step_);
        cudaFree(d_leapfrog_solver_);
    }

    // Copy time step to device and zero nodal velocities to start from rest.
    void SetParameters(void* params) override {
        LeapfrogParams* p = static_cast<LeapfrogParams*>(params);
        cudaMemcpy(d_time_step_, &p->time_step, sizeof(Real), cudaMemcpyHostToDevice);

        da_v_.SetVal(Real(0));
        da_v_.ToDevice();
    }

    // Copy struct mirror to device and compute the lumped mass from the
    // pre-assembled CSR consistent mass matrix stored in the element data.
    // Must be called after the element's CalcMassMatrix() and after
    // SetParameters().
    void Setup();

#if defined(__CUDACC__)
    // Half-step nodal velocity vector (length 3*n_coef_).
    __device__ Map<VectorXR> v() {
        return Map<VectorXR>(d_v_, n_coef_ * 3);
    }

    // Lumped nodal mass vector (length n_coef_).
    __device__ Map<VectorXR> mass_lump() {
        return Map<VectorXR>(d_mass_lump_, n_coef_);
    }

    // Half-step rotational velocity vector for TYPE_LDPM_TET4 (length 3*n_coef_).
    // Stored immediately after the translational velocities in da_v_.
    __device__ Map<VectorXR> v_rot() {
        return Map<VectorXR>(d_v_ + n_coef_ * 3, n_coef_ * 3);
    }

    // Per-node rotational inertia for TYPE_LDPM_TET4 (length n_coef_).
    // Stored immediately after the translational masses in da_mass_lump_.
    __device__ Map<VectorXR> mass_inertia() {
        return Map<VectorXR>(d_mass_lump_ + n_coef_, n_coef_);
    }

    __device__ Real solver_time_step() const {
        return *d_time_step_;
    }
#endif

    __host__ __device__ int get_n_coef() const {
        return n_coef_;
    }
    __host__ __device__ int get_n_beam() const {
        return n_beam_;
    }
    __host__ __device__ int get_n_total_qp() const {
        return n_total_qp_;
    }
    __host__ __device__ int get_n_shape() const {
        return n_shape_;
    }

    // Seed specific nodal translational velocities into the leapfrog velocity
    // array (da_v_).  Call after SetParameters() (which zeroes all velocities)
    // and before InitialHalfKick() or Solve() to set non-zero initial
    // conditions.
    //
    // node_indices  – global node indices, length n.
    // vel_values    – one Real3 velocity per node index.
    //
    // Note: this writes only the translational part (first 3*n_coef_ entries of
    // da_v_).  For TYPE_LDPM_TET4, rotational half-step velocities stored in
    // the second 3*n_coef_ entries are left unchanged.
    void SetNodalVelocity(const VectorXi& node_indices, const VectorReal3& vel_values) {
        const int n = static_cast<int>(node_indices.size());
        if (static_cast<int>(vel_values.size()) != n) {
            MOPHI_ERROR("LeapfrogSolver::SetNodalVelocity: vel_values.size() (%d) must equal "
                        "node_indices.size() (%d).",
                        static_cast<int>(vel_values.size()), n);
            return;
        }
        Real* h_v = da_v_.host();
        for (int i = 0; i < n; ++i) {
            const int node = node_indices(i);
            if (node < 0 || node >= n_coef_) {
                MOPHI_ERROR("LeapfrogSolver::SetNodalVelocity: node index %d out of range [0, %d).",
                            node, n_coef_);
                return;
            }
            const Vector3R& v = vel_values[static_cast<size_t>(i)];
            h_v[node * 3 + 0] = v(0);
            h_v[node * 3 + 1] = v(1);
            h_v[node * 3 + 2] = v(2);
        }
        da_v_.ToDevice();
    }

    // Backward-compatible overload (flat 3*n layout).
    void SetNodalVelocity(const VectorXi& node_indices, const VectorXR& vel_values_3n) {
        VectorReal3 vel_values;
        if (!UnflattenVectorReal3(vel_values_3n, vel_values)) {
            MOPHI_ERROR("LeapfrogSolver::SetNodalVelocity: flat velocity size (%d) is not divisible by 3.",
                        static_cast<int>(vel_values_3n.size()));
            return;
        }
        SetNodalVelocity(node_indices, vel_values);
    }

    // Register a set of nodes whose translational velocity is re-imposed to a
    // constant prescribed value after every velocity update (step 5 of each
    // leapfrog step and in InitialHalfKick / FinalHalfKick).  The prescribed
    // values are also applied in InitialHalfKick so that driven nodes keep
    // their prescribed velocity after the half-kick.
    //
    // node_indices  – global node indices, length n.
    // vel_values    – one Real3 velocity per node index.
    //
    // Typical use (velocity-driven tensile test):
    //   1. Call SetNodalFixed for truly fixed (clamped) nodes.
    //   2. Call SetPrescribedVelocityBC for driven nodes with their target v_z.
    //   3. Call SetNodalVelocity with the same values to seed da_v_.
    //   4. Call InitialHalfKick() to stagger the initial velocity correctly.
    //   5. Run Solve() — positions of driven nodes advance naturally via
    //      x += dt * v without any external position-update call.
    //
    // Must be called after Setup() and before the first Solve().
    // Replaces the old AdvanceDrivenNodesZ() position-update pattern.
    void SetPrescribedVelocityBC(const VectorXi& node_indices, const VectorReal3& vel_values) {
        const int n = static_cast<int>(node_indices.size());
        if (static_cast<int>(vel_values.size()) != n) {
            MOPHI_ERROR("LeapfrogSolver::SetPrescribedVelocityBC: vel_values.size() (%d) must equal "
                        "node_indices.size() (%d).",
                        static_cast<int>(vel_values.size()), n);
            return;
        }
        for (int i = 0; i < n; ++i) {
            const int node = node_indices(i);
            if (node < 0 || node >= n_coef_) {
                MOPHI_ERROR("LeapfrogSolver::SetPrescribedVelocityBC: node index %d out of range [0, %d).",
                            node, n_coef_);
                return;
            }
        }
        n_prescribed_vel_ = n;
        da_pvel_nodes_.resize(n_prescribed_vel_);
        da_pvel_nodes_.BindDevicePointer(&d_pvel_nodes_);
        da_pvel_values_.resize(n_prescribed_vel_ * 3);
        da_pvel_values_.BindDevicePointer(&d_pvel_values_);
        std::copy(node_indices.data(), node_indices.data() + n_prescribed_vel_, da_pvel_nodes_.host());
        for (int i = 0; i < n_prescribed_vel_; ++i) {
            const Vector3R& v = vel_values[static_cast<size_t>(i)];
            da_pvel_values_.host()[i * 3 + 0] = v(0);
            da_pvel_values_.host()[i * 3 + 1] = v(1);
            da_pvel_values_.host()[i * 3 + 2] = v(2);
        }
        da_pvel_nodes_.ToDevice();
        da_pvel_values_.ToDevice();
    }

    // Backward-compatible overload (flat 3*n layout).
    void SetPrescribedVelocityBC(const VectorXi& node_indices, const VectorXR& vel_values_3n) {
        VectorReal3 vel_values;
        if (!UnflattenVectorReal3(vel_values_3n, vel_values)) {
            MOPHI_ERROR("LeapfrogSolver::SetPrescribedVelocityBC: flat velocity size (%d) is not divisible by 3.",
                        static_cast<int>(vel_values_3n.size()));
            return;
        }
        SetPrescribedVelocityBC(node_indices, vel_values);
    }

    // Advance one leapfrog time step.
    void OneStepLeapfrog();

    void Solve() override {
        OneStepLeapfrog();
    }

    // Backward half-kick to initialize the staggered leapfrog from a
    // full-step initial velocity v_0 stored in da_v_:
    //
    //   v_{-1/2} = v_0 - (dt/2) * M_lump^{-1} * (f_ext - f_int(x_0))
    //
    // Call after Setup() when da_v_ has been seeded with non-zero physical
    // initial velocities.  For rest-start simulations this is a no-op.
    void InitialHalfKick();

    // Forward half-kick to recover a synchronized full-step velocity
    // after the last Solve():
    //
    //   v_N = v_{N-1/2} + (dt/2) * M_lump^{-1} * (f_ext - f_int(x_N))
    //
    // Call after the final Solve() when a velocity output synchronized
    // with the current position is required (e.g. kinetic-energy reporting).
    void FinalHalfKick();

  private:
    ElementType type_;
    // Device pointer to the element data struct (GPU_FEAT4_Data, etc.).
    // Owned by the ElementBase object; not freed here.
    ElementBase* d_data_;
    // Device-side mirror of this solver struct, copied via cudaMemcpy in Setup().
    // Passed directly to GPU kernels so they can access all device pointers and
    // scalar fields without re-deriving the host address.
    LeapfrogSolver* d_leapfrog_solver_;
    int n_total_qp_, n_shape_;
    int n_coef_, n_beam_;

    // DualArrays for long arrays (manage both pinned host and device memory).
    // d_v_: half-step nodal velocities, length 3*n_coef_
    mophi::DualArray<Real> da_v_;
    // d_mass_lump_: lumped (diagonal) nodal mass, one scalar per node
    mophi::DualArray<Real> da_mass_lump_;

    // Raw device pointers for GPU kernel access (bound to DualArrays above).
    Real* d_v_;
    Real* d_mass_lump_;

    // Physical simulation time step stored on device for kernel access.
    Real* d_time_step_;

    // Prescribed-velocity BC: nodes whose translational velocity is clamped to
    // a constant value after every velocity update step.
    // n_prescribed_vel_: number of prescribed-velocity nodes (0 = none).
    // da_pvel_nodes_: node indices, length n_prescribed_vel_.
    // da_pvel_values_: velocity values [vx,vy,vz] per node, length 3*n_prescribed_vel_.
    int n_prescribed_vel_ = 0;
    mophi::DualArray<int> da_pvel_nodes_;
    mophi::DualArray<Real> da_pvel_values_;
    int* d_pvel_nodes_ = nullptr;
    Real* d_pvel_values_ = nullptr;

    // Shared implementation for InitialHalfKick() and FinalHalfKick().
    // sign = -1: backward half-kick; sign = +1: forward half-kick.
    void HalfKickImpl(Real sign);
};

}  // namespace feris
