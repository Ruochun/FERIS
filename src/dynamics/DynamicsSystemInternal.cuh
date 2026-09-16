/* Project: FERIS
 * File: DynamicsSystemInternal.cuh
 * Brief: Internal element integration storage; not a public coupling API.
 */
#pragma once
#include "DynamicsSystem.h"
#include "../solvers/LeapfrogSolver.cuh"
namespace feris {
struct DynamicsSystem::Impl {
    struct Group {
        ElementBase* element;
        int offset;
        std::unique_ptr<LeapfrogSolver> solver;
        VectorReal3 reference;
        VectorXR inertia;
    };
    struct Body {
        RigidBody properties;
        int offset;
    };
    std::vector<Group> groups;
    std::vector<Body> bodies;
    VectorXR inverse_mass;
    LinearConstraints constraints;
    bool frozen = false;
};
}  // namespace feris
