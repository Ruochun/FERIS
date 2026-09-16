/* Project: FERIS
 * File: DynamicsSystem.cu
 * Brief: Element registration, rigid-body inertia, and system state ownership.
 */
#include "DynamicsSystemInternal.cuh"
#include <stdexcept>
#include <cmath>
namespace feris {
namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
}  // namespace
RigidBody RigidBody::Box(const Real3& size, Real density) {
    if (!size.allFinite() || size.minCoeff() <= 0 || !std::isfinite(density) || density <= 0)
        throw std::invalid_argument("Invalid rigid box");
    Real m = size.prod() * density;
    return {m, Real3(m * (size(1) * size(1) + size(2) * size(2)) / 12, m * (size(0) * size(0) + size(2) * size(2)) / 12,
                     m * (size(0) * size(0) + size(1) * size(1)) / 12)};
}
DynamicsSystem::DynamicsSystem() : impl_(new Impl) {}
DynamicsSystem::~DynamicsSystem() = default;
int DynamicsSystem::AddElement(ElementBase& element) {
    if (impl_->frozen)
        throw std::logic_error("Dynamics system is frozen");
    for (const auto& group : impl_->groups)
        if (group.element == &element)
            throw std::invalid_argument("Element already registered");
    int fixed = 0;
    switch (element.type) {
        case TYPE_T4:
            fixed = static_cast<GPU_FEAT4_Data&>(element).get_n_constraint();
            break;
        case TYPE_T10:
            fixed = static_cast<GPU_FEAT10_Data&>(element).get_n_constraint();
            break;
        case TYPE_LDPM_TET4:
            fixed = static_cast<GPU_LDPMTet4_Data&>(element).n_constraint;
            break;
        default:
            MOPHI_ERROR(
                "CoupledLeapfrogSolver: unsupported element type %s. Supported: TYPE_T4, TYPE_T10, TYPE_LDPM_TET4. Use "
                "LeapfrogSolver for uncoupled ANCF.",
                ElementTypeToString(element.type));
            return -1;
    }
    if (fixed)
        throw std::invalid_argument("Register fixed DOFs in DynamicsSystem constraints, not on the element");
    auto solver = std::make_unique<LeapfrogSolver>(&element);
    LeapfrogParams initial{1e-8};
    solver->SetParameters(&initial);
    solver->Setup();
    int count = element.get_n_coef(), offset = impl_->inverse_mass.size();
    VectorXR masses(element.type == TYPE_LDPM_TET4 ? 2 * count : count);
    Check(cudaMemcpy(masses.data(), solver->GetLumpedMassDevicePtr(), masses.size() * sizeof(Real),
                     cudaMemcpyDeviceToHost));
    if (!masses.allFinite() || masses.minCoeff() <= 0)
        throw std::invalid_argument("Coupled dynamics requires positive nodal mass and inertia");
    impl_->inverse_mass.conservativeResize(offset + 3 * count);
    for (int i = 0; i < count; ++i)
        impl_->inverse_mass.segment<3>(offset + 3 * i).setConstant(1 / masses(i));
    VectorXR x, y, z;
    element.RetrievePositionToCPU(x, y, z);
    VectorReal3 reference(count);
    for (int i = 0; i < count; ++i)
        reference[i] = Real3(x(i), y(i), z(i));
    VectorXR inertia;
    if (element.type == TYPE_LDPM_TET4)
        inertia = masses.tail(count);
    impl_->groups.push_back({&element, offset, std::move(solver), std::move(reference), std::move(inertia)});
    return offset;
}
int DynamicsSystem::AddRigidBody(const RigidBody& body) {
    if (impl_->frozen)
        throw std::logic_error("Dynamics system is frozen");
    if (!std::isfinite(body.mass) || body.mass <= 0 || !body.inertia.allFinite() || body.inertia.minCoeff() <= 0 ||
        !body.force.allFinite() || !body.moment.allFinite())
        throw std::invalid_argument("Invalid rigid body mass/inertia/load");
    int offset = impl_->inverse_mass.size();
    impl_->inverse_mass.conservativeResize(offset + 6);
    impl_->inverse_mass.segment<3>(offset).setConstant(1 / body.mass);
    impl_->inverse_mass.segment<3>(offset + 3) = body.inertia.cwiseInverse();
    impl_->bodies.push_back({body, offset});
    return offset;
}
LinearConstraints& DynamicsSystem::Constraints() {
    return impl_->constraints;
}
const VectorXR& DynamicsSystem::InverseMass() const {
    return impl_->inverse_mass;
}
}  // namespace feris
