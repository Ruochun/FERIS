# Auxiliary coupled dynamics system

This directory supplies the system-level bookkeeping needed to connect FERIS
element groups and simple rigid fixtures. It supports the project's FEA-centric
architecture while allowing mixed FEA/LDPM examples such as TPB. Constitutive
laws, element forces, stiffness, and mass policies remain in `src/elements/`.
Rigid bodies are auxiliary system components rather than a new `ElementType`.

`DynamicsSystem.h` is the public registration interface. `AddElement` accepts
T4, T10, and LDPM groups and returns their first generalized translational DOF.
`AddRigidBody` adds six coordinates: translation followed by small rotation.
`RigidBody::Box` calculates mass and principal inertia from dimensions and
density. Bodies can carry constant forces and moments.

`DynamicsSystem.cu` collects positive inverse masses and reference positions.
`DynamicsSystemInternal.cuh` holds the private group records, including one
owned **basic `LeapfrogSolver` per element group**. Elements are borrowed and
must outlive the system; the system must outlive its coupled solver. Configure
element material and mass before registration, and keep topology and mass
unchanged afterward. Creating the coupled solver freezes registration and
prepares the [system constraints](../constraints/README.md). Register coupled
fixed/prescribed DOFs there rather than on individual elements.

`CoupledLeapfrogSolver`, in `src/solvers/`, owns the common time step, packed
coupling buffers, and rigid-body integration state. It starts from rest using
the basic solver's `InitialHalfKick()`. Each step then:

1. Calls each basic solver's private `Kick()` for constitutive updates, internal
   force assembly, and half-step velocity updates; kicks rigid bodies as well.
2. Projects the combined velocities through the system constraints.
3. Calls each basic solver's private `Drift()` and advances rigid bodies.

This reuse explains the coupled solver's small implementation: element dispatch,
FEA forces, LDPM damage and particle rotations, and element integration kernels
already live in the basic solver and element implementations. Standalone
leapfrog calls the same kick and drift stages consecutively. The coupled solver
inserts projection between them, avoiding a separate integration implementation.

Current coupling uses element translations and small-rotation rigid bodies.
LDPM particle rotations remain dynamic inside the basic solver but are not
coupling coordinates. ANCF coupling, evolving contact, and finite rigid-body
rotations are outside the present scope. See the
[architecture guide](../../docs/ARCHITECTURE.md#coupled-explicit-dynamics) and
[TPB example](../../examples/ldmp_tests/tpb/README.md) for context.
