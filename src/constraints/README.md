# Auxiliary linear constraints

This directory provides bilateral connections between element groups and rigid
bodies in FERIS's FEA-centric simulation framework. Constraints describe shared
kinematics independently of the continuum or LDPM constitutive laws; they do
not introduce a new element type or replace existing solver-local constraints.

`LinearConstraints.h` exposes prescribed components, node ties, interpolated
node-to-face ties, and small-rotation node-to-rigid-body attachments. Rows are
stored as a sparse matrix `C` acting on generalized displacements, `C u = g(t)`.
Returned row indices identify reactions. Face weights come from the mesh
utilities; this class only assembles their constraint coefficients.

`LinearConstraints.cc` prepares the mass-weighted operator `G = C M^-1 C^T`.
Rows sharing DOFs are grouped into connected components so overlapping
connections are solved together. Each component is diagonally scaled and
factored with sparse LDLT; dependent or conflicting rows are rejected.
Components of at most 512 rows cache an inverse for GPU application. Larger
components retain their CPU sparse factors, avoiding a dense inverse.

The [coupled solver](../solvers/CoupledLeapfrogSolver.cu) computes
`G impulse = (g_next - g_now)/dt - C velocity_free`, applies
`M^-1 C^T impulse` to half-step velocities, then advances positions. Large
components transfer residuals and impulses between GPU and CPU each step.
Reported constraint reactions are `impulse/dt`.

Prepare constraints after all masses and rows are known; they cannot be changed
after preparation. The current formulation has constant Jacobians and supports
bilateral ties and prescribed motion. Contact, friction, and finite-rotation
constraints require additional implementations.

See the [architecture guide](../../docs/ARCHITECTURE.md#coupled-explicit-dynamics)
for project-level ownership and scope. `tpb_model_test` checks projection,
overlapping components, and rigid ties; `coupled_dynamics_test` also exercises
the GPU application and large-component fallback.
