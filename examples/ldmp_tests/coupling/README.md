# Coupled dynamics regression checks

`coupled_dynamics_test` checks the shared library mechanisms used by mixed
element examples, independently of the TPB specimen geometry. A CUDA GPU is
required. It verifies T4/T10 linearized force and stiffness consistency, rigid
modes, positive lumped mass, prescribed motion, basic/coupled leapfrog
equivalence, rigid-body force transfer, and a 521-row sparse constraint group.

After [configuring the build](../../../docs/BUILDING.md), run from the
repository root:

```bash
cmake --build build --target coupled_dynamics_test -j4
./build/bin/coupled_dynamics_test
```

The program reports success or exits with an error; it writes no output files.
The complementary host constraint and mesh checks live with
[TPB](../tpb/README.md) in `tpb_model_test`.
