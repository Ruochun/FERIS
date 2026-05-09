# AGENTS.md — Rules for Coding Agents

This file contains the rules that any automated coding agent (AI assistant,
code-generation bot, etc.) **must** follow when modifying this repository.
Human contributors are also encouraged to follow these guidelines.

---

## 1. Mandatory: Keep the Compatibility Matrix Up to Date

**Every time you add, remove, or change a solver–element pairing, you MUST
update [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](docs/SOLVER_ELEMENT_COMPATIBILITY.md).**

This includes:
- Adding a new element type (`ElementType` enum + concrete class).
- Adding a new solver class.
- Enabling an existing solver to work with an additional element type.
- Disabling or deprecating an existing solver–element pairing.
- Adding or removing a material model and documenting which elements use it.

Failure to update the compatibility matrix is considered an incomplete change.

---

## 2. Adding a New Element Type

When adding a new element type, you must:

1. **Add a new enum value** to `ElementType` in `src/elements/ElementBase.h`.
2. **Implement the concrete element struct** inheriting from `ElementBase`,
   overriding all pure-virtual methods.
3. **Override `GetDevicePtr()`** to return the device-side copy of the struct
   (cast to `ElementBase*`).
4. **Override `PrepareSolverData()`** if the element requires pre-computation
   before being passed to a solver (e.g., shape-function derivative tables).
   Otherwise the no-op default in `ElementBase` is inherited automatically.
5. **Override `IsConstraintSetup()` and `GetConstraintDevicePtr()`** if the
   element supports holonomic constraints.  Otherwise the default (false / nullptr)
   is inherited.
6. **Extend `GetElementDispatchInfo()`** in `ElementBase.h` to return the
   correct `{n_total_qp, n_shape}` values for the new type.
7. **Extend `ElementTypeToString()`** in `ElementBase.h` to return a human-
   readable name for the new type.
8. **Add dispatch branches** in each solver that is intended to support the
   new element type (both the constructor validation and the kernel-launch
   `switch`/`if-else` in the `.cu` implementation file).
9. **Update `src/FEASolver.h`** to include the new element header.
10. **Update [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](docs/SOLVER_ELEMENT_COMPATIBILITY.md)**
    (see Rule 1).

---

## 3. Adding a New Solver

When adding a new solver, you must:

1. **Inherit from `SolverBase`** and implement `Solve()` and `SetParameters()`.
2. **Validate the element type** in the constructor using the helper utilities
   in `ElementBase.h`:
   ```cpp
   // Example for a solver supporting TYPE_T4 and TYPE_T10
   switch (data->type) {
       case TYPE_T4:
       case TYPE_T10:
           break;
       default:
           MOPHI_ERROR("MySolver: unsupported element type %s. "
                       "Supported: TYPE_T4, TYPE_T10.",
                       ElementTypeToString(data->type));
           return;  // unreachable; MOPHI_ERROR is fatal
   }
   ```
3. **Use the provided helpers** `GetElementDispatchInfo()` and
   `data->GetDevicePtr()` / `data->PrepareSolverData()` to avoid duplicating
   element-specific dispatch logic in the constructor.
4. **Add the new solver header** to `src/FEASolver.h`.
5. **Document the solver** in [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](docs/SOLVER_ELEMENT_COMPATIBILITY.md)
   (see Rule 1).

---

## 4. Error Handling

- Solvers must call **`MOPHI_ERROR`** (after which execution does not continue)
  when constructed with an element type they do not support.  Do **not** follow
  `MOPHI_ERROR` with a `throw` — `MOPHI_ERROR` is the project's fatal-error
  mechanism and is sufficient on its own.
- Error messages must name **both the solver and the unsupported element
  type**, and must suggest which solver to use instead when applicable.
- Add a `return;` (or equivalent) after `MOPHI_ERROR` to silence compiler
  unreachable-code warnings; it will never execute.

---

## 5. Code Style

- Follow the existing `.clang-format` configuration.
- After making changes, run `./.format_all` before committing so all formatted 
  source files stay consistent with the repository style.
- File header comments (`Project`, `File`, `Brief`) must be updated when the
  file's purpose or public interface changes significantly.
- Keep the `namespace feris {` wrapping around all library code.
- Prefer `__host__ __device__` accessors over raw member access for portability.
- Add `override` to every virtual method in derived classes.
- For user-facing methods that accept or return collections of 3D quantities
  (for example forces, velocities, moments, or displacements), prefer
  `Real3` / `VectorReal3`-based APIs rather than flat `VectorXR` arrays of
  length `3*n`. Keep flat `VectorXR` layouts for internal storage, solver-side
  buffers, or backward-compatible overloads only.

---

## 6. Documentation

- Architecture-level decisions belong in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- Build instructions belong in [`docs/BUILDING.md`](docs/BUILDING.md).
- The top-level [`README.md`](README.md) overview table (element/solver feature
  matrix) must be kept consistent with
  [`docs/SOLVER_ELEMENT_COMPATIBILITY.md`](docs/SOLVER_ELEMENT_COMPATIBILITY.md).
- Implementation notes for complex algorithms belong in
  [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) or per-element
  `*_implementation.md` files.

---

## 7. Testing

- New element types or solvers should have a corresponding example or test
  program under `examples/`.
- Existing examples must not be broken by changes to the element or solver
  interfaces.

---

## 8. Deprecation

- Deprecated elements/solvers must be clearly marked with `[DEPRECATED]` in
  their file header comment and in the compatibility matrix.
- Do not remove deprecated code without a corresponding update to all call
  sites in `examples/` and documentation.
