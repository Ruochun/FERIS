# examples/CMakeLists.txt
#
# Three categories of examples:
#   beam_simulations/  – cantilever-beam problems covering all element types
#                        and all supported solvers.
#   solver_tests/      – focused tests of the gradient-based solvers
#                        (Nesterov, AdamW) on a simple solid block.
#   ldmp_tests/        – LDPM demos and benchmark cases, including Dogbone
#                        specimen simulations and single-tet unit tests.

add_subdirectory(beam_simulations)
add_subdirectory(solver_tests)
add_subdirectory(ldmp_tests)
