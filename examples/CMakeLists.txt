# examples/CMakeLists.txt
#
# Two categories of examples:
#   beam_simulations/  – cantilever-beam problems covering all element types
#                        and all supported solvers.
#   solver_tests/      – focused tests of the gradient-based solvers
#                        (Nesterov, AdamW) on a simple solid block.

add_subdirectory(beam_simulations)
add_subdirectory(solver_tests)
