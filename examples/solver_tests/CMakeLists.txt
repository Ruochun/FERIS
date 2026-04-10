# examples/solver_tests/CMakeLists.txt
#
# Solver-capability tests.  Each target exercises one of the iterative
# gradient-based solvers (Nesterov, AdamW) on a simple FEAT10 cube mesh.
#
# ┌──────────────────────────┬────────────────┬──────────────────────────────┐
# │ Target                   │ Element type   │ Solver                       │
# ├──────────────────────────┼────────────────┼──────────────────────────────┤
# │ test_feat10_nesterov     │ FEAT10 (TET10) │ SyncedNesterov               │
# │ test_feat10_adamw        │ FEAT10 (TET10) │ SyncedAdamW                  │
# └──────────────────────────┴────────────────┴──────────────────────────────┘

set(SOLVER_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# ── Nesterov accelerated gradient descent ────────────────────────────────────
add_executable(test_feat10_nesterov test_feat10_nesterov.cc)
target_link_libraries(test_feat10_nesterov tlfea_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(test_feat10_nesterov PRIVATE ${SOLVER_TEST_INCLUDE_DIRS})
set_target_properties(test_feat10_nesterov PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── AdamW optimizer-based solver ─────────────────────────────────────────────
add_executable(test_feat10_adamw test_feat10_adamw.cc)
target_link_libraries(test_feat10_adamw tlfea_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(test_feat10_adamw PRIVATE ${SOLVER_TEST_INCLUDE_DIRS})
set_target_properties(test_feat10_adamw PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
