# examples/beam_simulations/CMakeLists.txt
#
# Beam-simulation examples.  Each target exercises a different element type
# and/or solver on a cantilever-beam geometry.
#
# ┌─────────────────────────────┬───────────────────────┬─────────────────────┐
# │ Target                      │ Element type          │ Solver              │
# ├─────────────────────────────┼───────────────────────┼─────────────────────┤
# │ beam_simulation             │ FEAT10 (TET10)        │ SyncedNesterov      │
# │ beam_linear_static          │ FEAT10 (TET10)        │ Linear (cuSPARSE)   │
# │ beam_linear_static_t4       │ FEAT4  (TET4)         │ Linear (cuSPARSE)   │
# │ test_leapfrog_t4            │ FEAT4  (TET4)         │ Leapfrog            │
# │ test_ldpm_tet4              │ LDPMTet4 (TET4, 6-DOF)│ Leapfrog            │
# └─────────────────────────────┴───────────────────────┴─────────────────────┘

set(BEAM_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# ── Dynamic TET10 beam (Nesterov) ─────────────────────────────────────────────
add_executable(beam_simulation beam_simulation.cc)
target_link_libraries(beam_simulation feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(beam_simulation PRIVATE ${BEAM_INCLUDE_DIRS})
set_target_properties(beam_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Static TET10 beam (linear) ────────────────────────────────────────────────
add_executable(beam_linear_static beam_linear_static.cc)
target_link_libraries(beam_linear_static feris_lib Eigen3::Eigen CUDA::cudart
    CUDA::cusparse CUDA::cublas mophi_essentials)
target_include_directories(beam_linear_static PRIVATE ${BEAM_INCLUDE_DIRS})
set_target_properties(beam_linear_static PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Static TET4 beam (linear) ─────────────────────────────────────────────────
add_executable(beam_linear_static_t4 beam_linear_static_t4.cc)
target_link_libraries(beam_linear_static_t4 feris_lib Eigen3::Eigen CUDA::cudart
    CUDA::cusparse CUDA::cublas mophi_essentials)
target_include_directories(beam_linear_static_t4 PRIVATE ${BEAM_INCLUDE_DIRS})
set_target_properties(beam_linear_static_t4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Dynamic TET4 beam (FEAT4, Leapfrog) ──────────────────────────────────────
add_executable(test_leapfrog_t4 test_leapfrog_t4.cc)
target_link_libraries(test_leapfrog_t4 feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(test_leapfrog_t4 PRIVATE ${BEAM_INCLUDE_DIRS})
set_target_properties(test_leapfrog_t4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Dynamic LDPM-TET4 beam (6-DOF, Leapfrog) — current element ───────────────
# Uses TYPE_LDPM_TET4 (translational + linearised-rotational DOFs).
# This is the recommended LDPM element for new simulations.
add_executable(test_ldpm_tet4 test_ldpm_tet4.cc)
target_link_libraries(test_ldpm_tet4 feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(test_ldpm_tet4 PRIVATE ${BEAM_INCLUDE_DIRS})
set_target_properties(test_ldpm_tet4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
