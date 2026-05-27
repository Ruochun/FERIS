# examples/ldmp_tests/CMakeLists.txt
#
# LDPM demos and benchmark cases: Dogbone tensile simulations and
# single-tet unit tests.

set(LDMP_TESTS_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# ── LDPM-TET4 Dogbone force-BC tensile simulation ─────────────────────────────
# Loads all six LDPM data files, initialises GPU_LDPMTet4_Data via
# SetupFromMesh() (which overrides tet4-derived facet areas and tangent frame
# vectors with the richer file-provided values), and runs a 0.1 s tensile
# simulation with a constant applied nodal force (force-BC / load-controlled).
# Step count and output intervals are computed from the CFL time step.
add_executable(dogbone_forcebc dogbone_forcebc.cc)
target_link_libraries(dogbone_forcebc feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_forcebc PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(dogbone_forcebc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── LDPM-TET4 Dogbone velocity-driven (kinematic BC) simulation ───────────────
# Drives the top plate via a prescribed velocity of 1 mm/s (quasi-static
# relative to the elastic wave speed) rather than an applied load, replicating
# the ChLinkMotorLinearPosition kinematic-BC approach.  Runs for 0.1 s with
# step count and output intervals computed from the CFL time step.
add_executable(dogbone_velbc dogbone_velbc.cc)
target_link_libraries(dogbone_velbc feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_velbc PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(dogbone_velbc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── LDPM-TET4 single-tet Case 1 velocity-history benchmark ───────────────────
add_executable(ldpm_singletet_case1_velbc ldpm_singletet_case1_velbc.cc)
target_link_libraries(ldpm_singletet_case1_velbc feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(ldpm_singletet_case1_velbc PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(ldpm_singletet_case1_velbc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
