# examples/ldmp_tests/CMakeLists.txt
#
# LDPM demos and benchmark cases: Dogbone tensile simulations and
# single-tet unit tests, and notched three-point bending.

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
add_executable(dogbone_forcebc dogbone/dogbone_forcebc.cc)
target_link_libraries(dogbone_forcebc feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_forcebc PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(dogbone_forcebc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── LDPM-TET4 Dogbone velocity-driven (kinematic BC) simulation ───────────────
# Drives the top plate via a prescribed velocity of 1 mm/s (quasi-static
# relative to the elastic wave speed) rather than an applied load, replicating
# the ChLinkMotorLinearPosition kinematic-BC approach.  Runs for 0.1 s with
# step count and output intervals computed from the CFL time step.
add_executable(dogbone_velbc dogbone/dogbone_velbc.cc)
target_link_libraries(dogbone_velbc feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_velbc PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(dogbone_velbc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── LDPM-TET4 single-tet history benchmark ───────────────────
add_executable(ldpm_singletet singletet/ldpm_singletet.cc)
target_link_libraries(ldpm_singletet feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(ldpm_singletet PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
set_target_properties(ldpm_singletet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Reference notched three-point bending: LDPM core plus elastic arms.
add_executable(ldpm_tpb tpb/ldpm_tpb.cc tpb/tpb_model.cc)
target_link_libraries(ldpm_tpb PRIVATE feris_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(ldpm_tpb PRIVATE ${LDMP_TESTS_INCLUDE_DIRS})
target_compile_definitions(ldpm_tpb PRIVATE TPB_DEFAULT_MESH_DIR="${CMAKE_SOURCE_DIR}/data/meshes/LDPMTet4/TPB")
set_target_properties(ldpm_tpb PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Host-side physics checks; no CUDA device is needed to execute this target.
add_executable(tpb_model_test tpb/tpb_model_test.cc tpb/tpb_model.cc)
target_link_libraries(tpb_model_test PRIVATE feris_lib Eigen3::Eigen mophi_essentials)
target_compile_definitions(tpb_model_test PRIVATE TPB_DEFAULT_MESH_DIR="${CMAKE_SOURCE_DIR}/data/meshes/LDPMTet4/TPB")
set_target_properties(tpb_model_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(coupled_dynamics_test coupling/coupled_dynamics_test.cc)
target_link_libraries(coupled_dynamics_test PRIVATE feris_lib Eigen3::Eigen mophi_essentials)
set_target_properties(coupled_dynamics_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
