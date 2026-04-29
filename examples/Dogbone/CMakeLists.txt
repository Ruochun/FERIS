# examples/Dogbone/CMakeLists.txt
#
# Dogbone specimen demo using the LDPM-TET4 element.

set(DOGBONE_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# ── LDPM-TET4 Dogbone leapfrog simulation (file-based sub-facet geometry) ─────
# Loads all six LDPM data files, initialises GPU_LDPMTet4_Data via
# SetupFromMesh() (which overrides tet4-derived facet areas and tangent frame
# vectors with the richer file-provided values), and runs a short tensile
# simulation with LeapfrogSolver for verification.
add_executable(dogbone_leapfrog dogbone_leapfrog.cc)
target_link_libraries(dogbone_leapfrog tlfea_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_leapfrog PRIVATE ${DOGBONE_INCLUDE_DIRS})
set_target_properties(dogbone_leapfrog PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── LDPM-TET4 Dogbone velocity-driven (kinematic BC) simulation ───────────────
# GPU counterpart of dogbone_cpuref.cpp: drives the top plate via a prescribed
# velocity rather than an applied load, replicating the ChLinkMotorLinearPosition
# approach used in the Chrono-based CPU reference.
add_executable(dogbone_velbc dogbone_velbc.cc)
target_link_libraries(dogbone_velbc tlfea_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_velbc PRIVATE ${DOGBONE_INCLUDE_DIRS})
set_target_properties(dogbone_velbc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
