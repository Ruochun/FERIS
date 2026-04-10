# Examples/Demos CMakeLists.txt

# Add the basic infrastructure test example
add_executable(test_feat10_nesterov
    test_feat10_nesterov.cc
)

target_link_libraries(test_feat10_nesterov
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_feat10_nesterov PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_feat10_nesterov PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the AdamW test example
add_executable(test_feat10_adamw
    test_feat10_adamw.cc
)

target_link_libraries(test_feat10_adamw
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_feat10_adamw PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_feat10_adamw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the beam simulation example
add_executable(beam_simulation
    beam_simulation.cc
)

target_link_libraries(beam_simulation
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(beam_simulation PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the linear static beam example
add_executable(beam_linear_static
    beam_linear_static.cc
)

target_link_libraries(beam_linear_static
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    mophi_essentials
)

target_include_directories(beam_linear_static PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_linear_static PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the TET4 leapfrog dynamic example
add_executable(test_leapfrog_t4
    test_leapfrog_t4.cc
)

target_link_libraries(test_leapfrog_t4
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_leapfrog_t4 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_leapfrog_t4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the TET4 linear static beam example
add_executable(beam_linear_static_t4
    beam_linear_static_t4.cc
)

target_link_libraries(beam_linear_static_t4
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    mophi_essentials
)

target_include_directories(beam_linear_static_t4 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(beam_linear_static_t4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the LDPM4 leapfrog dynamic example
add_executable(test_ldpm_t4
    test_ldpm_t4.cc
)

target_link_libraries(test_ldpm_t4
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_ldpm_t4 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_ldpm_t4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add the LDPM-TET4 leapfrog dynamic example (6-DOF with rotations)
add_executable(test_ldpm_tet4
    test_ldpm_tet4.cc
)

target_link_libraries(test_ldpm_tet4
    tlfea_lib
    Eigen3::Eigen
    CUDA::cudart
    mophi_essentials
)

target_include_directories(test_ldpm_tet4 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

set_target_properties(test_ldpm_tet4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
