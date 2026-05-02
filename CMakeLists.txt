cmake_minimum_required(VERSION 3.18)
project(FERIS VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

# Add MoPhiEssentials subdirectory
add_subdirectory(external/MoPhiEssentials)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --expt-relaxed-constexpr")
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# Collect source files
# Note: CONFIGURE_DEPENDS ensures CMake rechecks when files are added/removed
file(GLOB ELEMENT_SOURCES CONFIGURE_DEPENDS src/elements/*.cu)
file(GLOB SOLVER_SOURCES CONFIGURE_DEPENDS src/solvers/*.cu)
file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS src/utils/*.cc)

# Create a library with all FEA components
add_library(feris_lib STATIC
    ${ELEMENT_SOURCES}
    ${SOLVER_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(feris_lib
    PUBLIC
        Eigen3::Eigen
        CUDA::cudart
        CUDA::cusparse
        CUDA::cublas
    PRIVATE
        mophi_essentials
)

target_include_directories(feris_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/feris>
)

# Add examples subdirectory
add_subdirectory(examples)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# ===========================================================================
# Installation policy
# ===========================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the compiled library and export its targets.
# mophi_essentials is also listed here so that CMake's install-export
# dependency checker (which flags even PRIVATE deps on older CMake) can
# resolve the target; the generated FERISTargets.cmake will carry it as
# FERIS::mophi_essentials.
install(TARGETS feris_lib mophi_essentials
    EXPORT FERISTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install all public headers, preserving the src/ subdirectory structure
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/feris
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

# Install bundled MoPhiEssentials headers alongside FERIS headers so
# downstream consumers that use the installed package still have them.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/feris
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

# Export targets file (for use from an install tree)
install(EXPORT FERISTargets
    FILE FERISTargets.cmake
    NAMESPACE FERIS::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FERIS
)

# Generate a version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FERISConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Generate the package config file from the template
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/FERISConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FERISConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FERIS
)

# Install the config and version files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FERISConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FERISConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FERIS
)
