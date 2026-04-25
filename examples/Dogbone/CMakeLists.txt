# examples/Dogbone/CMakeLists.txt
#
# Dogbone specimen demos using the LDPM-TET4 element.
#
# ┌──────────────────────┬──────────────────────────────────────────────────┐
# │ Target               │ Purpose                                          │
# ├──────────────────────┼──────────────────────────────────────────────────┤
# │ dogbone_vtk          │ Load LDPM mesh files; export two VTK meshes for  │
# │                      │ visual validation in ParaView (CPU-only)         │
# └──────────────────────┴──────────────────────────────────────────────────┘

set(DOGBONE_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/MoPhiEssentials/src
)

# ── LDPM-TET4 mesh VTK export (CPU-only, visual validation) ──────────────────
# Reads all six Chrono Workbench LDPM data files and writes two VTK meshes:
#   ldpm_tet4_tets.vtk      — TET4 mesh with per-particle diameter
#   ldpm_tet4_subfacets.vtk — Voronoi sub-facet triangles with pArea / matflag
add_executable(dogbone_vtk dogbone_vtk.cc)
target_link_libraries(dogbone_vtk tlfea_lib Eigen3::Eigen CUDA::cudart mophi_essentials)
target_include_directories(dogbone_vtk PRIVATE ${DOGBONE_INCLUDE_DIRS})
set_target_properties(dogbone_vtk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
