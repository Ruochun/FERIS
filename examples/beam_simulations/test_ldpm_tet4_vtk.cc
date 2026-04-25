/**
 * test_ldpm_tet4_vtk.cc
 *
 * LDPM-TET4 Mesh VTK Export Demo
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * Reads all six Chrono Workbench LDPM mesh data files for the Dogbone specimen
 * and writes two VTK meshes for visual validation in ParaView:
 *
 *   ldpm_tet4_tets.vtk      — TET4 mesh (particle nodes + tet elements)
 *   ldpm_tet4_subfacets.vtk — Voronoi sub-facet triangle mesh
 *
 * Background
 * ──────────
 * In LDPM the "nodes" and "particles" are synonymous: each carries 3 (or 6)
 * degrees of freedom and moves during the simulation.  The TET4 mesh is
 * therefore a Lagrangian moving mesh: particle positions evolve at every time
 * step while the TET connectivity (which particles share a tet) stays fixed.
 *
 * The Voronoi sub-facet geometry (area, centroid, local frame n/m/l) is
 * derived ONCE from the initial/reference configuration and does NOT change
 * during the simulation.  This is the geometrically-linear Total Lagrangian
 * formulation: strains are computed relative to these fixed reference facets.
 *
 * VTK output
 * ──────────
 * ldpm_tet4_tets.vtk
 *   - POINTS  : particle positions (particle_x/y/z from particles.dat)
 *   - CELLS   : VTK_TETRA (type 10) from tets.dat
 *   - POINT_DATA "diameter" : aggregate diameter (0 = boundary / embedding node)
 *
 * ldpm_tet4_subfacets.vtk
 *   - POINTS  : exact facet-vertex positions (facetsVertices.dat)
 *   - CELLS   : VTK_TRIANGLE (type 5) from facets.dat vertex IDs
 *   - CELL_DATA "pArea"   : projected facet area
 *   - CELL_DATA "matflag" : material zone (0 = mortar, >0 = aggregate/ITZ)
 *
 * Building
 * ────────
 *   mkdir -p build && cd build
 *   cmake ..
 *   make test_ldpm_tet4_vtk
 *
 * Running (from the build directory)
 * ───────────────────────────────────
 *   ./bin/test_ldpm_tet4_vtk
 *
 * Then open the two .vtk files in ParaView to inspect the meshes.
 */

#include <iostream>
#include <string>

#include "../src/utils/ldpm_mesh_utils.h"

using namespace tlfea;

int main() {
    // ── 1. Read all six LDPM mesh data files ────────────────────────────────
    const std::string prefix = "data/meshes/LDPMTet4/Dogbone/LDPMgeo000";

    std::cout << "Reading LDPM mesh files from prefix: " << prefix << "\n";

    LDPMTet4Mesh mesh;
    std::string err;
    if (!ReadLDPMTet4MeshFromFiles(prefix, mesh, &err)) {
        std::cerr << "Error loading mesh: " << err << "\n";
        return 1;
    }

    std::cout << "  Particles (nodes):  " << mesh.n_particles << "\n";
    std::cout << "  TET4 elements:      " << mesh.n_tets << "\n";
    std::cout << "  Sub-facets:         " << mesh.n_subfacets << "\n";
    std::cout << "  Facet vertices:     " << mesh.n_facet_vertices << "\n";
    std::cout << "  Surface face-facets:" << mesh.n_face_facets << "\n";

    // Sanity checks
    if (mesh.n_subfacets != 12 * mesh.n_tets) {
        std::cerr << "Warning: expected n_subfacets == 12 * n_tets ("
                  << 12 * mesh.n_tets << "), got " << mesh.n_subfacets << "\n";
    }
    if (mesh.n_facet_vertices != 3 * mesh.n_subfacets) {
        std::cerr << "Warning: expected n_facet_vertices == 3 * n_subfacets ("
                  << 3 * mesh.n_subfacets << "), got " << mesh.n_facet_vertices << "\n";
    }

    // Count aggregate vs boundary particles
    int n_aggregate = 0;
    for (int i = 0; i < mesh.n_particles; ++i) {
        if (mesh.particle_d(i) > 0.0)
            ++n_aggregate;
    }
    std::cout << "  Aggregate particles (d>0): " << n_aggregate << "\n";
    std::cout << "  Boundary particles  (d=0): " << (mesh.n_particles - n_aggregate) << "\n";

    // ── 2. Write TET4 mesh VTK ───────────────────────────────────────────────
    const std::string tet_vtk = "ldpm_tet4_tets.vtk";
    std::cout << "\nWriting TET4 mesh -> " << tet_vtk << " ...\n";
    if (!WriteLDPMTet4TetMeshToVTK(tet_vtk, mesh)) {
        std::cerr << "Error writing TET4 VTK.\n";
        return 1;
    }
    std::cout << "  " << mesh.n_particles << " points, " << mesh.n_tets << " VTK_TETRA cells.\n";

    // ── 3. Write sub-facet mesh VTK ──────────────────────────────────────────
    const std::string sf_vtk = "ldpm_tet4_subfacets.vtk";
    std::cout << "\nWriting sub-facet mesh -> " << sf_vtk << " ...\n";
    if (!WriteLDPMTet4SubfacetMeshToVTK(sf_vtk, mesh)) {
        std::cerr << "Error writing sub-facet VTK.\n";
        return 1;
    }
    std::cout << "  " << mesh.n_facet_vertices << " points, " << mesh.n_subfacets
              << " VTK_TRIANGLE cells.\n";

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\nDone.\n";
    std::cout << "  Open " << tet_vtk << " in ParaView to see the TET4 mesh\n";
    std::cout << "    → Color by 'diameter' to distinguish aggregate vs boundary nodes.\n";
    std::cout << "  Open " << sf_vtk << " in ParaView to see the sub-facet mesh\n";
    std::cout << "    → Color by 'matflag' to distinguish mortar / aggregate / ITZ zones.\n";
    std::cout << "    → Color by 'pArea' to inspect facet area distribution.\n";
    std::cout << "\nNote: sub-facet geometry is derived from the REFERENCE (initial) tet\n";
    std::cout << "mesh and stays fixed during simulation (Total Lagrangian formulation).\n";

    return 0;
}
