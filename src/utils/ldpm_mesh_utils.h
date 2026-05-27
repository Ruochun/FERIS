/*==============================================================
 *==============================================================
 * Project: FERIS
 * File:    ldpm_mesh_utils.h
 * Brief:   Declares data structures and reader functions for the
 *          Chrono Workbench / LDPM mesh data files produced
 *          by the chrono-preprocessor tool:
 *
 *    <prefix>-data-nodes.dat          — particle XYZ (no index/diameter)
 *    <prefix>-data-particles.dat      — particle index, XYZ, diameter
 *    <prefix>-data-tets.dat           — TET4 connectivity (4 node indices)
 *    <prefix>-data-facets.dat         — Voronoi sub-facet geometry
 *    <prefix>-data-faceFacets.dat     — surface boundary triangles (optional)
 *    <prefix>-data-facetsVertices.dat — exact sub-facet triangle vertices
 *
 *  All readers skip lines that begin with "//" (C-style comments) and
 *  blank lines.  No explicit row-count header is required; each reader
 *  simply consumes data until EOF.
 *
 *  Typical usage
 *  ─────────────
 *    LDPMTet4Mesh mesh;
 *    std::string err;
 *    if (!ReadLDPMTet4MeshFromFiles(
 *            "data/meshes/LDPMTet4/Dogbone/LDPMgeo000", mesh, &err)) {
 *        std::cerr << err << "\n";
 *    }
 *    // mesh now holds all required files' data (and faceFacets if present).
 *    // Pass to GPU_LDPMTet4_Data::SetupFromMesh(mesh) to initialize the
 *    // element and store the extra precomputed fields.
 *==============================================================
 *==============================================================*/

#pragma once

#include <string>
#include <vector>

#include "../types.h"

namespace feris {

// ============================================================
// LDPMTet4Mesh — aggregates the data from all six LDPM files
// ============================================================

struct LDPMTet4Mesh {
    // ── From nodes.dat: X Y Z (one row per particle, no index / diameter) ───────
    int n_nodes = 0;
    VectorXR node_x, node_y, node_z;  // size n_nodes each

    // ── From particles.dat: n  x  y  z  d ───────────────────────────────────────
    // n is the zero-indexed particle ID; d is the aggregate diameter.
    // d == 0 for boundary/surface nodes; d > 0 for interior aggregate particles.
    int n_particles = 0;
    VectorXR particle_x, particle_y, particle_z;  // reference (initial) positions
    VectorXR particle_d;                          // aggregate diameters

    // ── From tets.dat: Node1  Node2  Node3  Node4 (zero-indexed) ────────────────
    int n_tets = 0;
    MatrixXi tet_connectivity;  // n_tets × 4

    // ── From facets.dat: Voronoi sub-facet geometry ──────────────────────────────
    // 12 sub-facets are generated per TET element by the Voronoi dual construction,
    // so n_subfacets == 12 * n_tets.
    //
    // Column layout per row:
    //   Tet  IDx  IDy  IDz  Vol  pArea  cx cy cz  px py pz  qx qy qz  sx sy sz  mF
    //
    // where:
    //   Tet        — index of the owning TET (0-indexed)
    //   IDx/IDy/IDz — indices into facetsVertices.dat for the 3 triangle corners
    //   Vol        — volume of the Voronoi sub-cell associated with this sub-facet
    //   pArea      — projected (triangle) area of the sub-facet
    //   cx, cy, cz — sub-facet centroid
    //   px, py, pz — unit normal  (= unit vector along particle-to-particle edge)
    //   qx, qy, qz — first in-plane tangent unit vector
    //   sx, sy, sz — second in-plane tangent unit vector (= p × q)
    //   mF         — material zone flag (0 = mortar/uniform; >0 = aggregate/ITZ)
    int n_subfacets = 0;
    VectorXi subfacet_tet;  // size n_subfacets
    // Vertex indices into facetsVertices.dat, stored flat (3 per sub-facet):
    //   subfacet_vertex_ids[3*i+0] = IDx,  [3*i+1] = IDy,  [3*i+2] = IDz
    VectorXi subfacet_vertex_ids;  // size 3 * n_subfacets
    VectorXR subfacet_vol;         // size n_subfacets
    VectorXR subfacet_parea;       // size n_subfacets
    // Centroid stored flat (3 per sub-facet): [cx0,cy0,cz0, cx1,cy1,cz1, ...]
    VectorXR subfacet_centroid;  // size 3 * n_subfacets
    // Unit normal (edge direction) stored flat: [px0,py0,pz0, ...]
    VectorXR subfacet_normal;  // size 3 * n_subfacets
    // First tangent stored flat: [qx0,qy0,qz0, ...]
    VectorXR subfacet_tangent_q;  // size 3 * n_subfacets
    // Second tangent stored flat: [sx0,sy0,sz0, ...]
    VectorXR subfacet_tangent_s;  // size 3 * n_subfacets
    VectorXi subfacet_matflag;    // size n_subfacets

    // ── From faceFacets.dat: surface boundary triangles ──────────────────────────
    // One triangle per row, associated with one particle node.
    // Column layout: n  x1 y1 z1  x2 y2 z2  x3 y3 z3
    int n_face_facets = 0;
    VectorXi face_facet_node;  // particle node index, size n_face_facets
    // 9 vertex coordinates per face facet stored flat (v1x,v1y,v1z, v2x,…, v3z):
    VectorXR face_facet_vertices;  // size 9 * n_face_facets

    // ── From facetsVertices.dat: exact triangle vertex coordinates ───────────────
    // 3 vertices per sub-facet => n_facet_vertices == 3 * n_subfacets.
    // Rows 3*i, 3*i+1, 3*i+2 are the three corners of sub-facet i.
    int n_facet_vertices = 0;
    VectorXR facet_vertex_x, facet_vertex_y, facet_vertex_z;  // size n_facet_vertices each
};

// ============================================================
// Combined reader — reads all required LDPM files from a common prefix.
//
// The prefix is the path up to and not including "-data-particles.dat".
// Example: "data/meshes/LDPMTet4/Dogbone/LDPMgeo000"
//
// Required files:
//   nodes, particles, tets, facets, facetsVertices
// Optional file:
//   faceFacets
//
// Returns true if all required files were read successfully.
// On failure the partially filled mesh and an error message are available.
// ============================================================
bool ReadLDPMTet4MeshFromFiles(const std::string& prefix, LDPMTet4Mesh& out, std::string* error = nullptr);

// ============================================================
// Individual file readers (also exported for selective use)
// ============================================================

// Read *-data-nodes.dat  →  fills out.n_nodes, out.node_x/y/z
bool ReadLDPMTet4NodesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// Read *-data-particles.dat  →  fills out.n_particles, particle_x/y/z/d
bool ReadLDPMTet4ParticlesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// Read *-data-tets.dat  →  fills out.n_tets, out.tet_connectivity
bool ReadLDPMTet4TetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// Read *-data-facets.dat  →  fills all subfacet_* fields
bool ReadLDPMTet4FacetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// Read *-data-faceFacets.dat  →  fills face_facet_node, face_facet_vertices
bool ReadLDPMTet4FaceFacetsFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// Read *-data-facetsVertices.dat  →  fills facet_vertex_x/y/z
bool ReadLDPMTet4FacetVerticesFile(const std::string& path, LDPMTet4Mesh& out, std::string* error = nullptr);

// ============================================================
// VTK writers for visual validation
// ============================================================

/**
 * Write the TET4 mesh from a parsed LDPMTet4Mesh to a VTK legacy ASCII file.
 *
 * POINTS   — particle positions (particle_x/y/z)
 * CELLS    — TET4 elements (tet_connectivity), VTK_TETRA (type 10)
 * POINT_DATA:
 *   "diameter" (scalar) — aggregate diameter per particle (0 for boundary nodes)
 *
 * @param filename  Output path (e.g. "ldpm_tet4_tets.vtk")
 * @param mesh      Populated LDPMTet4Mesh (particles.dat + tets.dat required)
 * @return true on success
 */
bool WriteLDPMTet4TetMeshToVTK(const std::string& filename, const LDPMTet4Mesh& mesh);

/**
 * Write the TET4 mesh from a parsed LDPMTet4Mesh to a VTK legacy ASCII file,
 * using the current (deformed) particle positions instead of the reference ones.
 *
 * This overload is intended for writing time-step snapshots during a dynamic
 * simulation.  The TET4 connectivity is identical to the reference-configuration
 * overload; only the POINTS block and added displacement scalar differ.
 *
 * POINTS   — current particle positions supplied via x_cur / y_cur / z_cur
 * CELLS    — TET4 elements (tet_connectivity), VTK_TETRA (type 10)
 * POINT_DATA:
 *   "diameter"     (scalar) — aggregate diameter per particle (0 = boundary)
 *   "displacement" (scalar) — Euclidean displacement magnitude |u| = |x_cur − x_ref|
 *
 * @param filename  Output path (e.g. "dogbone_tet4_00001.vtk")
 * @param mesh      Populated LDPMTet4Mesh (particles.dat + tets.dat required;
 *                  particle_x/y/z are used as the reference positions)
 * @param x_cur     Current x coordinates for all particles (size n_particles)
 * @param y_cur     Current y coordinates for all particles (size n_particles)
 * @param z_cur     Current z coordinates for all particles (size n_particles)
 * @return true on success
 */
bool WriteLDPMTet4TetMeshToVTK(const std::string& filename,
                               const LDPMTet4Mesh& mesh,
                               const VectorXR& x_cur,
                               const VectorXR& y_cur,
                               const VectorXR& z_cur);

/**
 * Write the Voronoi sub-facet triangle mesh from a parsed LDPMTet4Mesh to a
 * VTK legacy ASCII file.
 *
 * POINTS   — exact facet-vertex positions (facet_vertex_x/y/z)
 * CELLS    — one VTK_TRIANGLE (type 5) per sub-facet, indexed by
 *            subfacet_vertex_ids
 * CELL_DATA:
 *   "pArea"   (scalar) — projected (triangle) area of each sub-facet
 *   "matflag" (scalar) — material zone flag (0 = mortar; >0 = aggregate/ITZ)
 *
 * @param filename  Output path (e.g. "ldpm_tet4_subfacets.vtk")
 * @param mesh      Populated LDPMTet4Mesh (facets.dat + facetsVertices.dat required)
 * @return true on success
 */
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename, const LDPMTet4Mesh& mesh);

/**
 * Write the Voronoi sub-facet triangle mesh with a per-subfacet failure ratio.
 *
 * This overload is intended for time-step snapshots: instead of the static
 * geometric fields (pArea / matflag), it writes the projected LDPM damage
 * variable ω (the "failure ratio") as CELL_DATA, produced by
 * GPU_LDPMTet4_Data::ProjectEdgeDamageToSubfacets().
 *
 * POINTS   — exact facet-vertex positions (facet_vertex_x/y/z)
 * CELLS    — one VTK_TRIANGLE (type 5) per sub-facet
 * CELL_DATA:
 *   "damage" (scalar) — per-subfacet failure ratio ω ∈ [0, 1]
 *                       0 = undamaged / elastic,  1 = fully fractured
 *
 * @param filename        Output path (e.g. "dogbone_subfacets_00001.vtk")
 * @param mesh            Populated LDPMTet4Mesh (facets.dat + facetsVertices.dat)
 * @param subfacet_damage Per-subfacet damage values (size n_subfacets)
 * @return true on success
 */
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename,
                                    const LDPMTet4Mesh& mesh,
                                    const VectorXR& subfacet_damage);

/**
 * Write the Voronoi sub-facet triangle mesh with a named per-subfacet scalar.
 *
 * Generic overload that writes any per-subfacet scalar field under the
 * caller-supplied name.  Use this to write "crack_distance" (physical crack
 * opening in mm from ProjectEdgeCrackDistanceToSubfacets) or any other
 * derived quantity alongside the standard CELL_DATA block.
 *
 * POINTS   — exact facet-vertex positions (facet_vertex_x/y/z)
 * CELLS    — one VTK_TRIANGLE (type 5) per sub-facet
 * CELL_DATA:
 *   <field_name> (scalar double) — per-subfacet values supplied by the caller
 *
 * @param filename     Output path (e.g. "dogbone_crack_00001.vtk")
 * @param mesh         Populated LDPMTet4Mesh (facets.dat + facetsVertices.dat)
 * @param field_name   VTK scalar field name (e.g. "crack_distance", "damage")
 * @param subfacet_values Per-subfacet scalar values (size n_subfacets)
 * @return true on success
 */
bool WriteLDPMTet4SubfacetMeshToVTK(const std::string& filename,
                                    const LDPMTet4Mesh& mesh,
                                    const std::string& field_name,
                                    const VectorXR& subfacet_values);

/**
 * Write a VTK line-segment mesh showing the LDPM edge lattice coloured by
 * the per-edge Cusatis damage variable ω.
 *
 * This is the primary visualization for LDPM fracture/damage: each unique
 * particle-to-particle strut is drawn as a VTK_LINE (type 3), coloured by
 * ω ∈ [0, 1].  Blue (0) = undamaged; red (1) = fully fractured.
 *
 * POINTS   — current (possibly deformed) particle positions
 *            (x_cur / y_cur / z_cur, size n_particles)
 * CELLS    — one VTK_LINE per unique edge (2 endpoint indices)
 * CELL_DATA:
 *   "damage" (scalar) — per-edge damage ω ∈ [0, 1]
 *
 * @param filename      Output path (e.g. "dogbone_damage_00001.vtk")
 * @param n_particles   Number of particles (= size of x_cur/y_cur/z_cur)
 * @param n_edges       Number of unique edges
 * @param edge_nodes    Edge connectivity (flat, size 2*n_edges):
 *                      edge_nodes[2*e+0] = node i, [2*e+1] = node j
 * @param x_cur         Current x coordinates for all particles
 * @param y_cur         Current y coordinates for all particles
 * @param z_cur         Current z coordinates for all particles
 * @param edge_damage   Per-edge damage values ω (size n_edges)
 * @return true on success
 */
bool WriteLDPMTet4EdgeDamageToVTK(const std::string& filename,
                                  int n_particles,
                                  int n_edges,
                                  const std::vector<int>& edge_nodes,
                                  const VectorXR& x_cur,
                                  const VectorXR& y_cur,
                                  const VectorXR& z_cur,
                                  const VectorXR& edge_damage);

}  // namespace feris
