/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    ldpm_mesh_utils.h
 * Brief:   Declares data structures and reader functions for the
 *          six Chrono Workbench / LDPM mesh data files produced
 *          by the chrono-preprocessor tool:
 *
 *    <prefix>-data-nodes.dat          — particle XYZ (no index/diameter)
 *    <prefix>-data-particles.dat      — particle index, XYZ, diameter
 *    <prefix>-data-tets.dat           — TET4 connectivity (4 node indices)
 *    <prefix>-data-facets.dat         — Voronoi sub-facet geometry
 *    <prefix>-data-faceFacets.dat     — surface boundary triangles
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
 *    // mesh now holds all six files' data.
 *    // Pass to GPU_LDPMTet4_Data::SetupFromMesh(mesh) to initialise the
 *    // element and store the extra precomputed fields.
 *==============================================================
 *==============================================================*/

#pragma once

#include <string>
#include <vector>

#include "../types.h"

namespace tlfea {

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
    VectorXR particle_d;                           // aggregate diameters

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
    VectorXi subfacet_tet;           // size n_subfacets
    // Vertex indices into facetsVertices.dat, stored flat (3 per sub-facet):
    //   subfacet_vertex_ids[3*i+0] = IDx,  [3*i+1] = IDy,  [3*i+2] = IDz
    VectorXi subfacet_vertex_ids;    // size 3 * n_subfacets
    VectorXR subfacet_vol;           // size n_subfacets
    VectorXR subfacet_parea;         // size n_subfacets
    // Centroid stored flat (3 per sub-facet): [cx0,cy0,cz0, cx1,cy1,cz1, ...]
    VectorXR subfacet_centroid;      // size 3 * n_subfacets
    // Unit normal (edge direction) stored flat: [px0,py0,pz0, ...]
    VectorXR subfacet_normal;        // size 3 * n_subfacets
    // First tangent stored flat: [qx0,qy0,qz0, ...]
    VectorXR subfacet_tangent_q;     // size 3 * n_subfacets
    // Second tangent stored flat: [sx0,sy0,sz0, ...]
    VectorXR subfacet_tangent_s;     // size 3 * n_subfacets
    VectorXi subfacet_matflag;       // size n_subfacets

    // ── From faceFacets.dat: surface boundary triangles ──────────────────────────
    // One triangle per row, associated with one particle node.
    // Column layout: n  x1 y1 z1  x2 y2 z2  x3 y3 z3
    int n_face_facets = 0;
    VectorXi face_facet_node;        // particle node index, size n_face_facets
    // 9 vertex coordinates per face facet stored flat (v1x,v1y,v1z, v2x,…, v3z):
    VectorXR face_facet_vertices;    // size 9 * n_face_facets

    // ── From facetsVertices.dat: exact triangle vertex coordinates ───────────────
    // 3 vertices per sub-facet => n_facet_vertices == 3 * n_subfacets.
    // Rows 3*i, 3*i+1, 3*i+2 are the three corners of sub-facet i.
    int n_facet_vertices = 0;
    VectorXR facet_vertex_x, facet_vertex_y, facet_vertex_z;  // size n_facet_vertices each
};

// ============================================================
// Combined reader — reads all six files from a common prefix.
//
// The prefix is the path up to and not including "-data-particles.dat".
// Example: "data/meshes/LDPMTet4/Dogbone/LDPMgeo000"
//
// Returns true if all files were read successfully.
// On failure the partially filled mesh and an error message are available.
// ============================================================
bool ReadLDPMTet4MeshFromFiles(const std::string& prefix,
                               LDPMTet4Mesh& out,
                               std::string* error = nullptr);

// ============================================================
// Individual file readers (also exported for selective use)
// ============================================================

// Read *-data-nodes.dat  →  fills out.n_nodes, out.node_x/y/z
bool ReadLDPMTet4NodesFile(const std::string& path,
                           LDPMTet4Mesh& out,
                           std::string* error = nullptr);

// Read *-data-particles.dat  →  fills out.n_particles, particle_x/y/z/d
bool ReadLDPMTet4ParticlesFile(const std::string& path,
                               LDPMTet4Mesh& out,
                               std::string* error = nullptr);

// Read *-data-tets.dat  →  fills out.n_tets, out.tet_connectivity
bool ReadLDPMTet4TetsFile(const std::string& path,
                          LDPMTet4Mesh& out,
                          std::string* error = nullptr);

// Read *-data-facets.dat  →  fills all subfacet_* fields
bool ReadLDPMTet4FacetsFile(const std::string& path,
                            LDPMTet4Mesh& out,
                            std::string* error = nullptr);

// Read *-data-faceFacets.dat  →  fills face_facet_node, face_facet_vertices
bool ReadLDPMTet4FaceFacetsFile(const std::string& path,
                                LDPMTet4Mesh& out,
                                std::string* error = nullptr);

// Read *-data-facetsVertices.dat  →  fills facet_vertex_x/y/z
bool ReadLDPMTet4FacetVerticesFile(const std::string& path,
                                   LDPMTet4Mesh& out,
                                   std::string* error = nullptr);

}  // namespace tlfea
