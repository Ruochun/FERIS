/* Project: FERIS
 * File: tetrahedral_mesh.h
 * Brief: T4/T10 Abaqus input, boundary faces, and surface interpolation.
 */
#pragma once
#include <string>
#include "../types.h"
namespace feris {
struct TetrahedralMesh {
    VectorReal3 positions;
    MatrixXi connectivity;
};
struct TetrahedralFace {
    std::vector<int> nodes;
};
TetrahedralMesh ReadAbaqusTetrahedralMesh(const std::string& path);
std::vector<TetrahedralFace> ExtractBoundaryFaces(const TetrahedralMesh& mesh);
// T4: triangle barycentric weights. T10: six quadratic shape weights, with
// an isoparametric inverse map for curved faces. False means no face match.
bool InterpolateFace(const TetrahedralMesh& mesh,
                     const TetrahedralFace& face,
                     const Real3& point,
                     Real tolerance,
                     VectorXR& weights);
}  // namespace feris
