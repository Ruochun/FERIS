/* Project: FERIS
 * File: tpb_model.h
 * Brief: TPB-specific geometry, load-strip selection, and output data for LDPM TPB.
 */
#pragma once
#include <array>
#include <string>
#include <utility>
#include <vector>
#include "types.h"
#include "utils/ldpm_mesh_utils.h"
#include "utils/tetrahedral_mesh.h"
#include "dynamics/DynamicsSystem.h"

namespace feris {
namespace tpb {
struct Model {
    VectorReal3 arm_pos_ref;
    std::vector<std::array<int, 4>> arm_tets;
    VectorXR inv_mass;
    HostCsrMatrix stiffness;
    std::vector<Real> drive;
    std::vector<int> top_nodes;
    std::array<int, 2> cmod_nodes;
    std::array<int, 2> support_rows;
    int core_dofs = 0, plate_offset = 0;
    Real min_edge = 0, elastic_dt = 0;
};
TetrahedralMesh LoadArms(const std::string& inp);
Model BuildModel(const LDPMTet4Mesh& core,
                 const TetrahedralMesh& arms,
                 const HostCsrMatrix& stiffness,
                 DynamicsSystem& system,
                 int arm_offset);
void WriteArms(const std::string& path, const Model& model, const VectorXR& disp);
// Signed plate displacement, continuous in position and velocity at 2 ms.
inline Real PlateDisplacement(Real t) {
    return t <= Real(0.002) ? -Real(3750) * t * t : -Real(0.015) - Real(15) * (t - Real(0.002));
}
}  // namespace tpb
}  // namespace feris
