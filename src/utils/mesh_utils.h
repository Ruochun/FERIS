#pragma once

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "../types.h"

namespace feris {

namespace ANCFCPUUtils {

/**
 * Node structure for GridMesh
 */
struct GridNode {
    int id;
    int i, j;  // grid coordinates
    Real x, y;
    Real dof_x, dof_dx_du, dof_dx_dv, dof_dx_dw;
};

/**
 * Element structure for GridMesh
 */
struct GridElement {
    int id;
    int n0, n1;               // node IDs
    std::string orientation;  // "H" or "V"
    Real length;
};

/**
 * GridMeshGenerator class for generating structured grid meshes
 */
class GridMeshGenerator {
  public:
    /**
     * Constructor for grid mesh generator
     * @param X Domain width in x-direction
     * @param Y Domain height in y-direction
     * @param L Element length (spacing)
     * @param include_horizontal Whether to include horizontal elements
     * @param include_vertical Whether to include vertical elements
     */
    GridMeshGenerator(Real X, Real Y, Real L, bool include_horizontal = true, bool include_vertical = true);

    /**
     * Generate the mesh (nodes and elements)
     */
    void generate_mesh();

    /**
     * Get generated coordinates
     * @param x Output x coordinates (4 DOFs per node)
     * @param y Output y coordinates (4 DOFs per node)
     * @param z Output z coordinates (4 DOFs per node)
     */
    void get_coordinates(VectorXR& x, VectorXR& y, VectorXR& z);

    /**
     * Get element connectivity matrix
     * @param connectivity Output connectivity matrix (n_elements × 2)
     */
    void get_element_connectivity(MatrixXi& connectivity);

    /**
     * Get number of nodes
     * @return Number of nodes
     */
    int get_num_nodes() const;

    /**
     * Get number of elements
     * @return Number of elements
     */
    int get_num_elements() const;

    /**
     * Get mesh summary
     * @return Summary dictionary
     */
    std::map<std::string, Real> summary() const;

  private:
    Real X_, Y_, L_;
    bool include_horizontal_, include_vertical_;
    int nx_, ny_;  // number of intervals in x and y
    std::vector<GridNode> nodes_;
    std::vector<GridElement> elements_;

    void generate_nodes();
    void generate_elements();
    int node_id(int i, int j) const;
    static std::tuple<int, int, int, int> global_dof_indices_for_node(int node_id);
};

// ============================================================
// ANCF3243 mesh IO + constraints (general utilities)
// ============================================================

struct LinearConstraintCSR {
    // Constraint rows are scalar equations. Columns index the flattened DOF
    // space:
    //   col = coef_index * 3 + component (0:x, 1:y, 2:z)
    // where coef_index is an ANCF "coefficient index" (4 per node for ANCF3243).
    //
    // The constraint value is:
    //   c[row] = sum_j values[j] * dof(columns[j]) - rhs[row]
    std::vector<int> offsets;  // size = n_rows + 1
    std::vector<int> columns;  // size = nnz
    std::vector<Real> values;  // size = nnz
    VectorXR rhs;              // size = n_rows

    int NumRows() const { return static_cast<int>(rhs.size()); }
    int NumNonZeros() const { return static_cast<int>(columns.size()); }
    bool Empty() const { return rhs.size() == 0; }
};

class LinearConstraintBuilder {
  public:
    explicit LinearConstraintBuilder(int n_dofs);
    LinearConstraintBuilder(int n_dofs, const LinearConstraintCSR& initial);

    int n_dofs() const { return n_dofs_; }
    int num_rows() const { return static_cast<int>(rhs_.size()); }
    int nnz() const { return static_cast<int>(columns_.size()); }

    // Adds a constraint row: sum(entries[i].second * dof(entries[i].first)) =
    // rhs. Returns the new row index.
    int AddRow(const std::vector<std::pair<int, Real>>& entries, Real rhs);

    // Convenience: dof(col) = rhs (a "fixed" scalar DOF constraint).
    int AddFixedDof(int col, Real rhs);

    // Serialize the builder to CSR.
    LinearConstraintCSR ToCSR() const;

  private:
    int n_dofs_;
    std::vector<int> offsets_;
    std::vector<int> columns_;
    std::vector<Real> values_;
    std::vector<Real> rhs_;
};

struct ANCF3243Mesh {
    // Parsed header/grid metadata (when present).
    int version = 0;  // file format version
    std::optional<int> grid_nx;
    std::optional<int> grid_ny;
    std::optional<Real> grid_L;
    std::optional<Vector3R> grid_origin;

    // Geometry + connectivity.
    int n_nodes = 0;
    int n_elements = 0;
    std::vector<std::string> node_family;  // size = n_nodes ("H"/"V"/...)
    VectorXR x_cur, y_cur, z_cur;                // size = 4 * n_nodes each
    MatrixXi element_connectivity;         // n_elements x 2 (node IDs)

    // Linear constraints encoded in scalar-DOF space (see LinearConstraintCSR).
    LinearConstraintCSR constraints;
};

// Reads an `.ancf3243mesh` file containing ANCF3243 geometry/connectivity and
// optional linear constraints. Returns false on parse/validation errors.
bool ReadANCF3243MeshFromFile(const std::string& path, ANCF3243Mesh& out, std::string* error = nullptr);

// ============================================================
// ANCF3443 shell mesh IO + constraints (general utilities)
// ============================================================

struct ANCF3443Mesh {
    // Parsed header metadata (when present).
    int version = 0;  // file format version

    // Geometry + connectivity.
    int n_nodes = 0;
    int n_elements = 0;
    std::vector<std::string> node_family;  // size = n_nodes ("R"/"S"/...)
    VectorXR x_cur, y_cur, z_cur;                // size = 4 * n_nodes each

    std::vector<std::string> element_family;  // size = n_elements
    VectorXR element_L;                       // size = n_elements
    VectorXR element_W;                       // size = n_elements
    VectorXR element_H;                       // size = n_elements
    MatrixXi element_connectivity;            // n_elements x 4 (node IDs)

    // Linear constraints encoded in scalar-DOF space (see LinearConstraintCSR).
    LinearConstraintCSR constraints;
};

// Reads an `.ancf3443mesh` file containing ANCF3443 geometry/connectivity and
// optional linear constraints. Returns false on parse/validation errors.
bool ReadANCF3443MeshFromFile(const std::string& path, ANCF3443Mesh& out, std::string* error = nullptr);

// Appends a 3D vector equality constraint: r(b,coef_slot) - r(a,coef_slot) = 0,
// where coef_slot is 0 for position, 1/2/3 for (r_u, r_v, r_w).
void AppendANCF3443VectorEqualityConstraint(LinearConstraintBuilder& builder, int node_a, int node_b, int coef_slot);

// Appends a 3D vector welded constraint: r(b,coef_slot) - Q * r(a,coef_slot) =
// 0. Q is row-major 3x3.
void AppendANCF3443VectorWeldedConstraint(LinearConstraintBuilder& builder,
                                          int node_a,
                                          int node_b,
                                          int coef_slot,
                                          const Matrix3R& Q);

// Appends a 3D vector equality constraint: r(b,coef_slot) - r(a,coef_slot) = 0,
// where coef_slot is 0 for position, 1/2/3 for (r_u, r_v, r_w).
void AppendANCF3243VectorEqualityConstraint(LinearConstraintBuilder& builder, int node_a, int node_b, int coef_slot);

// Appends a 3D vector welded constraint: r(b,coef_slot) - Q * r(a,coef_slot) =
// 0. Q is row-major 3x3.
void AppendANCF3243VectorWeldedConstraint(LinearConstraintBuilder& builder,
                                          int node_a,
                                          int node_b,
                                          int coef_slot,
                                          const Matrix3R& Q);

// Appends a "fixed coefficient" constraint for coef_index (ANCF coefficient
// index): component-wise equality to the provided reference (x_cur/y_cur/z_cur).
void AppendANCF3243FixedCoefficient(LinearConstraintBuilder& builder,
                                    int coef_index,
                                    const VectorXR& x12_ref,
                                    const VectorXR& y12_ref,
                                    const VectorXR& z12_ref);

/**
 * Write FEAT10 tetrahedral mesh to VTK format for visualization
 * @param filename Output VTK file path
 * @param nodes Node coordinates (n_nodes × 3)
 * @param elements Element connectivity (n_elements × 10)
 * @param x Deformed x coordinates (size n_nodes)
 * @param y Deformed y coordinates (size n_nodes)
 * @param z Deformed z coordinates (size n_nodes)
 * @return true if write successful, false otherwise
 */
bool WriteFEAT10ToVTK(const std::string& filename,
                      const MatrixXR& nodes,
                      const MatrixXi& elements,
                      const VectorXR& x,
                      const VectorXR& y,
                      const VectorXR& z);

/**
 * Write FEAT4 tetrahedral mesh to VTK format for visualization
 * @param filename Output VTK file path
 * @param nodes Node coordinates (n_nodes × 3)
 * @param elements Element connectivity (n_elements × 4)
 * @param x Deformed x coordinates (size n_nodes)
 * @param y Deformed y coordinates (size n_nodes)
 * @param z Deformed z coordinates (size n_nodes)
 * @return true if write successful, false otherwise
 */
bool WriteFEAT4ToVTK(const std::string& filename,
                     const MatrixXR& nodes,
                     const MatrixXi& elements,
                     const VectorXR& x,
                     const VectorXR& y,
                     const VectorXR& z);

}  // namespace ANCFCPUUtils

}  // namespace feris
