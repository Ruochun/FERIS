/**
 * dogbone_velbc.cc
 *
 * Dogbone — LDPM-TET4 Leapfrog Simulation with Velocity-Driven (Kinematic) BC
 *
 * Author: Ruochun Zhang
 * Email:  ruochunz@gmail.com
 *
 * GPU velocity-driven (kinematic-BC) dogbone tensile demo.
 *
 * The top plate is driven by a prescribed velocity rather than an applied load,
 * matching the ChLinkMotorLinearPosition approach used in Chrono-based LDPM
 * reference implementations.  This is the kinematic-BC counterpart of
 * dogbone_forcebc.cc (which applies an external nodal force instead).
 *
 *   1. Read all six Chrono Workbench LDPM mesh data files.
 *   2. Initialise GPU_LDPMTet4_Data via SetupFromMesh().
 *   3. Set elastic moduli and Cusatis tensile damage parameters.
 *   4. Fix particles on the z = z_min face (clamped bottom end).
 *   5. Mark particles on the z = z_max face as "kinematically driven"
 *      — they are also added to the fixed-node list so the leapfrog
 *      solver zeros their velocity each step; their positions are then
 *      advanced by an externally prescribed displacement increment
 *      v_prescribed(t) * dt after every solver step via
 *      GPU_LDPMTet4_Data::AdvanceDrivenNodesZ().
 *   6. Prescribed velocity: constant V_PLATE from t = 0.
 *   7. Time-march with LeapfrogSolver; at each output interval write VTK.
 *
 * Differences from dogbone_forcebc.cc (force-BC GPU demo)
 * ───────────────────────────────────────────────────────────────
 *  dogbone_forcebc.cc       │ this file
 *  ──────────────────────────┼────────────────────────────────────
 *  External force on top     │ No external forces
 *  Bottom nodes: fixed       │ Bottom nodes: fixed (same)
 *  Top nodes: loaded (free)  │ Top nodes: kinematically driven
 *  Physics drives top motion │ Prescribed velocity drives top motion
 *
 * Physics: Cusatis LDPM (2011) — same constitutive model as the load demo.
 *
 * VTK output (same format as dogbone_forcebc.cc)
 * ────────────────────────────────────────────────
 *   dogbone_velbc_tet4_NNNNN.vtk      — deformed TET4 mesh (displacement)
 *   dogbone_velbc_subfacets_NNNNN.vtk — sub-facet mesh with crack_distance [mm]
 *                                       (= ω × κ × l₀, the inelastic facet opening)
 *   dogbone_velbc_stress_disp.csv     — stress [MPa] vs displacement [mm] table
 *
 * Building
 * ────────
 *   mkdir -p build && cd build
 *   cmake ..
 *   make dogbone_velbc
 *
 * Running (from the build directory)
 * ───────────────────────────────────
 *   ./bin/dogbone_velbc
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <MoPhiEssentials.h>

#include "../src/elements/LDPMTet4Data.cuh"
#include "../src/solvers/LeapfrogSolver.cuh"
#include "../src/types.h"
#include "../src/utils/ldpm_mesh_utils.h"

using namespace feris;

// ── Simulation parameters ─────────────────────────────────────────────────────

// Material parameters — identical to dogbone_forcebc.cc.
// Unit system: mm-tonne-s (stress in N/mm² = MPa).
static constexpr Real E_N_VAL = Real(60273.0);   // N/mm²  normal modulus (E₀)
static constexpr Real ALPHA_T = Real(0.25);      // E_T / E_N  shear-to-normal ratio
static constexpr Real BETA_K = Real(0.25);       // rotational coupling
static constexpr Real RHO_VAL = Real(2.338e-9);  // tonne/mm³  (= 2338 kg/m³)
static constexpr Real SIGMA_T_VAL = Real(3.44);  // N/mm²  mesoscale tensile strength
static constexpr Real G_FT_VAL = Real(0.0491);   // N/mm   mode-I fracture energy

// Prescribed velocity of the top (driven) plate in mm/s.
// A 1 mm/s loading rate is quasi-static relative to the elastic wave speed:
// with specimen length ~150 mm and c_N ≈ 5e6 mm/s, V/c_N ≈ 2e-7 << 1.
static constexpr Real V_PLATE = Real(1.0);  // mm/s

// Total simulation time, VTK output interval, and CSV output interval.
// The step count and output intervals are computed from the CFL time step
// at runtime so the simulation always covers T_SIM regardless of mesh size.
static constexpr Real T_SIM = Real(0.1);     // s — total simulation time
static constexpr Real T_VTK = Real(0.005);   // s — VTK + console output interval (~20 frames)
static constexpr Real T_CSV = Real(0.0005);  // s — CSV stress-displacement output (~200 points)

// Fraction of bounding-box extent used as tolerance for boundary detection.
static constexpr Real BC_TOL_FRAC = Real(1e-3);

int main() {
    mophi::Logger::GetInstance().SetVerbosity(mophi::VERBOSITY_INFO);

    // ──────────────────────────────────────────────────────────────────────────
    // 1. Read all six LDPM mesh data files
    // ──────────────────────────────────────────────────────────────────────────
    const std::string prefix = "data/meshes/LDPMTet4/Dogbone/LDPMgeo000";

    std::cout << "Reading LDPM mesh from prefix: " << prefix << "\n";

    LDPMTet4Mesh mesh;
    std::string err;
    if (!ReadLDPMTet4MeshFromFiles(prefix, mesh, &err)) {
        std::cerr << "Error loading mesh: " << err << "\n";
        return 1;
    }

    std::cout << "  Particles : " << mesh.n_particles << "\n";
    std::cout << "  TETs      : " << mesh.n_tets << "\n";
    std::cout << "  Sub-facets: " << mesh.n_subfacets << "\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 2. Bounding box and minimum edge length
    // ──────────────────────────────────────────────────────────────────────────
    Real x_min = std::numeric_limits<Real>::max();
    Real x_max = -std::numeric_limits<Real>::max();
    Real y_min = x_min, y_max = x_max;
    Real z_min = x_min, z_max = x_max;

    for (int i = 0; i < mesh.n_particles; ++i) {
        x_min = std::min(x_min, mesh.particle_x(i));
        x_max = std::max(x_max, mesh.particle_x(i));
        y_min = std::min(y_min, mesh.particle_y(i));
        y_max = std::max(y_max, mesh.particle_y(i));
        z_min = std::min(z_min, mesh.particle_z(i));
        z_max = std::max(z_max, mesh.particle_z(i));
    }

    std::cout << "  Bounding box  x=[" << x_min << ", " << x_max << "]"
              << "  y=[" << y_min << ", " << y_max << "]"
              << "  z=[" << z_min << ", " << z_max << "]\n";

    // Approximate cross-sectional area (bounding-box XY face).
    const Real A_cross_approx = (x_max - x_min) * (y_max - y_min);

    // Minimum edge length (for CFL and rotational moduli).
    Real l_min = std::numeric_limits<Real>::max();
    for (int e = 0; e < mesh.n_tets; ++e) {
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                const int na = mesh.tet_connectivity(e, a);
                const int nb = mesh.tet_connectivity(e, b);
                const Real dx = mesh.particle_x(nb) - mesh.particle_x(na);
                const Real dy = mesh.particle_y(nb) - mesh.particle_y(na);
                const Real dz = mesh.particle_z(nb) - mesh.particle_z(na);
                const Real l = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (l < l_min)
                    l_min = l;
            }
        }
    }
    std::cout << "  Min edge length: " << l_min << " mm\n";

    // Derived material parameters.
    const Real E_T_VAL = ALPHA_T * E_N_VAL;
    const Real E_kT_VAL = BETA_K * E_N_VAL * l_min * l_min;
    const Real E_kM_VAL = E_kT_VAL;
    const Real E_kL_VAL = E_kT_VAL;

    // Cusatis softening modulus H_t ≈ l_min * σ_t / G_ft.
    const Real H_T_VAL = l_min * SIGMA_T_VAL / G_FT_VAL;
    std::cout << "  Damage threshold strain e_t0 = " << SIGMA_T_VAL / E_N_VAL << "\n";
    std::cout << "  Cusatis softening modulus H_t = " << H_T_VAL << "\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Identify fixed (bottom) and driven (top) particles
    //    Fixed  : z ≈ z_min  (clamped end — same as load-controlled demo)
    //    Driven : z ≈ z_max  (prescribed-velocity end — replaces applied force)
    // ──────────────────────────────────────────────────────────────────────────
    const Real z_range = z_max - z_min;
    const Real tol_z = BC_TOL_FRAC * z_range;

    std::vector<int> fixed_idx, driven_idx;
    for (int i = 0; i < mesh.n_particles; ++i) {
        const Real z = mesh.particle_z(i);
        if (std::abs(z - z_min) < tol_z)
            fixed_idx.push_back(i);
        if (std::abs(z - z_max) < tol_z)
            driven_idx.push_back(i);
    }

    if (fixed_idx.empty()) {
        std::cerr << "Could not find any particle at z = z_min for fixed BC.\n";
        return 1;
    }
    if (driven_idx.empty()) {
        std::cerr << "Could not find any particle at z = z_max for driven BC.\n";
        return 1;
    }

    std::cout << "  Fixed particles (z ≈ " << z_min << "):  " << fixed_idx.size() << "\n";
    std::cout << "  Driven particles (z ≈ " << z_max << "): " << driven_idx.size() << "\n";

    // Both fixed AND driven nodes are passed to SetNodalFixed() so that the
    // leapfrog solver zeros their velocity every step.  The driven nodes then
    // receive their prescribed displacement increment via AdvanceDrivenNodesZ().
    const int n_fixed = static_cast<int>(fixed_idx.size());
    const int n_driven = static_cast<int>(driven_idx.size());
    VectorXi h_constrained(n_fixed + n_driven);
    for (int i = 0; i < n_fixed; ++i)
        h_constrained(i) = fixed_idx[i];
    for (int i = 0; i < n_driven; ++i)
        h_constrained(n_fixed + i) = driven_idx[i];

    // Copy driven node index list to device for use in AdvanceDrivenNodesZ().
    int* d_driven_idx = nullptr;
    MOPHI_GPU_CALL(cudaMalloc(&d_driven_idx, n_driven * sizeof(int)));
    MOPHI_GPU_CALL(cudaMemcpy(d_driven_idx, driven_idx.data(), n_driven * sizeof(int), cudaMemcpyHostToDevice));

    // ──────────────────────────────────────────────────────────────────────────
    // 4. Create and configure GPU_LDPMTet4_Data
    // ──────────────────────────────────────────────────────────────────────────
    GPU_LDPMTet4_Data element_data;
    element_data.SetupFromMesh(mesh);

    std::cout << "  Unique edges: " << element_data.n_edge << "\n";

    element_data.SetMaterial(E_N_VAL, E_T_VAL, E_kT_VAL, E_kM_VAL, E_kL_VAL);
    element_data.SetDamageParams(SIGMA_T_VAL, H_T_VAL);
    element_data.SetDensity(RHO_VAL);

    // No external forces — the top plate is driven kinematically.
    VectorXR h_f_ext(mesh.n_particles * 3);
    h_f_ext.setZero();
    element_data.SetExternalForce(h_f_ext);

    // Register both fixed and driven nodes as "solver-fixed" (velocity = 0).
    element_data.SetNodalFixed(h_constrained);

    // ──────────────────────────────────────────────────────────────────────────
    // 5. Compute lumped mass
    // ──────────────────────────────────────────────────────────────────────────
    element_data.CalcMassMatrix();

    // ──────────────────────────────────────────────────────────────────────────
    // 6. CFL time step:  dt = 0.5 * l_min / c_N
    // ──────────────────────────────────────────────────────────────────────────
    const Real c_N = std::sqrt(E_N_VAL / RHO_VAL);  // mm/s  longitudinal wave speed
    const Real dt_crit = l_min / c_N;
    const Real dt = Real(0.5) * dt_crit;

    std::cout << "  c_N = " << c_N << " mm/s,  dt_crit = " << dt_crit << " s,  dt = " << dt << " s\n";
    std::cout << "  Prescribed plate velocity V_PLATE = " << V_PLATE << " mm/s\n";

    // ──────────────────────────────────────────────────────────────────────────
    // 7. Set up and run LeapfrogSolver
    // ──────────────────────────────────────────────────────────────────────────
    LeapfrogSolver solver(&element_data);

    LeapfrogParams params{dt};
    solver.SetParameters(&params);
    solver.Setup();

    // Step count and output intervals derived from CFL dt so the simulation
    // always runs for T_SIM seconds regardless of mesh refinement.
    const int n_steps = static_cast<int>(std::ceil(T_SIM / dt));
    const int vtk_interval = std::max(1, static_cast<int>(std::round(T_VTK / dt)));
    const int csv_interval = std::max(1, static_cast<int>(std::round(T_CSV / dt)));
    const int print_interval = vtk_interval;

    std::cout << "\nRunning " << n_steps << " leapfrog steps (dt = " << dt << " s)...\n";
    std::cout << "  Total simulation time : " << n_steps * dt << " s\n";
    std::cout << "  VTK output every      : " << vtk_interval << " steps (" << T_VTK << " s)\n";
    std::cout << "  CSV output every      : " << csv_interval << " steps (" << T_CSV << " s)\n";
    std::cout << std::fixed << std::setprecision(6);

    // Helpers: build numbered VTK filenames.
    auto tet4_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_velbc_tet4_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };
    auto subfacet_vtk_name = [](int f) {
        std::ostringstream s;
        s << "dogbone_velbc_subfacets_" << std::setfill('0') << std::setw(5) << f << ".vtk";
        return s.str();
    };

    // Open stress-displacement CSV for writing.
    // Each row: displacement_mm,stress_MPa
    std::ofstream stress_disp_csv("dogbone_velbc_stress_disp.csv");
    if (!stress_disp_csv.is_open()) {
        std::cerr << "Warning: could not open dogbone_velbc_stress_disp.csv for writing.\n";
    } else {
        stress_disp_csv << "displacement_mm,stress_MPa\n";
    }

    // ── Write frame 0: initial (undeformed) state ─────────────────────────────
    int frame = 0;
    {
        const std::string fname_t4 = tet4_vtk_name(frame);
        if (!WriteLDPMTet4TetMeshToVTK(fname_t4, mesh, mesh.particle_x, mesh.particle_y, mesh.particle_z)) {
            std::cerr << "Warning: failed to write initial TET4 VTK.\n";
        } else {
            std::cout << "Wrote: " << fname_t4 << "  (initial configuration)\n";
        }

        VectorXR sf_crack0(mesh.n_subfacets);
        sf_crack0.setZero();
        const std::string fname_sf = subfacet_vtk_name(frame);
        if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, "crack_distance", sf_crack0)) {
            std::cerr << "Warning: failed to write initial sub-facet VTK.\n";
        } else {
            std::cout << "Wrote: " << fname_sf << "  (initial, all crack_distance=0)\n";
        }
        ++frame;
    }

    // Print column header.
    const int n_edge = element_data.n_edge;
    std::cout << "\n"
              << std::setw(8) << "Step" << std::setw(20) << "prescribed dz [mm]" << std::setw(20)
              << "mean dz (driven) [mm]" << std::setw(18) << "stress [MPa]"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    Real prescribed_z_total = Real(0);  // cumulative prescribed displacement

    for (int step = 0; step < n_steps; ++step) {
        // Prescribed velocity: constant V_PLATE from t = 0.
        const Real dz = V_PLATE * dt;
        prescribed_z_total += dz;

        // Leapfrog step: computes tractions, updates velocities (zeroed at
        // constrained nodes), and advances unconstrained particle positions.
        solver.Solve();

        // Kinematic BC: advance driven (top-plate) particle z-positions by the
        // prescribed displacement increment.  These nodes had their velocity
        // zeroed by the solver, so this is their only source of z-motion.
        element_data.AdvanceDrivenNodesZ(d_driven_idx, n_driven, dz);

        if ((step + 1) % csv_interval == 0 || (step + 1) % vtk_interval == 0) {
            VectorXR x_cur, y_cur, z_cur;
            element_data.RetrievePositionToCPU(x_cur, y_cur, z_cur);

            // Internal forces for stress computation.
            VectorXR f_int;
            element_data.RetrieveInternalForceToCPU(f_int);

            // Mean actual z-displacement of driven nodes.
            Real dz_sum = Real(0);
            for (int i : driven_idx)
                dz_sum += z_cur(i) - mesh.particle_z(i);
            const Real dz_mean_driven = dz_sum / static_cast<Real>(driven_idx.size());

            // Reaction stress: sum z-component of internal forces at fixed
            // (bottom) nodes, divide by cross-sectional area.  The internal
            // force at a fixed node points in the direction the material
            // pulls it; negate to get the compressive reaction (tensile = positive).
            Real f_reaction_z = Real(0);
            for (int i : fixed_idx)
                f_reaction_z += f_int(i * 3 + 2);
            const Real stress = -f_reaction_z / A_cross_approx;

            // Write to stress-displacement CSV at high frequency.
            if ((step + 1) % csv_interval == 0 && stress_disp_csv.is_open()) {
                stress_disp_csv << dz_mean_driven << "," << stress << "\n";
            }

            // Console progress at VTK (lower) frequency.
            if ((step + 1) % print_interval == 0) {
                std::cout << std::setw(8) << (step + 1) << std::setw(20) << prescribed_z_total << std::setw(20)
                          << dz_mean_driven << std::setw(18) << stress << "\n";
            }

            // VTK snapshot.
            if ((step + 1) % vtk_interval == 0) {
                const std::string fname_t4 = tet4_vtk_name(frame);
                if (!WriteLDPMTet4TetMeshToVTK(fname_t4, mesh, x_cur, y_cur, z_cur)) {
                    std::cerr << "Warning: failed to write TET4 VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_t4 << "\n";
                }

                VectorXR sf_crack;
                element_data.ProjectEdgeCrackDistanceToSubfacets(sf_crack);
                const std::string fname_sf = subfacet_vtk_name(frame);
                if (!WriteLDPMTet4SubfacetMeshToVTK(fname_sf, mesh, "crack_distance", sf_crack)) {
                    std::cerr << "Warning: failed to write sub-facet VTK at step " << (step + 1) << ".\n";
                } else {
                    std::cout << "Wrote: " << fname_sf << "\n";
                }
                ++frame;
            }
        }
    }

    stress_disp_csv.close();

    // ──────────────────────────────────────────────────────────────────────────
    // 8. Final fracture statistics
    // ──────────────────────────────────────────────────────────────────────────
    {
        VectorXR x_final, y_final, z_final;
        element_data.RetrievePositionToCPU(x_final, y_final, z_final);

        VectorXR omega_final, kappa_final;
        element_data.RetrieveFacetDamageToCPU(omega_final);
        element_data.RetrieveFacetKappaToCPU(kappa_final);

        // Host-side edge data for per-edge crack distance.
        const std::vector<int>& edge_nodes = element_data.GetEdgeNodes();

        Real dz_sum = Real(0);
        for (int i : driven_idx)
            dz_sum += z_final(i) - mesh.particle_z(i);
        const Real dz_final_mean = dz_sum / static_cast<Real>(driven_idx.size());

        Real omega_max = Real(0);
        Real crack_dist_max = Real(0);
        int n_damaged = 0;
        for (int e = 0; e < n_edge; ++e) {
            if (omega_final(e) > omega_max)
                omega_max = omega_final(e);
            const int ni = edge_nodes[2 * e + 0];
            const int nj = edge_nodes[2 * e + 1];
            const Real dx0 = mesh.particle_x(nj) - mesh.particle_x(ni);
            const Real dy0 = mesh.particle_y(nj) - mesh.particle_y(ni);
            const Real dz0 = mesh.particle_z(nj) - mesh.particle_z(ni);
            const Real l0_e = std::sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
            const Real cd = omega_final(e) * kappa_final(e) * l0_e;
            if (cd > crack_dist_max)
                crack_dist_max = cd;
            if (omega_final(e) > Real(0.5))
                ++n_damaged;
        }

        const Real t_final = n_steps * dt;
        const Real L = z_max - z_min;
        const Real strain_final = prescribed_z_total / L;
        const Real e_t0 = SIGMA_T_VAL / E_N_VAL;

        std::cout << "\n── Final state at t=" << t_final << " s ───\n";
        std::cout << "  Prescribed top-plate displacement: " << prescribed_z_total << " mm\n";
        std::cout << "  Mean actual driven-node dz       : " << dz_final_mean << " mm\n";
        std::cout << "  Nominal axial strain             : " << strain_final << " (= " << strain_final / e_t0
                  << " × e_t0)\n";
        std::cout << "  Maximum edge damage omega_max    : " << omega_max << "\n";
        std::cout << "  Max crack opening (ω×κ×l₀)      : " << crack_dist_max << " mm\n";
        std::cout << "  Edges with omega > 0.5           : " << n_damaged << " / " << n_edge << "\n";
        std::cout << "(Non-zero damage indicates Cusatis LDPM fracture is active.)\n";
    }

    // Cleanup.
    MOPHI_GPU_CALL(cudaFree(d_driven_idx));
    element_data.Destroy();

    std::cout << "\nDone.  " << frame << " VTK frame(s) written.\n";
    std::cout << "\nOutput files (two sets per frame + stress-displacement CSV):\n";
    std::cout << "  dogbone_velbc_tet4_NNNNN.vtk         — deformed TET4 mesh\n";
    std::cout << "    → Color by 'displacement' to visualise particle motion.\n";
    std::cout << "  dogbone_velbc_subfacets_NNNNN.vtk    — Voronoi sub-facet mesh\n";
    std::cout << "    → Color by 'crack_distance' [mm] (ω×κ×l₀) to visualise fracture.\n";
    std::cout << "    → Use Threshold filter (crack_distance > 0) to show cracked facets.\n";
    std::cout << "  dogbone_velbc_stress_disp.csv        — stress [MPa] vs displacement [mm]\n";
    std::cout << "    → Plot displacement_mm vs stress_MPa for the load-displacement curve.\n";
    return 0;
}
