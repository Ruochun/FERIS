/*==============================================================
 *==============================================================
 * Project: TLFEA
 * File:    FEASolver.h
 * Brief:   Convenience header that includes all element and solver
 *          implementations in this project.  Including this single
 *          header gives access to every element type (FEAT4, FEAT10,
 *          ANCF3243, ANCF3443) and every solver (LinearStaticSolver,
 *          SyncedAdamW, SyncedAdamWNocoop, SyncedNesterov,
 *          LeapfrogSolver).
 *==============================================================
 *==============================================================*/

#pragma once

// Elements
#include "elements/ANCF3243Data.cuh"
#include "elements/ANCF3443Data.cuh"
#include "elements/FEAT10Data.cuh"
#include "elements/FEAT4Data.cuh"

// Solvers
#include "solvers/LeapfrogSolver.cuh"
#include "solvers/LinearStaticSolver.cuh"
#include "solvers/SyncedAdamW.cuh"
#include "solvers/SyncedAdamWNocoop.cuh"
#include "solvers/SyncedNesterov.cuh"
