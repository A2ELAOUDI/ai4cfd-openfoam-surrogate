# OpenFOAM Dam-Break Base Case

## Overview

This directory contains a complete interFoam setup for a 2D dam-break
simulation. The VOF (Volume of Fluid) method tracks the water–air interface
as it propagates across the domain after the virtual gate is released at t = 0.

## Domain

```
y [m]
2.0 ┤ atmosphere (open top)
    │
    │  air
    │
    │ ████████ ← water column
    │ ████████   height: H  (variable)
    │ ████████   width:  W  (variable)
0.0 ┤───────────────────────────────── x [m]
    0                               4.0
    leftWall                     rightWall
         lowerWall (no-slip floor)
```

| Parameter | Base value | Range (sweep) |
|-----------|-----------|---------------|
| Domain x  | 4.0 m     | fixed         |
| Domain y  | 2.0 m     | fixed         |
| Water height H | 0.6 m | 0.4 – 0.8 m |
| Water width W  | 0.4 m | 0.3 – 0.6 m |
| Mesh (x × y) | 160 × 80 | fixed       |

## Files

| File | Purpose |
|------|---------|
| `0/alpha.water` | Initial VOF fraction field (0=air, 1=water) |
| `0/U`           | Initial velocity field (uniform zero) |
| `0/p_rgh`       | Initial reduced pressure field |
| `constant/transportProperties` | Fluid densities and viscosities |
| `constant/turbulenceProperties` | Laminar assumption |
| `system/blockMeshDict` | Structured 160×80×1 mesh |
| `system/controlDict` | Solver, time step, output settings |
| `system/fvSchemes` | Discretisation schemes (vanLeer VOF, linearUpwind U) |
| `system/fvSolution` | Linear solvers (PCG+DIC for p, PBiCG+DILU for U) |
| `system/setFieldsDict` | Water column region for initialisation |

## Running Manually

```bash
# From a case directory (not base_case):
blockMesh          # generate mesh
setFields          # initialize alpha.water
interFoam          # run solver (or: mpirun -np 4 interFoam -parallel)
paraFoam           # post-process in ParaView
```

## Key Solver Settings

- **Solver:** interFoam (incompressible multiphase VOF)
- **Time step:** adaptive, maxCo = 0.5 for interface stability
- **VOF advection:** MULES with vanLeer limiter
- **Pressure:** PCG + DIC preconditioner
- **Velocity:** PBiCG + DILU preconditioner
- **Output:** every 0.1 s → 26 snapshots over 2.5 s

## Physical Properties

| Fluid | ρ [kg/m³] | ν [m²/s] |
|-------|-----------|----------|
| Water | 1000      | 1×10⁻⁶   |
| Air   | 1         | 1.48×10⁻⁵ |
| σ (surface tension) | — | 0.07 N/m |

## Expected Output

After a successful run, time directories `0.1/`, `0.2/`, ..., `2.5/` are
created, each containing:
- `alpha.water` — 2D VOF field
- `U` — velocity vector field
- `p_rgh` — pressure field

`extract_fields.py` reads these to compute:
- `wave_front_x`: max x where α_water > 0.5
- `max_velocity`: max |U| over the domain
