# Parallel CFD Solver - Terminal-Based

A high-performance, terminal-based Computational Fluid Dynamics (CFD) solver for 2D incompressible laminar flow using the Navier-Stokes equations. This solver is optimized for multi-core CPUs using OpenMP parallelization.

## Features

- **2D Incompressible Navier-Stokes equations** using Finite Difference Method (FDM)
- **Fractional Step (Projection) Method** for pressure-velocity coupling
- **Red-Black Gauss-Seidel** iterative solver for pressure Poisson equation
- **OpenMP parallelization** for multi-core performance
- **Memory-optimized** contiguous 1D array storage for 2D grids
- **Configurable parameters** via command-line arguments
- **CSV export** for visualization with ParaView or Python

## Table of Contents

- [Governing Equations](#governing-equations)
- [Numerical Methods](#numerical-methods)
- [Parallelization Strategy](#parallelization-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Test Cases](#test-cases)
- [Performance Results](#performance-results)
- [Project Structure](#project-structure)

## Governing Equations

### 2D Incompressible Navier-Stokes Equations

The solver implements the dimensionless form of the Navier-Stokes equations:

**Momentum Equation:**
```
∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u
```

**Continuity Equation (Incompressibility):**
```
∇·u = 0
```

Where:
- `u = (u, v)` is the velocity vector
- `p` is the pressure
- `Re` is the Reynolds number
- `∇` is the gradient operator
- `∇²` is the Laplacian operator

### Boundary Conditions

The solver supports:
- **No-slip walls**: `u = v = 0` at solid walls
- **Moving walls**: `u = U_wall, v = 0` for lid-driven cavity
- **Inflow**: Specified velocity `u = U_in, v = 0`
- **Outflow**: Zero-gradient condition `∂u/∂n = 0`

## Numerical Methods

### Fractional Step (Projection) Method

The solver uses a three-step projection method:

#### Step 1: Velocity Prediction
Compute intermediate velocity `u*` without pressure:
```
u* = uⁿ + Δt[-（u·∇)u + (1/Re)∇²u]
```

#### Step 2: Pressure Correction
Solve the pressure Poisson equation:
```
∇²p = (1/Δt)∇·u*
```

#### Step 3: Velocity Projection
Project to divergence-free velocity field:
```
uⁿ⁺¹ = u* - Δt·∇p
```

### Spatial Discretization

- **Central differences** for convective terms
- **Central differences** for diffusive terms (5-point Laplacian stencil)
- **Second-order accurate** in space

### Temporal Discretization

- **Explicit Euler** time stepping
- Stability constraint: CFL condition `Δt < min(Δx, Δy)/|u_max|`

### Pressure Solver: Red-Black Gauss-Seidel

The pressure Poisson equation is solved iteratively using Red-Black Gauss-Seidel:

1. **Red sweep**: Update cells where `(i+j) % 2 == 0`
2. **Black sweep**: Update cells where `(i+j) % 2 == 1`

This ordering eliminates data dependencies within each sweep, enabling parallelization.

**SOR (Successive Over-Relaxation)** option with relaxation factor ω:
```
p_new = (1-ω)p_old + ω·p_GS
```

## Parallelization Strategy

### OpenMP Implementation

The solver uses OpenMP for shared-memory parallelization:

#### Parallelized Components:

1. **Velocity Prediction Loop**
   ```cpp
   #pragma omp parallel for schedule(static) collapse(2)
   ```
   - Loop over all interior grid points
   - Each thread computes convection and diffusion independently

2. **Pressure Solver (Red-Black)**
   ```cpp
   #pragma omp parallel for schedule(static) collapse(2)
   ```
   - Red and black sweeps are separated
   - No data dependencies within each color sweep

3. **Velocity Projection Loop**
   - Similar parallelization as velocity prediction

4. **Boundary Condition Application**
   - Parallel updates along each boundary

#### Thread Safety

- **Red-Black ordering** ensures no race conditions in pressure solver
- **Static scheduling** for load balancing
- **Collapse clause** for better loop parallelization

### Expected Speedup

Typical speedup on multi-core systems:

| Threads | Speedup | Efficiency |
|---------|---------|------------|
| 1       | 1.0x    | 100%       |
| 2       | ~1.9x   | ~95%       |
| 4       | ~3.5x   | ~88%       |
| 8       | ~6.0x   | ~75%       |

## Installation

### Prerequisites

- **C++ Compiler**: g++ (GCC 7.0+) or clang++ with C++17 support
- **OpenMP**: Usually included with GCC
- **Make**: GNU Make

### Compilation

Clone and build:

```bash
git clone https://github.com/yourusername/Parallel-CFD-Solver-Terminal-Based-.git
cd Parallel-CFD-Solver-Terminal-Based-
make release
```

Build options:

```bash
make release    # Optimized build (-O3 -march=native)
make debug      # Debug build with symbols
make clean      # Clean build artifacts
```

### Compiler Flags

The optimized build uses:
- `-O3`: High optimization level
- `-march=native`: Target current CPU architecture
- `-fopenmp`: Enable OpenMP parallelization
- `-funroll-loops`: Loop unrolling
- `-ffast-math`: Fast floating-point operations

## Usage

### Basic Usage

```bash
./cfd_solver [options]
```

### Command-Line Options

#### Grid Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--nx <int>` | Grid points in x-direction | 64 |
| `--ny <int>` | Grid points in y-direction | 64 |
| `--Lx <float>` | Domain length in x | 1.0 |
| `--Ly <float>` | Domain length in y | 1.0 |

#### Physics Parameters
| Option | Description | Default |
|--------|-------------|---------|
| `--Re <float>` | Reynolds number | 100 |
| `--dt <float>` | Time step | 0.001 |
| `--t_end <float>` | End time | 1.0 |

#### Problem Selection
| Option | Description | Default |
|--------|-------------|---------|
| `--problem <string>` | `cavity` or `channel` | cavity |
| `--lid_vel <float>` | Lid velocity (cavity) | 1.0 |
| `--inflow_vel <float>` | Inflow velocity (channel) | 1.0 |

#### Output Options
| Option | Description | Default |
|--------|-------------|---------|
| `--output_interval <int>` | Output every N steps | 100 |
| `--export` | Export CSV files | disabled |
| `--output_dir <string>` | Output directory | output |
| `--quiet` | Suppress verbose output | disabled |

#### Parallel Options
| Option | Description | Default |
|--------|-------------|---------|
| `--threads <int>` | Number of OpenMP threads | auto |

#### Benchmarking
| Option | Description |
|--------|-------------|
| `--benchmark` | Run performance benchmark |
| `--help, -h` | Show help message |

### Examples

**Lid-driven cavity at Re=1000:**
```bash
./cfd_solver --nx 128 --ny 128 --Re 1000 --t_end 10.0 --export
```

**Channel flow with CSV export:**
```bash
./cfd_solver --problem channel --Re 100 --export --output_dir results
```

**Performance benchmark:**
```bash
./cfd_solver --benchmark --nx 64 --ny 64 --t_end 0.5
```

**Quick test with 4 threads:**
```bash
./cfd_solver --nx 32 --ny 32 --t_end 0.1 --threads 4
```

## Test Cases

### 1. Lid-Driven Cavity Flow

Classic benchmark case for incompressible flow solvers.

**Boundary Conditions:**
- Top: Moving lid with velocity `U = 1`
- Other walls: No-slip (`u = v = 0`)

**Run:**
```bash
make test
```

### 2. Channel Flow

Steady flow between parallel plates.

**Boundary Conditions:**
- Left: Inflow with specified velocity
- Right: Outflow (zero-gradient)
- Top/Bottom: No-slip walls

**Run:**
```bash
./cfd_solver --problem channel --Re 100 --t_end 2.0
```

## Performance Results

### Memory Optimization

- **Contiguous 1D arrays** for 2D grids improve cache utilization
- **Row-major ordering** aligned with C++ memory layout
- **Minimal temporary allocations** during iteration

### Typical Performance (64×64 grid, Re=100)

| Metric | Value |
|--------|-------|
| Timestep duration | ~0.5 ms |
| Pressure iterations | ~50-200 |
| Memory footprint | ~1 MB |

### Scalability

Run the benchmark to measure speedup:
```bash
./cfd_solver --benchmark --nx 128 --ny 128 --t_end 1.0
```

## Project Structure

```
Parallel-CFD-Solver-Terminal-Based-/
├── Makefile                    # Build system
├── README.md                   # Documentation
├── include/
│   ├── Field2D.hpp            # 2D field storage class
│   ├── Grid.hpp               # Grid and boundary conditions
│   ├── NavierStokesSolver.hpp # Main solver class
│   └── PressureSolver.hpp     # Pressure Poisson solver
├── src/
│   ├── main.cpp               # Main program and CLI
│   ├── NavierStokesSolver.cpp # Navier-Stokes implementation
│   └── PressureSolver.cpp     # Red-Black GS implementation
└── output/                    # CSV output directory
```

### Class Overview

| Class | Description |
|-------|-------------|
| `Field2D` | 2D scalar field with contiguous storage |
| `Grid` | Domain geometry and boundary conditions |
| `PressureSolver` | Red-Black Gauss-Seidel iterative solver |
| `NavierStokesSolver` | Fractional step Navier-Stokes solver |

## Visualization

Export velocity and pressure fields to CSV:
```bash
./cfd_solver --export --output_dir output
```

### Python Visualization Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
u = np.loadtxt('output/solution_t1.0000_u.csv', delimiter=',', comments='#')
v = np.loadtxt('output/solution_t1.0000_v.csv', delimiter=',', comments='#')

# Plot velocity magnitude
speed = np.sqrt(u**2 + v**2)
plt.figure(figsize=(8, 8))
plt.contourf(speed, levels=50, cmap='viridis')
plt.colorbar(label='Velocity Magnitude')
plt.title('Lid-Driven Cavity Flow')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('velocity_field.png', dpi=150)
plt.show()
```

## References

1. Chorin, A.J. (1968). "Numerical solution of the Navier-Stokes equations." *Mathematics of Computation*, 22(104), 745-762.

2. Ghia, U., Ghia, K.N., & Shin, C.T. (1982). "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method." *Journal of Computational Physics*, 48(3), 387-411.

3. Ferziger, J.H., & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.

## License

This project is open source and available under the MIT License.

## Author

Developed as a demonstration of parallel CFD implementation with OpenMP