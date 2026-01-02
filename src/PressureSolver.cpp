/**
 * @file PressureSolver.cpp
 * @brief Implementation of Red-Black Gauss-Seidel pressure solver
 * 
 * Uses OpenMP for parallel execution of the Red-Black update scheme.
 * The Red-Black ordering ensures thread-safe updates without race conditions.
 */

#include "PressureSolver.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>

PressureSolver::PressureSolver(const Grid& grid, const PressureSolverConfig& config)
    : grid_(grid),
      config_(config),
      last_iterations_(0),
      last_residual_(0.0)
{
    double dx = grid_.dx();
    double dy = grid_.dy();
    
    dx2_inv_ = 1.0 / (dx * dx);
    dy2_inv_ = 1.0 / (dy * dy);
    coeff_center_ = 2.0 * (dx2_inv_ + dy2_inv_);
}

bool PressureSolver::solve(Field2D& pressure, const Field2D& rhs) {
    last_iterations_ = 0;
    last_residual_ = 0.0;
    
    // Apply initial boundary conditions
    applyBoundaryConditions(pressure);
    
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Red sweep (i + j) % 2 == 0
        redBlackIteration(pressure, rhs, 0);
        
        // Black sweep (i + j) % 2 == 1
        redBlackIteration(pressure, rhs, 1);
        
        // Apply boundary conditions after each iteration
        applyBoundaryConditions(pressure);
        
        // Check convergence every few iterations for efficiency
        if ((iter + 1) % 10 == 0 || iter == config_.max_iterations - 1) {
            last_residual_ = computeResidual(pressure, rhs);
            last_iterations_ = iter + 1;
            
            if (last_residual_ < config_.tolerance) {
                return true;
            }
        }
    }
    
    last_iterations_ = config_.max_iterations;
    return false;
}

void PressureSolver::redBlackIteration(Field2D& pressure, const Field2D& rhs, int color) {
    const size_t nx = grid_.nx();
    const size_t ny = grid_.ny();
    const double omega = config_.omega;
    
    // Parallel Red-Black Gauss-Seidel iteration
    // The Red-Black scheme ensures no data dependencies within each color sweep
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            // Only update cells of the current color
            if ((i + j) % 2 == static_cast<size_t>(color)) {
                // 5-point stencil Laplacian
                double p_new = (
                    dx2_inv_ * (pressure(i, j-1) + pressure(i, j+1)) +
                    dy2_inv_ * (pressure(i-1, j) + pressure(i+1, j)) -
                    rhs(i, j)
                ) / coeff_center_;
                
                // Apply SOR relaxation
                pressure(i, j) = (1.0 - omega) * pressure(i, j) + omega * p_new;
            }
        }
    }
}

void PressureSolver::applyBoundaryConditions(Field2D& pressure) {
    const size_t nx = grid_.nx();
    const size_t ny = grid_.ny();
    
    // Neumann boundary conditions (zero pressure gradient at walls)
    // This is standard for incompressible flow with no-slip walls
    
    // Left boundary (j = 0)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        pressure(i, 0) = pressure(i, 1);
    }
    
    // Right boundary (j = nx-1)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        pressure(i, nx-1) = pressure(i, nx-2);
    }
    
    // Bottom boundary (i = 0)
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        pressure(0, j) = pressure(1, j);
    }
    
    // Top boundary (i = ny-1)
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        pressure(ny-1, j) = pressure(ny-2, j);
    }
}

double PressureSolver::computeResidual(const Field2D& pressure, const Field2D& rhs) {
    const size_t nx = grid_.nx();
    const size_t ny = grid_.ny();
    
    double residual = 0.0;
    
    // Compute L2 norm of residual: r = ∇²p - rhs
    #pragma omp parallel for schedule(static) collapse(2) reduction(+:residual)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            double laplacian = 
                dx2_inv_ * (pressure(i, j-1) - 2.0 * pressure(i, j) + pressure(i, j+1)) +
                dy2_inv_ * (pressure(i-1, j) - 2.0 * pressure(i, j) + pressure(i+1, j));
            
            double r = laplacian - rhs(i, j);
            residual += r * r;
        }
    }
    
    // Return RMS residual
    size_t interior_points = (nx - 2) * (ny - 2);
    return std::sqrt(residual / interior_points);
}
