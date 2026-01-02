/**
 * @file NavierStokesSolver.cpp
 * @brief Implementation of 2D Incompressible Navier-Stokes solver
 * 
 * Uses the fractional step (projection) method with:
 * - Central differences for spatial derivatives
 * - Explicit Euler time stepping
 * - Red-Black Gauss-Seidel for pressure Poisson equation
 * 
 * OpenMP parallelization is applied to:
 * - Velocity prediction loops
 * - Pressure correction (via PressureSolver)
 * - Velocity projection loops
 * - Boundary condition application
 */

#include "NavierStokesSolver.hpp"
#include <omp.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

NavierStokesSolver::NavierStokesSolver(const GridConfig& grid_config, 
                                        const SimulationConfig& sim_config)
    : config_(sim_config),
      nu_(1.0 / sim_config.Re),
      rho_(1.0),
      current_time_(0.0),
      current_step_(0)
{
    // Create grid
    grid_ = std::make_unique<Grid>(grid_config);
    
    // Initialize fields
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    
    u_.resize(ny, nx, 0.0);
    v_.resize(ny, nx, 0.0);
    u_star_.resize(ny, nx, 0.0);
    v_star_.resize(ny, nx, 0.0);
    p_.resize(ny, nx, 0.0);
    rhs_.resize(ny, nx, 0.0);
    
    // Create pressure solver
    pressure_solver_ = std::make_unique<PressureSolver>(*grid_, pressure_config_);
}

void NavierStokesSolver::initialize() {
    // Initialize velocity fields to zero
    u_.fill(0.0);
    v_.fill(0.0);
    p_.fill(0.0);
    
    // Apply boundary conditions
    applyBoundaryConditions();
    
    current_time_ = 0.0;
    current_step_ = 0;
}

void NavierStokesSolver::setPressureSolverConfig(const PressureSolverConfig& config) {
    pressure_config_ = config;
    pressure_solver_->setConfig(config);
}

SimulationStats NavierStokesSolver::step() {
    double start_time = omp_get_wtime();
    
    // Step 1: Velocity prediction
    velocityPrediction();
    
    // Step 2: Pressure correction
    pressureCorrection();
    
    // Step 3: Velocity projection
    velocityProjection();
    
    // Apply final boundary conditions
    applyBoundaryConditions();
    
    // Update time
    current_time_ += config_.dt;
    current_step_++;
    
    double end_time = omp_get_wtime();
    
    // Compute statistics
    SimulationStats stats;
    stats.time = current_time_;
    stats.timestep = current_step_;
    stats.max_u = u_.maxAbs();
    stats.max_v = v_.maxAbs();
    stats.max_div = computeMaxDivergence();
    stats.pressure_residual = pressure_solver_->getLastResidual();
    stats.pressure_iterations = pressure_solver_->getLastIterations();
    stats.wall_time = end_time - start_time;
    
    return stats;
}

void NavierStokesSolver::run(std::function<void(const SimulationStats&)> progress_callback) {
    initialize();
    
    while (current_time_ < config_.t_end) {
        SimulationStats stats = step();
        
        if (progress_callback && 
            (current_step_ % config_.output_interval == 0 || current_time_ >= config_.t_end)) {
            progress_callback(stats);
        }
    }
}

void NavierStokesSolver::velocityPrediction() {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    const double dt = config_.dt;
    
    // Compute intermediate velocity u* = u + dt * (-(u·∇)u + ν∇²u)
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            double conv_u, conv_v;
            double diff_u, diff_v;
            
            computeConvection(conv_u, conv_v, i, j);
            computeDiffusion(diff_u, diff_v, i, j);
            
            // Explicit Euler update (without pressure gradient)
            u_star_(i, j) = u_(i, j) + dt * (-conv_u + diff_u);
            v_star_(i, j) = v_(i, j) + dt * (-conv_v + diff_v);
        }
    }
    
    // Apply boundary conditions to intermediate velocity
    applyBoundaryConditionsStar();
}

void NavierStokesSolver::pressureCorrection() {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    const double dt = config_.dt;
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    
    // Compute RHS of pressure Poisson equation: ∇²p = (ρ/Δt)∇·u*
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            // Divergence of intermediate velocity using central differences
            double div_u_star = (u_star_(i, j+1) - u_star_(i, j-1)) / (2.0 * dx) +
                                (v_star_(i+1, j) - v_star_(i-1, j)) / (2.0 * dy);
            
            rhs_(i, j) = (rho_ / dt) * div_u_star;
        }
    }
    
    // Solve pressure Poisson equation
    pressure_solver_->solve(p_, rhs_);
}

void NavierStokesSolver::velocityProjection() {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    const double dt = config_.dt;
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    
    // Project velocity: u^(n+1) = u* - (Δt/ρ)∇p
    #pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            // Pressure gradient using central differences
            double dp_dx = (p_(i, j+1) - p_(i, j-1)) / (2.0 * dx);
            double dp_dy = (p_(i+1, j) - p_(i-1, j)) / (2.0 * dy);
            
            u_(i, j) = u_star_(i, j) - (dt / rho_) * dp_dx;
            v_(i, j) = v_star_(i, j) - (dt / rho_) * dp_dy;
        }
    }
}

void NavierStokesSolver::applyBoundaryConditions() {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    
    // Left boundary (j = 0)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        switch (grid_->bcLeft()) {
            case BoundaryType::NO_SLIP:
                u_(i, 0) = 0.0;
                v_(i, 0) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_(i, 0) = grid_->uLeft();
                v_(i, 0) = grid_->vLeft();
                break;
            case BoundaryType::OUTFLOW:
                u_(i, 0) = u_(i, 1);
                v_(i, 0) = v_(i, 1);
                break;
            case BoundaryType::MOVING_WALL:
                u_(i, 0) = grid_->uLeft();
                v_(i, 0) = grid_->vLeft();
                break;
            default:
                break;
        }
    }
    
    // Right boundary (j = nx-1)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        switch (grid_->bcRight()) {
            case BoundaryType::NO_SLIP:
                u_(i, nx-1) = 0.0;
                v_(i, nx-1) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_(i, nx-1) = grid_->uRight();
                v_(i, nx-1) = grid_->vRight();
                break;
            case BoundaryType::OUTFLOW:
                u_(i, nx-1) = u_(i, nx-2);
                v_(i, nx-1) = v_(i, nx-2);
                break;
            case BoundaryType::MOVING_WALL:
                u_(i, nx-1) = grid_->uRight();
                v_(i, nx-1) = grid_->vRight();
                break;
            default:
                break;
        }
    }
    
    // Bottom boundary (i = 0)
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        switch (grid_->bcBottom()) {
            case BoundaryType::NO_SLIP:
                u_(0, j) = 0.0;
                v_(0, j) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_(0, j) = grid_->uBottom();
                v_(0, j) = grid_->vBottom();
                break;
            case BoundaryType::OUTFLOW:
                u_(0, j) = u_(1, j);
                v_(0, j) = v_(1, j);
                break;
            case BoundaryType::MOVING_WALL:
                u_(0, j) = grid_->uBottom();
                v_(0, j) = grid_->vBottom();
                break;
            default:
                break;
        }
    }
    
    // Top boundary (i = ny-1)
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        switch (grid_->bcTop()) {
            case BoundaryType::NO_SLIP:
                u_(ny-1, j) = 0.0;
                v_(ny-1, j) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_(ny-1, j) = grid_->uTop();
                v_(ny-1, j) = grid_->vTop();
                break;
            case BoundaryType::OUTFLOW:
                u_(ny-1, j) = u_(ny-2, j);
                v_(ny-1, j) = v_(ny-2, j);
                break;
            case BoundaryType::MOVING_WALL:
                u_(ny-1, j) = grid_->uTop();
                v_(ny-1, j) = grid_->vTop();
                break;
            default:
                break;
        }
    }
}

void NavierStokesSolver::applyBoundaryConditionsStar() {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    
    // Apply same boundary conditions to intermediate velocity
    // Left boundary
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        switch (grid_->bcLeft()) {
            case BoundaryType::NO_SLIP:
                u_star_(i, 0) = 0.0;
                v_star_(i, 0) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_star_(i, 0) = grid_->uLeft();
                v_star_(i, 0) = grid_->vLeft();
                break;
            case BoundaryType::OUTFLOW:
                u_star_(i, 0) = u_star_(i, 1);
                v_star_(i, 0) = v_star_(i, 1);
                break;
            case BoundaryType::MOVING_WALL:
                u_star_(i, 0) = grid_->uLeft();
                v_star_(i, 0) = grid_->vLeft();
                break;
            default:
                break;
        }
    }
    
    // Right boundary
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ny; ++i) {
        switch (grid_->bcRight()) {
            case BoundaryType::NO_SLIP:
                u_star_(i, nx-1) = 0.0;
                v_star_(i, nx-1) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_star_(i, nx-1) = grid_->uRight();
                v_star_(i, nx-1) = grid_->vRight();
                break;
            case BoundaryType::OUTFLOW:
                u_star_(i, nx-1) = u_star_(i, nx-2);
                v_star_(i, nx-1) = v_star_(i, nx-2);
                break;
            case BoundaryType::MOVING_WALL:
                u_star_(i, nx-1) = grid_->uRight();
                v_star_(i, nx-1) = grid_->vRight();
                break;
            default:
                break;
        }
    }
    
    // Bottom boundary
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        switch (grid_->bcBottom()) {
            case BoundaryType::NO_SLIP:
                u_star_(0, j) = 0.0;
                v_star_(0, j) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_star_(0, j) = grid_->uBottom();
                v_star_(0, j) = grid_->vBottom();
                break;
            case BoundaryType::OUTFLOW:
                u_star_(0, j) = u_star_(1, j);
                v_star_(0, j) = v_star_(1, j);
                break;
            case BoundaryType::MOVING_WALL:
                u_star_(0, j) = grid_->uBottom();
                v_star_(0, j) = grid_->vBottom();
                break;
            default:
                break;
        }
    }
    
    // Top boundary
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < nx; ++j) {
        switch (grid_->bcTop()) {
            case BoundaryType::NO_SLIP:
                u_star_(ny-1, j) = 0.0;
                v_star_(ny-1, j) = 0.0;
                break;
            case BoundaryType::INFLOW:
                u_star_(ny-1, j) = grid_->uTop();
                v_star_(ny-1, j) = grid_->vTop();
                break;
            case BoundaryType::OUTFLOW:
                u_star_(ny-1, j) = u_star_(ny-2, j);
                v_star_(ny-1, j) = v_star_(ny-2, j);
                break;
            case BoundaryType::MOVING_WALL:
                u_star_(ny-1, j) = grid_->uTop();
                v_star_(ny-1, j) = grid_->vTop();
                break;
            default:
                break;
        }
    }
}

void NavierStokesSolver::computeConvection(double& conv_u, double& conv_v,
                                           size_t i, size_t j) const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    
    // Get local velocities
    double u_ij = u_(i, j);
    double v_ij = v_(i, j);
    
    // Central differences for velocity gradients
    double du_dx = (u_(i, j+1) - u_(i, j-1)) / (2.0 * dx);
    double du_dy = (u_(i+1, j) - u_(i-1, j)) / (2.0 * dy);
    double dv_dx = (v_(i, j+1) - v_(i, j-1)) / (2.0 * dx);
    double dv_dy = (v_(i+1, j) - v_(i-1, j)) / (2.0 * dy);
    
    // Convective terms: (u·∇)u
    conv_u = u_ij * du_dx + v_ij * du_dy;
    conv_v = u_ij * dv_dx + v_ij * dv_dy;
}

void NavierStokesSolver::computeDiffusion(double& diff_u, double& diff_v,
                                          size_t i, size_t j) const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    
    // Laplacian using central differences: ∇²u
    double d2u_dx2 = (u_(i, j-1) - 2.0 * u_(i, j) + u_(i, j+1)) / dx2;
    double d2u_dy2 = (u_(i-1, j) - 2.0 * u_(i, j) + u_(i+1, j)) / dy2;
    double d2v_dx2 = (v_(i, j-1) - 2.0 * v_(i, j) + v_(i, j+1)) / dx2;
    double d2v_dy2 = (v_(i-1, j) - 2.0 * v_(i, j) + v_(i+1, j)) / dy2;
    
    // Diffusive terms: ν∇²u
    diff_u = nu_ * (d2u_dx2 + d2u_dy2);
    diff_v = nu_ * (d2v_dx2 + d2v_dy2);
}

double NavierStokesSolver::computeCFL() const {
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    const double dt = config_.dt;
    
    double max_u = u_.maxAbs();
    double max_v = v_.maxAbs();
    
    double cfl_x = max_u * dt / dx;
    double cfl_y = max_v * dt / dy;
    
    return std::max(cfl_x, cfl_y);
}

double NavierStokesSolver::computeMaxDivergence() const {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    const double dx = grid_->dx();
    const double dy = grid_->dy();
    
    double max_div = 0.0;
    
    #pragma omp parallel for schedule(static) collapse(2) reduction(max:max_div)
    for (size_t i = 1; i < ny - 1; ++i) {
        for (size_t j = 1; j < nx - 1; ++j) {
            double div = (u_(i, j+1) - u_(i, j-1)) / (2.0 * dx) +
                         (v_(i+1, j) - v_(i-1, j)) / (2.0 * dy);
            max_div = std::max(max_div, std::fabs(div));
        }
    }
    
    return max_div;
}

void NavierStokesSolver::exportFields(const std::string& filename_prefix) const {
    const size_t nx = grid_->nx();
    const size_t ny = grid_->ny();
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(config_.output_dir);
    
    // Export velocity field (u)
    {
        std::string filename = config_.output_dir + "/" + filename_prefix + "_u.csv";
        std::ofstream file(filename);
        file << std::scientific << std::setprecision(8);
        
        // Header
        file << "# x-velocity field (u)\n";
        file << "# Grid: " << nx << " x " << ny << "\n";
        file << "# Time: " << current_time_ << "\n";
        
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                if (j > 0) file << ",";
                file << u_(i, j);
            }
            file << "\n";
        }
    }
    
    // Export velocity field (v)
    {
        std::string filename = config_.output_dir + "/" + filename_prefix + "_v.csv";
        std::ofstream file(filename);
        file << std::scientific << std::setprecision(8);
        
        file << "# y-velocity field (v)\n";
        file << "# Grid: " << nx << " x " << ny << "\n";
        file << "# Time: " << current_time_ << "\n";
        
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                if (j > 0) file << ",";
                file << v_(i, j);
            }
            file << "\n";
        }
    }
    
    // Export pressure field
    {
        std::string filename = config_.output_dir + "/" + filename_prefix + "_p.csv";
        std::ofstream file(filename);
        file << std::scientific << std::setprecision(8);
        
        file << "# Pressure field\n";
        file << "# Grid: " << nx << " x " << ny << "\n";
        file << "# Time: " << current_time_ << "\n";
        
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                if (j > 0) file << ",";
                file << p_(i, j);
            }
            file << "\n";
        }
    }
    
    // Export grid coordinates
    {
        std::string filename = config_.output_dir + "/" + filename_prefix + "_grid.csv";
        std::ofstream file(filename);
        file << std::scientific << std::setprecision(8);
        
        file << "# Grid coordinates\n";
        file << "i,j,x,y\n";
        
        for (size_t i = 0; i < ny; ++i) {
            for (size_t j = 0; j < nx; ++j) {
                file << i << "," << j << "," << grid_->x(j) << "," << grid_->y(i) << "\n";
            }
        }
    }
}
