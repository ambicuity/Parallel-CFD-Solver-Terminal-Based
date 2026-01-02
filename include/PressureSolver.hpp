/**
 * @file PressureSolver.hpp
 * @brief Pressure Poisson equation solver using Red-Black Gauss-Seidel
 * 
 * Implements a parallel Red-Black Gauss-Seidel iterative solver for the
 * pressure Poisson equation in incompressible flow simulations.
 */

#ifndef PRESSURE_SOLVER_HPP
#define PRESSURE_SOLVER_HPP

#include "Field2D.hpp"
#include "Grid.hpp"

/**
 * @brief Configuration for pressure solver
 */
struct PressureSolverConfig {
    int max_iterations;      // Maximum number of iterations
    double tolerance;        // Convergence tolerance
    double omega;            // Relaxation factor (1.0 for standard GS)
    
    PressureSolverConfig()
        : max_iterations(10000), tolerance(1e-6), omega(1.0) {}
};

/**
 * @brief Pressure Poisson solver using Red-Black Gauss-Seidel
 */
class PressureSolver {
private:
    const Grid& grid_;
    PressureSolverConfig config_;
    
    // Coefficients for the discrete Laplacian
    double dx2_inv_;
    double dy2_inv_;
    double coeff_center_;
    
    int last_iterations_;
    double last_residual_;
    
public:
    /**
     * @brief Constructor
     * @param grid Reference to computational grid
     * @param config Solver configuration
     */
    PressureSolver(const Grid& grid, const PressureSolverConfig& config = PressureSolverConfig());
    
    /**
     * @brief Solve pressure Poisson equation: ∇²p = RHS
     * @param pressure Pressure field (input/output)
     * @param rhs Right-hand side field
     * @return True if converged, false otherwise
     */
    bool solve(Field2D& pressure, const Field2D& rhs);
    
    /**
     * @brief Get last iteration count
     */
    int getLastIterations() const { return last_iterations_; }
    
    /**
     * @brief Get last residual
     */
    double getLastResidual() const { return last_residual_; }
    
    /**
     * @brief Set solver configuration
     */
    void setConfig(const PressureSolverConfig& config) { config_ = config; }
    
private:
    /**
     * @brief Apply boundary conditions to pressure field
     */
    void applyBoundaryConditions(Field2D& pressure);
    
    /**
     * @brief Perform one Red-Black Gauss-Seidel iteration
     * @param pressure Pressure field
     * @param rhs Right-hand side
     * @param color 0 for red, 1 for black
     */
    void redBlackIteration(Field2D& pressure, const Field2D& rhs, int color);
    
    /**
     * @brief Compute residual norm
     */
    double computeResidual(const Field2D& pressure, const Field2D& rhs);
};

#endif // PRESSURE_SOLVER_HPP
