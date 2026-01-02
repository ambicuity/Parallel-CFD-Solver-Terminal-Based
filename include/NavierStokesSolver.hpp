/**
 * @file NavierStokesSolver.hpp
 * @brief 2D Incompressible Navier-Stokes solver using projection method
 * 
 * Implements the fractional step (projection) method for solving
 * 2D incompressible Navier-Stokes equations with explicit time stepping.
 * 
 * The governing equations are:
 *   du/dt + (u.grad)u = -grad(p)/rho + nu*laplacian(u)   (Momentum)
 *   div(u) = 0                                            (Continuity)
 * 
 * The projection method decomposes the solution into three steps:
 *   1. Velocity prediction: Compute intermediate velocity u* without pressure
 *   2. Pressure correction: Solve Poisson equation laplacian(p) = rho*div(u*)/dt
 *   3. Velocity projection: u^(n+1) = u* - dt*grad(p)/rho
 */

#ifndef NAVIER_STOKES_SOLVER_HPP
#define NAVIER_STOKES_SOLVER_HPP

#include "Field2D.hpp"
#include "Grid.hpp"
#include "PressureSolver.hpp"
#include <memory>
#include <string>
#include <functional>

/**
 * @brief Simulation configuration parameters
 */
struct SimulationConfig {
    double Re;               // Reynolds number
    double dt;               // Time step
    double t_end;            // End time
    int output_interval;     // Output frequency (timesteps)
    bool verbose;            // Verbose terminal output
    std::string output_dir;  // Output directory for CSV files
    
    SimulationConfig()
        : Re(100.0), dt(0.001), t_end(10.0),
          output_interval(100), verbose(true), output_dir("output") {}
};

/**
 * @brief Simulation statistics
 */
struct SimulationStats {
    double time;
    int timestep;
    double max_u;
    double max_v;
    double max_div;
    double pressure_residual;
    int pressure_iterations;
    double wall_time;
};

/**
 * @brief 2D Incompressible Navier-Stokes Solver
 */
class NavierStokesSolver {
private:
    // Grid and configuration
    std::unique_ptr<Grid> grid_;
    SimulationConfig config_;
    PressureSolverConfig pressure_config_;
    
    // Velocity fields (u, v)
    Field2D u_;          // x-velocity at current timestep
    Field2D v_;          // y-velocity at current timestep
    Field2D u_star_;     // Intermediate x-velocity
    Field2D v_star_;     // Intermediate y-velocity
    
    // Pressure field
    Field2D p_;
    
    // Right-hand side for pressure Poisson equation
    Field2D rhs_;
    
    // Pressure solver
    std::unique_ptr<PressureSolver> pressure_solver_;
    
    // Physical parameters
    double nu_;          // Kinematic viscosity (1/Re for dimensionless form)
    double rho_;         // Density (1.0 for dimensionless form)
    
    // Simulation state
    double current_time_;
    int current_step_;
    
public:
    /**
     * @brief Constructor
     * @param grid_config Grid configuration
     * @param sim_config Simulation configuration
     */
    NavierStokesSolver(const GridConfig& grid_config, const SimulationConfig& sim_config);
    
    /**
     * @brief Initialize fields with given conditions
     */
    void initialize();
    
    /**
     * @brief Advance solution by one timestep
     * @return Simulation statistics for this timestep
     */
    SimulationStats step();
    
    /**
     * @brief Run simulation until end time
     * @param progress_callback Optional callback for progress updates
     */
    void run(std::function<void(const SimulationStats&)> progress_callback = nullptr);
    
    /**
     * @brief Export fields to CSV files
     * @param filename_prefix Prefix for output files
     */
    void exportFields(const std::string& filename_prefix) const;
    
    /**
     * @brief Get current simulation time
     */
    double getCurrentTime() const { return current_time_; }
    
    /**
     * @brief Get current timestep
     */
    int getCurrentStep() const { return current_step_; }
    
    /**
     * @brief Get grid reference
     */
    const Grid& getGrid() const { return *grid_; }
    
    /**
     * @brief Get mutable grid reference for setup
     */
    Grid& getGrid() { return *grid_; }
    
    /**
     * @brief Get velocity fields (read-only)
     */
    const Field2D& getU() const { return u_; }
    const Field2D& getV() const { return v_; }
    const Field2D& getPressure() const { return p_; }
    
    /**
     * @brief Set pressure solver configuration
     */
    void setPressureSolverConfig(const PressureSolverConfig& config);
    
    /**
     * @brief Compute CFL number for stability check
     */
    double computeCFL() const;
    
    /**
     * @brief Compute maximum divergence of velocity field
     */
    double computeMaxDivergence() const;
    
private:
    /**
     * @brief Step 1: Compute intermediate velocity (prediction step)
     */
    void velocityPrediction();
    
    /**
     * @brief Step 2: Compute pressure field (pressure correction)
     */
    void pressureCorrection();
    
    /**
     * @brief Step 3: Project velocity to divergence-free field
     */
    void velocityProjection();
    
    /**
     * @brief Apply boundary conditions to velocity fields
     */
    void applyBoundaryConditions();
    
    /**
     * @brief Apply boundary conditions to intermediate velocity
     */
    void applyBoundaryConditionsStar();
    
    /**
     * @brief Compute convective term (u·∇)u using central differences
     */
    void computeConvection(double& conv_u, double& conv_v, 
                           size_t i, size_t j) const;
    
    /**
     * @brief Compute diffusive term ν∇²u using central differences
     */
    void computeDiffusion(double& diff_u, double& diff_v,
                          size_t i, size_t j) const;
};

#endif // NAVIER_STOKES_SOLVER_HPP
