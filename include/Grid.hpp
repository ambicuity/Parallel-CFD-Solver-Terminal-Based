/**
 * @file Grid.hpp
 * @brief Grid class for domain geometry and boundary conditions
 * 
 * Manages computational domain parameters including grid spacing,
 * physical dimensions, and boundary condition types.
 */

#ifndef GRID_HPP
#define GRID_HPP

#include <cstddef>
#include <string>

/**
 * @brief Boundary condition types
 */
enum class BoundaryType {
    NO_SLIP,     // No-slip wall (zero velocity)
    INFLOW,      // Specified inflow velocity
    OUTFLOW,     // Zero-gradient outflow
    PERIODIC,    // Periodic boundary
    MOVING_WALL  // Moving wall (e.g., lid-driven cavity)
};

/**
 * @brief Boundary location enumeration
 */
enum class BoundaryLocation {
    LEFT,
    RIGHT,
    BOTTOM,
    TOP
};

/**
 * @brief Grid configuration structure
 */
struct GridConfig {
    size_t nx;            // Grid points in x-direction
    size_t ny;            // Grid points in y-direction
    double Lx;            // Domain length in x-direction
    double Ly;            // Domain length in y-direction
    
    GridConfig(size_t nx_ = 64, size_t ny_ = 64, 
               double Lx_ = 1.0, double Ly_ = 1.0)
        : nx(nx_), ny(ny_), Lx(Lx_), Ly(Ly_) {}
};

/**
 * @brief Grid class for CFD domain management
 */
class Grid {
private:
    GridConfig config_;
    double dx_;           // Grid spacing in x-direction
    double dy_;           // Grid spacing in y-direction
    
    BoundaryType bc_left_;
    BoundaryType bc_right_;
    BoundaryType bc_bottom_;
    BoundaryType bc_top_;
    
    // Boundary velocities for inflow/moving wall
    double u_left_, u_right_, u_bottom_, u_top_;
    double v_left_, v_right_, v_bottom_, v_top_;
    
public:
    /**
     * @brief Constructor with grid configuration
     */
    explicit Grid(const GridConfig& config)
        : config_(config),
          dx_(config.Lx / (config.nx - 1)),
          dy_(config.Ly / (config.ny - 1)),
          bc_left_(BoundaryType::NO_SLIP),
          bc_right_(BoundaryType::NO_SLIP),
          bc_bottom_(BoundaryType::NO_SLIP),
          bc_top_(BoundaryType::NO_SLIP),
          u_left_(0.0), u_right_(0.0), u_bottom_(0.0), u_top_(0.0),
          v_left_(0.0), v_right_(0.0), v_bottom_(0.0), v_top_(0.0) {}
    
    // Getters for grid parameters
    inline size_t nx() const { return config_.nx; }
    inline size_t ny() const { return config_.ny; }
    inline double Lx() const { return config_.Lx; }
    inline double Ly() const { return config_.Ly; }
    inline double dx() const { return dx_; }
    inline double dy() const { return dy_; }
    
    /**
     * @brief Get x-coordinate at grid point j
     */
    inline double x(size_t j) const { return j * dx_; }
    
    /**
     * @brief Get y-coordinate at grid point i
     */
    inline double y(size_t i) const { return i * dy_; }
    
    /**
     * @brief Set boundary condition
     */
    void setBoundaryCondition(BoundaryLocation loc, BoundaryType type,
                              double u_val = 0.0, double v_val = 0.0) {
        switch (loc) {
            case BoundaryLocation::LEFT:
                bc_left_ = type;
                u_left_ = u_val;
                v_left_ = v_val;
                break;
            case BoundaryLocation::RIGHT:
                bc_right_ = type;
                u_right_ = u_val;
                v_right_ = v_val;
                break;
            case BoundaryLocation::BOTTOM:
                bc_bottom_ = type;
                u_bottom_ = u_val;
                v_bottom_ = v_val;
                break;
            case BoundaryLocation::TOP:
                bc_top_ = type;
                u_top_ = u_val;
                v_top_ = v_val;
                break;
        }
    }
    
    // Boundary condition getters
    inline BoundaryType bcLeft() const { return bc_left_; }
    inline BoundaryType bcRight() const { return bc_right_; }
    inline BoundaryType bcBottom() const { return bc_bottom_; }
    inline BoundaryType bcTop() const { return bc_top_; }
    
    // Boundary velocity getters
    inline double uLeft() const { return u_left_; }
    inline double uRight() const { return u_right_; }
    inline double uBottom() const { return u_bottom_; }
    inline double uTop() const { return u_top_; }
    inline double vLeft() const { return v_left_; }
    inline double vRight() const { return v_right_; }
    inline double vBottom() const { return v_bottom_; }
    inline double vTop() const { return v_top_; }
    
    /**
     * @brief Setup for lid-driven cavity flow
     */
    void setupLidDrivenCavity(double lid_velocity = 1.0) {
        setBoundaryCondition(BoundaryLocation::LEFT, BoundaryType::NO_SLIP);
        setBoundaryCondition(BoundaryLocation::RIGHT, BoundaryType::NO_SLIP);
        setBoundaryCondition(BoundaryLocation::BOTTOM, BoundaryType::NO_SLIP);
        setBoundaryCondition(BoundaryLocation::TOP, BoundaryType::MOVING_WALL, lid_velocity, 0.0);
    }
    
    /**
     * @brief Setup for channel flow
     */
    void setupChannelFlow(double inflow_velocity = 1.0) {
        setBoundaryCondition(BoundaryLocation::LEFT, BoundaryType::INFLOW, inflow_velocity, 0.0);
        setBoundaryCondition(BoundaryLocation::RIGHT, BoundaryType::OUTFLOW);
        setBoundaryCondition(BoundaryLocation::BOTTOM, BoundaryType::NO_SLIP);
        setBoundaryCondition(BoundaryLocation::TOP, BoundaryType::NO_SLIP);
    }
    
    /**
     * @brief Get boundary condition description
     */
    static std::string boundaryTypeToString(BoundaryType type) {
        switch (type) {
            case BoundaryType::NO_SLIP: return "No-Slip Wall";
            case BoundaryType::INFLOW: return "Inflow";
            case BoundaryType::OUTFLOW: return "Outflow";
            case BoundaryType::PERIODIC: return "Periodic";
            case BoundaryType::MOVING_WALL: return "Moving Wall";
            default: return "Unknown";
        }
    }
};

#endif // GRID_HPP
