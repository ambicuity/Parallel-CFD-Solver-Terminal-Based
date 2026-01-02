/**
 * @file Field2D.hpp
 * @brief 2D Field class for CFD grid data storage
 * 
 * Provides a contiguous 1D array storage for 2D scalar fields
 * optimized for cache-friendly access patterns and OpenMP parallelization.
 */

#ifndef FIELD2D_HPP
#define FIELD2D_HPP

#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <algorithm>

class Field2D {
private:
    std::vector<double> data_;   // Contiguous 1D storage for 2D field
    size_t nx_;                   // Number of grid points in x-direction
    size_t ny_;                   // Number of grid points in y-direction
    
public:
    /**
     * @brief Default constructor
     */
    Field2D() : nx_(0), ny_(0) {}
    
    /**
     * @brief Constructor with grid dimensions
     * @param nx Number of grid points in x-direction
     * @param ny Number of grid points in y-direction
     * @param init_value Initial value for all grid points (default 0.0)
     */
    Field2D(size_t nx, size_t ny, double init_value = 0.0)
        : data_(nx * ny, init_value), nx_(nx), ny_(ny) {}
    
    /**
     * @brief Copy constructor
     */
    Field2D(const Field2D& other) = default;
    
    /**
     * @brief Move constructor
     */
    Field2D(Field2D&& other) noexcept = default;
    
    /**
     * @brief Copy assignment
     */
    Field2D& operator=(const Field2D& other) = default;
    
    /**
     * @brief Move assignment
     */
    Field2D& operator=(Field2D&& other) noexcept = default;
    
    /**
     * @brief Access element at (i, j) - row-major order
     * @param i Row index (y-direction)
     * @param j Column index (x-direction)
     * @return Reference to element
     */
    inline double& operator()(size_t i, size_t j) {
        assert(i < ny_ && j < nx_);
        return data_[i * nx_ + j];
    }
    
    /**
     * @brief Const access element at (i, j)
     */
    inline const double& operator()(size_t i, size_t j) const {
        assert(i < ny_ && j < nx_);
        return data_[i * nx_ + j];
    }
    
    /**
     * @brief Get number of grid points in x-direction
     */
    inline size_t nx() const { return nx_; }
    
    /**
     * @brief Get number of grid points in y-direction
     */
    inline size_t ny() const { return ny_; }
    
    /**
     * @brief Get total number of grid points
     */
    inline size_t size() const { return data_.size(); }
    
    /**
     * @brief Get raw data pointer for efficient access
     */
    inline double* data() { return data_.data(); }
    
    /**
     * @brief Get const raw data pointer
     */
    inline const double* data() const { return data_.data(); }
    
    /**
     * @brief Fill all elements with a value
     */
    void fill(double value) {
        std::fill(data_.begin(), data_.end(), value);
    }
    
    /**
     * @brief Copy data from another field
     */
    void copyFrom(const Field2D& other) {
        assert(nx_ == other.nx_ && ny_ == other.ny_);
        data_ = other.data_;
    }
    
    /**
     * @brief Compute maximum absolute value in the field
     */
    double maxAbs() const {
        double max_val = 0.0;
        for (const auto& val : data_) {
            max_val = std::max(max_val, std::fabs(val));
        }
        return max_val;
    }
    
    /**
     * @brief Compute L2 norm of the field
     */
    double l2Norm() const {
        double sum = 0.0;
        for (const auto& val : data_) {
            sum += val * val;
        }
        return std::sqrt(sum / data_.size());
    }
    
    /**
     * @brief Swap contents with another field
     */
    void swap(Field2D& other) {
        data_.swap(other.data_);
        std::swap(nx_, other.nx_);
        std::swap(ny_, other.ny_);
    }
    
    /**
     * @brief Resize the field
     */
    void resize(size_t nx, size_t ny, double init_value = 0.0) {
        nx_ = nx;
        ny_ = ny;
        data_.assign(nx * ny, init_value);
    }
};

#endif // FIELD2D_HPP
