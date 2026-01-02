# Parallel CFD Solver - Makefile
# 2D Incompressible Navier-Stokes Solver with OpenMP

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic
OPTFLAGS = -O3
OMPFLAGS = -fopenmp
INCLUDES = -I./include

# Linker flags
LDFLAGS = -fopenmp

# Source files
SRC_DIR = src
SOURCES = $(SRC_DIR)/main.cpp \
          $(SRC_DIR)/NavierStokesSolver.cpp \
          $(SRC_DIR)/PressureSolver.cpp

# Object files
OBJ_DIR = obj
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Output executable
TARGET = cfd_solver

# Debug build
DEBUG_FLAGS = -g -DDEBUG -O0

# Phony targets
.PHONY: all clean debug release test benchmark help

# Default target
all: release

# Release build (optimized)
release: CXXFLAGS += $(OPTFLAGS)
release: $(TARGET)
	@echo "✓ Release build complete: $(TARGET)"

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)
	@echo "✓ Debug build complete: $(TARGET)"

# Link executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# Create object directory
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR) $(TARGET) output/

# Run test simulation (lid-driven cavity)
test: release
	@echo "\n═══════════════════════════════════════════════════════════════════════"
	@echo "                    Running Test: Lid-Driven Cavity"
	@echo "═══════════════════════════════════════════════════════════════════════\n"
	./$(TARGET) --nx 32 --ny 32 --Re 100 --dt 0.001 --t_end 0.1 --export

# Run benchmark
benchmark: release
	@echo "\n═══════════════════════════════════════════════════════════════════════"
	@echo "                       Running Performance Benchmark"
	@echo "═══════════════════════════════════════════════════════════════════════\n"
	./$(TARGET) --benchmark --nx 64 --ny 64 --t_end 0.5

# Quick validation test
validate: release
	@echo "\nRunning quick validation test..."
	./$(TARGET) --nx 16 --ny 16 --Re 100 --dt 0.001 --t_end 0.05 --quiet

# Help
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║                 Parallel CFD Solver - Build System                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build optimized release version"
	@echo "  make release  - Build optimized release version"
	@echo "  make debug    - Build debug version with symbols"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Run test simulation"
	@echo "  make benchmark- Run performance benchmark"
	@echo "  make validate - Run quick validation test"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Build options:"
	@echo "  CXX=<compiler>  - Specify compiler (default: g++)"
	@echo ""
	@echo "Example:"
	@echo "  make release CXX=g++-11"
	@echo ""
