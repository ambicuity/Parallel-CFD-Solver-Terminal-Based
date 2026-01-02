/**
 * @file main.cpp
 * @brief Terminal-based Parallel CFD Solver main program
 * 
 * Implements a command-line interface for running CFD simulations
 * with configurable parameters and performance benchmarking.
 */

#include "NavierStokesSolver.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <omp.h>
#include <cstring>
#include <vector>
#include <sstream>

/**
 * @brief Command-line argument parser
 */
struct Arguments {
    // Grid parameters
    size_t nx = 64;
    size_t ny = 64;
    double Lx = 1.0;
    double Ly = 1.0;
    
    // Physics parameters
    double Re = 100.0;
    double dt = 0.001;
    double t_end = 1.0;
    
    // Solver parameters
    int pressure_max_iter = 10000;
    double pressure_tol = 1e-6;
    double omega = 1.5;  // SOR relaxation factor
    
    // Problem selection
    std::string problem = "cavity";  // "cavity" or "channel"
    double lid_velocity = 1.0;
    double inflow_velocity = 1.0;
    
    // Output options
    int output_interval = 100;
    bool export_csv = false;
    std::string output_dir = "output";
    bool verbose = true;
    
    // Parallel options
    int num_threads = 0;  // 0 = auto (use OMP_NUM_THREADS or default)
    
    // Benchmarking
    bool benchmark = false;
    
    // Help flag
    bool show_help = false;
};

void printHelp(const char* program_name) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Parallel CFD Solver - 2D Incompressible Navier-Stokes          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Usage: " << program_name << " [options]\n\n";
    
    std::cout << "Grid Parameters:\n";
    std::cout << "  --nx <int>           Grid points in x-direction (default: 64)\n";
    std::cout << "  --ny <int>           Grid points in y-direction (default: 64)\n";
    std::cout << "  --Lx <float>         Domain length in x (default: 1.0)\n";
    std::cout << "  --Ly <float>         Domain length in y (default: 1.0)\n\n";
    
    std::cout << "Physics Parameters:\n";
    std::cout << "  --Re <float>         Reynolds number (default: 100)\n";
    std::cout << "  --dt <float>         Time step (default: 0.001)\n";
    std::cout << "  --t_end <float>      End time (default: 1.0)\n\n";
    
    std::cout << "Solver Parameters:\n";
    std::cout << "  --pressure_iter <int>  Max pressure solver iterations (default: 10000)\n";
    std::cout << "  --pressure_tol <float> Pressure solver tolerance (default: 1e-6)\n";
    std::cout << "  --omega <float>        SOR relaxation factor (default: 1.5)\n\n";
    
    std::cout << "Problem Selection:\n";
    std::cout << "  --problem <string>     Problem type: 'cavity' or 'channel' (default: cavity)\n";
    std::cout << "  --lid_vel <float>      Lid velocity for cavity flow (default: 1.0)\n";
    std::cout << "  --inflow_vel <float>   Inflow velocity for channel flow (default: 1.0)\n\n";
    
    std::cout << "Output Options:\n";
    std::cout << "  --output_interval <int>  Output every N timesteps (default: 100)\n";
    std::cout << "  --export                 Export fields to CSV files\n";
    std::cout << "  --output_dir <string>    Output directory (default: output)\n";
    std::cout << "  --quiet                  Suppress verbose output\n\n";
    
    std::cout << "Parallel Options:\n";
    std::cout << "  --threads <int>        Number of OpenMP threads (0 = auto)\n\n";
    
    std::cout << "Benchmarking:\n";
    std::cout << "  --benchmark            Run performance benchmark (1 vs N threads)\n\n";
    
    std::cout << "Other:\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --nx 128 --ny 128 --Re 1000 --t_end 5.0\n";
    std::cout << "  " << program_name << " --problem channel --Re 100 --export\n";
    std::cout << "  " << program_name << " --benchmark --nx 64 --ny 64\n\n";
}

Arguments parseArguments(int argc, char* argv[]) {
    Arguments args;
    
    auto parseDouble = [](const char* str, const std::string& arg_name) -> double {
        try {
            return std::stod(str);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid value for " << arg_name << ": " << str << "\n";
            std::exit(1);
        }
    };
    
    auto parseULong = [](const char* str, const std::string& arg_name) -> size_t {
        try {
            return std::stoul(str);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid value for " << arg_name << ": " << str << "\n";
            std::exit(1);
        }
    };
    
    auto parseInt = [](const char* str, const std::string& arg_name) -> int {
        try {
            return std::stoi(str);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid value for " << arg_name << ": " << str << "\n";
            std::exit(1);
        }
    };
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        } else if (arg == "--nx" && i + 1 < argc) {
            args.nx = parseULong(argv[++i], "--nx");
        } else if (arg == "--ny" && i + 1 < argc) {
            args.ny = parseULong(argv[++i], "--ny");
        } else if (arg == "--Lx" && i + 1 < argc) {
            args.Lx = parseDouble(argv[++i], "--Lx");
        } else if (arg == "--Ly" && i + 1 < argc) {
            args.Ly = parseDouble(argv[++i], "--Ly");
        } else if (arg == "--Re" && i + 1 < argc) {
            args.Re = parseDouble(argv[++i], "--Re");
        } else if (arg == "--dt" && i + 1 < argc) {
            args.dt = parseDouble(argv[++i], "--dt");
        } else if (arg == "--t_end" && i + 1 < argc) {
            args.t_end = parseDouble(argv[++i], "--t_end");
        } else if (arg == "--pressure_iter" && i + 1 < argc) {
            args.pressure_max_iter = parseInt(argv[++i], "--pressure_iter");
        } else if (arg == "--pressure_tol" && i + 1 < argc) {
            args.pressure_tol = parseDouble(argv[++i], "--pressure_tol");
        } else if (arg == "--omega" && i + 1 < argc) {
            args.omega = parseDouble(argv[++i], "--omega");
        } else if (arg == "--problem" && i + 1 < argc) {
            args.problem = argv[++i];
        } else if (arg == "--lid_vel" && i + 1 < argc) {
            args.lid_velocity = parseDouble(argv[++i], "--lid_vel");
        } else if (arg == "--inflow_vel" && i + 1 < argc) {
            args.inflow_velocity = parseDouble(argv[++i], "--inflow_vel");
        } else if (arg == "--output_interval" && i + 1 < argc) {
            args.output_interval = parseInt(argv[++i], "--output_interval");
        } else if (arg == "--export") {
            args.export_csv = true;
        } else if (arg == "--output_dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--quiet") {
            args.verbose = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            args.num_threads = parseInt(argv[++i], "--threads");
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }
    
    return args;
}

void printHeader() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Parallel CFD Solver - 2D Incompressible Navier-Stokes          ║\n";
    std::cout << "║                   Finite Difference Method                           ║\n";
    std::cout << "║              Red-Black Gauss-Seidel + OpenMP                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

void printConfiguration(const Arguments& args) {
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                        SIMULATION CONFIGURATION\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    
    std::cout << "\n  Grid Parameters:\n";
    std::cout << "    Grid Size:        " << args.nx << " × " << args.ny << "\n";
    std::cout << "    Domain Size:      " << args.Lx << " × " << args.Ly << "\n";
    std::cout << "    Grid Spacing:     dx = " << args.Lx/(args.nx-1) 
              << ", dy = " << args.Ly/(args.ny-1) << "\n";
    
    std::cout << "\n  Physics Parameters:\n";
    std::cout << "    Reynolds Number:  " << args.Re << "\n";
    std::cout << "    Time Step:        " << args.dt << "\n";
    std::cout << "    End Time:         " << args.t_end << "\n";
    std::cout << "    Total Timesteps:  " << static_cast<int>(args.t_end / args.dt) << "\n";
    
    std::cout << "\n  Problem Setup:\n";
    std::cout << "    Problem Type:     " << (args.problem == "cavity" ? "Lid-Driven Cavity" : "Channel Flow") << "\n";
    if (args.problem == "cavity") {
        std::cout << "    Lid Velocity:     " << args.lid_velocity << "\n";
    } else {
        std::cout << "    Inflow Velocity:  " << args.inflow_velocity << "\n";
    }
    
    std::cout << "\n  Solver Parameters:\n";
    std::cout << "    Pressure Max Iter: " << args.pressure_max_iter << "\n";
    std::cout << "    Pressure Tolerance: " << args.pressure_tol << "\n";
    std::cout << "    SOR Omega:         " << args.omega << "\n";
    
    std::cout << "\n  Parallel Configuration:\n";
    std::cout << "    OpenMP Threads:   " << omp_get_max_threads() << "\n";
    
    std::cout << "\n═══════════════════════════════════════════════════════════════════════\n\n";
}

double runSimulation(const Arguments& args, int num_threads, bool verbose = true) {
    // Set number of threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    
    // Create configurations
    GridConfig grid_config(args.nx, args.ny, args.Lx, args.Ly);
    
    SimulationConfig sim_config;
    sim_config.Re = args.Re;
    sim_config.dt = args.dt;
    sim_config.t_end = args.t_end;
    sim_config.output_interval = args.output_interval;
    sim_config.verbose = verbose;
    sim_config.output_dir = args.output_dir;
    
    // Create solver
    NavierStokesSolver solver(grid_config, sim_config);
    
    // Configure pressure solver
    PressureSolverConfig pressure_config;
    pressure_config.max_iterations = args.pressure_max_iter;
    pressure_config.tolerance = args.pressure_tol;
    pressure_config.omega = args.omega;
    solver.setPressureSolverConfig(pressure_config);
    
    // Setup boundary conditions based on problem type
    if (args.problem == "cavity") {
        solver.getGrid().setupLidDrivenCavity(args.lid_velocity);
    } else if (args.problem == "channel") {
        solver.getGrid().setupChannelFlow(args.inflow_velocity);
    }
    
    // Initialize
    solver.initialize();
    
    // Timer
    double total_time = 0.0;
    int total_steps = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Progress callback
    auto progress_callback = [&verbose, &args](const SimulationStats& stats) {
        if (verbose) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Step " << std::setw(6) << stats.timestep 
                      << " | Time: " << std::setw(10) << stats.time
                      << " | max|u|: " << std::setw(10) << stats.max_u
                      << " | max|div|: " << std::scientific << std::setw(10) << stats.max_div
                      << " | P_iter: " << std::setw(5) << stats.pressure_iterations
                      << " | Wall: " << std::fixed << std::setw(8) << stats.wall_time * 1000 << " ms"
                      << "\n";
        }
    };
    
    // Run simulation
    if (verbose) {
        std::cout << "═══════════════════════════════════════════════════════════════════════\n";
        std::cout << "                         SIMULATION PROGRESS\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";
    }
    
    solver.run(progress_callback);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(end_time - start_time).count();
    total_steps = solver.getCurrentStep();
    
    if (verbose) {
        std::cout << "\n═══════════════════════════════════════════════════════════════════════\n";
        std::cout << "                         SIMULATION COMPLETE\n";
        std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";
        
        std::cout << "  Final Statistics:\n";
        std::cout << "    Total Timesteps:     " << total_steps << "\n";
        std::cout << "    Final Time:          " << solver.getCurrentTime() << "\n";
        std::cout << "    Total Wall Time:     " << total_time << " seconds\n";
        std::cout << "    Time per Step:       " << (total_time / total_steps) * 1000.0 << " ms\n";
        std::cout << "    Max |u|:             " << solver.getU().maxAbs() << "\n";
        std::cout << "    Max |v|:             " << solver.getV().maxAbs() << "\n";
        std::cout << "    Max |divergence|:    " << solver.computeMaxDivergence() << "\n";
        std::cout << "    CFL Number:          " << solver.computeCFL() << "\n";
        std::cout << "\n";
    }
    
    // Export fields if requested
    if (args.export_csv) {
        if (verbose) {
            std::cout << "  Exporting fields to CSV...\n";
        }
        
        std::stringstream ss;
        ss << "solution_t" << std::fixed << std::setprecision(4) << solver.getCurrentTime();
        solver.exportFields(ss.str());
        
        if (verbose) {
            std::cout << "    Exported to: " << args.output_dir << "/\n\n";
        }
    }
    
    return total_time;
}

void runBenchmark(const Arguments& args) {
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                       PERFORMANCE BENCHMARK\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";
    
    int max_threads = omp_get_max_threads();
    
    std::cout << "  Running benchmark with grid size: " << args.nx << " × " << args.ny << "\n";
    std::cout << "  Simulation time: " << args.t_end << " seconds\n";
    std::cout << "  Maximum available threads: " << max_threads << "\n\n";
    
    // Run single-threaded baseline
    std::cout << "  Running single-threaded baseline...\n";
    double baseline_time = runSimulation(args, 1, false);
    std::cout << "    Baseline time (1 thread): " << baseline_time << " seconds\n\n";
    
    // Run with multiple thread counts
    std::vector<int> thread_counts;
    for (int t = 2; t <= max_threads; t *= 2) {
        thread_counts.push_back(t);
    }
    if (thread_counts.empty() || thread_counts.back() != max_threads) {
        thread_counts.push_back(max_threads);
    }
    
    std::cout << "  ┌─────────────┬──────────────────┬─────────────┬─────────────────┐\n";
    std::cout << "  │   Threads   │   Time (sec)     │   Speedup   │   Efficiency    │\n";
    std::cout << "  ├─────────────┼──────────────────┼─────────────┼─────────────────┤\n";
    std::cout << "  │      1      │ " << std::setw(16) << std::fixed << std::setprecision(4) 
              << baseline_time << " │    1.00x    │     100.0%      │\n";
    
    double best_speedup = 1.0;
    int best_threads = 1;
    
    for (int threads : thread_counts) {
        std::cout << "  Running with " << threads << " threads...\n";
        double multi_time = runSimulation(args, threads, false);
        double speedup = baseline_time / multi_time;
        double efficiency = speedup / threads * 100.0;
        
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_threads = threads;
        }
        
        std::cout << "  │ " << std::setw(5) << threads << "       │ " 
                  << std::setw(16) << std::fixed << std::setprecision(4) << multi_time 
                  << " │ " << std::setw(7) << std::setprecision(2) << speedup << "x    │ "
                  << std::setw(10) << std::setprecision(1) << efficiency << "%     │\n";
    }
    
    std::cout << "  └─────────────┴──────────────────┴─────────────┴─────────────────┘\n\n";
    
    // Summary using tracked best values
    std::cout << "  Summary:\n";
    std::cout << "    Maximum speedup achieved: " << std::setprecision(2) << best_speedup << "x\n";
    std::cout << "    Using " << best_threads << " threads\n\n";
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    Arguments args = parseArguments(argc, argv);
    
    if (args.show_help) {
        printHelp(argv[0]);
        return 0;
    }
    
    printHeader();
    
    // Set OpenMP threads if specified
    if (args.num_threads > 0) {
        omp_set_num_threads(args.num_threads);
    }
    
    if (args.benchmark) {
        // Run benchmark mode
        runBenchmark(args);
    } else {
        // Print configuration
        if (args.verbose) {
            printConfiguration(args);
        }
        
        // Run normal simulation
        runSimulation(args, args.num_threads, args.verbose);
    }
    
    return 0;
}
