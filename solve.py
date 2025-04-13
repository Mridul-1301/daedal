#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Dict, Type
import scipy.sparse as sp

from linear_solver_project.core.problem import Problem
from linear_solver_project.core.base_solver import BaseSolver
from linear_solver_project.methods.jacobi import JacobiSolver
from linear_solver_project.methods.gauss_seidel import GaussSeidelSolver
from linear_solver_project.methods.cg_base import ConjugateGradientSolver
from linear_solver_project.methods.cg_optimization import CGOptimizationSolver
from linear_solver_project.methods.cg_projection import CGProjectionSolver
from linear_solver_project.methods.cg_krylov import CGKrylovSolver
from linear_solver_project.utils.timers import Timer
from linear_solver_project.utils.memory_profiler import MemoryProfiler
from linear_solver_project.utils.plotters import plot_residual_history, plot_comparison, plot_performance_comparison

# Map of available solvers
SOLVERS: Dict[str, Type[BaseSolver]] = {
    'jacobi': JacobiSolver,
    'gauss-seidel': GaussSeidelSolver,
    'cg': ConjugateGradientSolver,
    'cg-opt': CGOptimizationSolver,
    'cg-proj': CGProjectionSolver,
    'cg-krylov': CGKrylovSolver
}

def create_example_problem(example_type: str, size: int) -> Problem:
    """Create an example linear system based on the specified type."""
    if example_type == 'diagonally-dominant':
        # Create a diagonally dominant matrix
        A = np.random.rand(size, size)
        A = A + size * np.eye(size)  # Make it diagonally dominant
        b = np.random.rand(size)
        
    elif example_type == 'ill-conditioned':
        # Create an ill-conditioned matrix
        A = np.random.rand(size, size)
        A = A @ A.T  # Make it symmetric
        A = A + 1e-6 * np.eye(size)  # Make it slightly ill-conditioned
        b = np.random.rand(size)
        
    elif example_type == 'sparse':
        # Create a sparse SPD matrix
        A = sp.random(size, size, density=0.1, format='csr')
        A = A @ A.T + sp.diags(np.ones(size), 0).toarray()
        b = np.random.rand(size)
        
    else:
        raise ValueError(f"Unknown example type: {example_type}")
        
    return Problem(A, b)

def main():
    parser = argparse.ArgumentParser(description='Linear System Solver')
    parser.add_argument('--method', choices=SOLVERS.keys(), required=True,
                      help='Solver method to use')
    parser.add_argument('--example', choices=['diagonally-dominant', 'ill-conditioned', 'sparse'],
                      default='diagonally-dominant', help='Example problem type')
    parser.add_argument('--size', type=int, default=100,
                      help='Size of the linear system')
    parser.add_argument('--max-iter', type=int, default=1000,
                      help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-6,
                      help='Convergence tolerance')
    parser.add_argument('--num-threads', type=int, default=4,
                      help='Number of threads for parallel computation (Jacobi method only)')
    parser.add_argument('--plot', action='store_true',
                      help='Plot residual history')
    parser.add_argument('--compare', action='store_true',
                      help='Compare all solvers')
    
    args = parser.parse_args()
    
    # Create the problem
    problem = create_example_problem(args.example, args.size)
    
    if args.compare:
        # Compare all solvers
        results = {}
        times = {}
        memory = {}
        
        for solver_name, solver_class in SOLVERS.items():
            print(f"\nRunning {solver_name} solver...")
            
            # Initialize solver with appropriate parameters
            if solver_name == 'jacobi':
                solver = solver_class(max_iter=args.max_iter, tol=args.tol, num_threads=args.num_threads)
            else:
                solver = solver_class(max_iter=args.max_iter, tol=args.tol)
            
            # Solve with timing and memory tracking
            timer = Timer()
            mem_profiler = MemoryProfiler()
            
            timer.start('solve')
            mem_profiler.measure('start')
            
            x, stats = solver.solve(problem.A, problem.b)
            
            solve_time = timer.stop('solve')
            peak_memory = mem_profiler.measure('end')
            
            # Store results
            results[solver_name] = solver.get_residual_history()
            times[solver_name] = solve_time
            memory[solver_name] = peak_memory
            
            # Print results
            print(f"Time taken: {solve_time:.3f} seconds")
            print(f"Peak memory: {peak_memory:.1f} MB")
            print(f"Iterations: {stats['iterations']}")
            print(f"Final residual: {stats['final_residual']:.2e}")
            print(f"Converged: {stats['converged']}")
            
            # Print additional stats for Jacobi solver
            if solver_name == 'jacobi':
                print(f"Number of threads: {stats['num_threads']}")
            
            # Print additional stats for CG variants
            if solver_name.startswith('cg-'):
                if solver_name == 'cg-opt':
                    quad_value = solver.compute_quadratic_value(problem.A, problem.b, x)
                    print(f"Quadratic function value: {quad_value:.6e}")
                elif solver_name == 'cg-proj':
                    proj_error = solver.compute_projection_error(problem.A, problem.b, x)
                    print(f"Projection error: {proj_error:.6e}")
                elif solver_name == 'cg-krylov':
                    # For Krylov solver, we need an exact solution to compute A-norm error
                    # Here we use a direct solver as a reference
                    x_exact = np.linalg.solve(problem.A, problem.b)
                    a_norm_error = solver.compute_a_norm_error(problem.A, x, x_exact)
                    print(f"A-norm error: {a_norm_error:.6e}")
        
        # Plot comparison
        if args.plot:
            plot_comparison(results, title=f"Solver Comparison - {args.example} Example")
            plot_performance_comparison(times, memory, title=f"Performance Comparison - {args.example} Example")
    
    else:
        # Run a single solver
        solver_class = SOLVERS[args.method]
        
        # Initialize solver with appropriate parameters
        if args.method == 'jacobi':
            solver = solver_class(max_iter=args.max_iter, tol=args.tol, num_threads=args.num_threads)
        else:
            solver = solver_class(max_iter=args.max_iter, tol=args.tol)
        
        # Solve with timing and memory tracking
        timer = Timer()
        mem_profiler = MemoryProfiler()
        
        timer.start('solve')
        mem_profiler.measure('start')
        
        x, stats = solver.solve(problem.A, problem.b)
        
        solve_time = timer.stop('solve')
        peak_memory = mem_profiler.measure('end')
        
        # Print results
        print(f"\nResults for {args.method} solver:")
        print(f"Problem size: {args.size}x{args.size}")
        print(f"Example type: {args.example}")
        print(f"Time taken: {solve_time:.3f} seconds")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"Iterations: {stats['iterations']}")
        print(f"Final residual: {stats['final_residual']:.2e}")
        print(f"Converged: {stats['converged']}")
        
        # Print additional stats for Jacobi solver
        if args.method == 'jacobi':
            print(f"Number of threads: {stats['num_threads']}")
        
        # Print additional stats for CG variants
        if args.method.startswith('cg-'):
            if args.method == 'cg-opt':
                quad_value = solver.compute_quadratic_value(problem.A, problem.b, x)
                print(f"Quadratic function value: {quad_value:.6e}")
            elif args.method == 'cg-proj':
                proj_error = solver.compute_projection_error(problem.A, problem.b, x)
                print(f"Projection error: {proj_error:.6e}")
            elif args.method == 'cg-krylov':
                # For Krylov solver, we need an exact solution to compute A-norm error
                # Here we use a direct solver as a reference
                x_exact = np.linalg.solve(problem.A, problem.b)
                a_norm_error = solver.compute_a_norm_error(problem.A, x, x_exact)
                print(f"A-norm error: {a_norm_error:.6e}")
        
        # Plot if requested
        if args.plot:
            plot_residual_history(solver.get_residual_history(),
                                title=f"Residual History - {args.method}")

if __name__ == '__main__':
    main() 