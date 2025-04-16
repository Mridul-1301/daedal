#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Type, List, Tuple, Optional
from matplotlib.animation import FuncAnimation
import scipy.sparse as sp

from core.problem import Problem
from core.base_solver import BaseSolver
from methods.jacobi import JacobiSolver
from methods.gauss_seidel import GaussSeidelSolver
from methods.cg_base import ConjugateGradientSolver
from methods.cg_optimization import CGOptimizationSolver
from methods.cg_projection import CGProjectionSolver
from methods.cg_krylov import CGKrylovSolver
from utils.timers import Timer

# Map of available solvers
SOLVERS: Dict[str, Type[BaseSolver]] = {
    'jacobi': JacobiSolver,
    'gauss-seidel': GaussSeidelSolver,
    'cg': ConjugateGradientSolver,
    'cg-opt': CGOptimizationSolver,
    'cg-proj': CGProjectionSolver,
    'cg-krylov': CGKrylovSolver
}

class InstrumentedSolver:
    """Wrapper class that instruments a solver to track intermediate solutions."""
    
    def __init__(self, solver_class: Type[BaseSolver], max_iter: int = 1000, 
                 tol: float = 1e-6, checkpoint_freq: int = 1):
        """
        Initialize the instrumented solver.
        
        Args:
            solver_class: The solver class to instrument
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            checkpoint_freq: How often to save intermediate solutions (every N iterations)
        """
        self.solver = solver_class(max_iter=max_iter, tol=tol)
        self.checkpoint_freq = checkpoint_freq
        self.solution_history = []
        
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Solve the system while tracking intermediate solutions.
        
        This method overrides the solve method of the base solver to track
        intermediate solutions at specified checkpoints.
        """
        # Create copies to avoid modifying the originals
        A_copy = A.copy()
        b_copy = b.copy()
        x0_copy = x0.copy() if x0 is not None else None
        
        # Get the solver class name to use appropriate tracking strategy
        solver_name = self.solver.__class__.__name__
        
        if solver_name == 'JacobiSolver':
            return self._solve_jacobi(A_copy, b_copy, x0_copy)
        elif solver_name == 'GaussSeidelSolver':
            return self._solve_gauss_seidel(A_copy, b_copy, x0_copy)
        elif 'ConjugateGradient' in solver_name or 'CG' in solver_name:
            return self._solve_cg(A_copy, b_copy, x0_copy)
        else:
            # Default case, just run the solver without tracking
            return self.solver.solve(A_copy, b_copy, x0_copy)
    
    def _solve_jacobi(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Instrumented Jacobi method to track intermediate solutions."""
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Save initial solution
        self.solution_history = [x.copy()]
        
        # Extract diagonal and create D^(-1)
        D = np.diag(A)
        D_inv = 1.0 / D
        
        self.solver.residual_history = []
        iterations = 0
        
        while iterations < self.solver.max_iter:
            # Compute new iterate
            x_new = np.zeros(n)
            for i in range(n):
                x_new[i] = D_inv[i] * (b[i] - np.dot(A[i, :], x) + A[i, i] * x[i])
            
            # Compute residual
            residual = self.solver._compute_residual(A, b, x_new)
            self.solver.residual_history.append(residual)
            
            # Save intermediate solution at checkpoints
            if iterations % self.checkpoint_freq == 0:
                self.solution_history.append(x_new.copy())
            
            # Check convergence
            if self.solver._check_convergence(residual):
                break
            
            x = x_new
            iterations += 1
        
        # Ensure final solution is saved
        if iterations % self.checkpoint_freq != 0:
            self.solution_history.append(x.copy())
        
        stats = {
            'iterations': iterations,
            'final_residual': residual,
            'converged': iterations < self.solver.max_iter,
            'solution_history_count': len(self.solution_history)
        }
        
        self.solver.iterations = iterations
        
        return x, stats
    
    def _solve_gauss_seidel(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Instrumented Gauss-Seidel method to track intermediate solutions."""
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Save initial solution
        self.solution_history = [x.copy()]
        
        self.solver.residual_history = []
        iterations = 0
        
        while iterations < self.solver.max_iter:
            x_new = x.copy()  # Start with current solution
            
            # Update each component in-place
            for i in range(n):
                x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            
            # Compute residual
            residual = self.solver._compute_residual(A, b, x_new)
            self.solver.residual_history.append(residual)
            
            # Save intermediate solution at checkpoints
            if iterations % self.checkpoint_freq == 0:
                self.solution_history.append(x_new.copy())
            
            # Check convergence
            if self.solver._check_convergence(residual):
                break
            
            x = x_new
            iterations += 1
        
        # Ensure final solution is saved
        if iterations % self.checkpoint_freq != 0:
            self.solution_history.append(x.copy())
        
        stats = {
            'iterations': iterations,
            'final_residual': residual,
            'converged': iterations < self.solver.max_iter,
            'solution_history_count': len(self.solution_history)
        }
        
        self.solver.iterations = iterations
        
        return x, stats

    def _solve_cg(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Instrumented CG method to track intermediate solutions."""
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Save initial solution
        self.solution_history = [x.copy()]
        
        # Compute initial residual
        r = b - A @ x
        p = r.copy()  # Initial search direction
        
        self.solver.residual_history = []
        iterations = 0
        
        while iterations < self.solver.max_iter:
            # Compute residual norm
            residual = np.linalg.norm(r)
            self.solver.residual_history.append(residual)
            
            # Save intermediate solution at checkpoints
            if iterations % self.checkpoint_freq == 0:
                self.solution_history.append(x.copy())
            
            # Check convergence
            if self.solver._check_convergence(residual):
                break
            
            # Compute Ap
            Ap = A @ p
            
            # Compute step length
            alpha = np.dot(r, r) / np.dot(p, Ap)
            
            # Update solution and residual
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Compute new search direction
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            
            r = r_new
            iterations += 1
        
        # Ensure final solution is saved
        if iterations % self.checkpoint_freq != 0:
            self.solution_history.append(x.copy())
        
        stats = {
            'iterations': iterations,
            'final_residual': residual,
            'converged': iterations < self.solver.max_iter,
            'solution_history_count': len(self.solution_history)
        }
        
        self.solver.iterations = iterations
        
        return x, stats

    def get_solution_history(self) -> List[np.ndarray]:
        """Return the history of intermediate solutions."""
        return self.solution_history

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
        
    elif example_type == '2d-poisson':
        # Create a 2D Poisson problem
        # This creates a banded matrix with a specific structure
        n = int(np.sqrt(size))  # Grid size
        h = 1.0 / (n + 1)  # Grid spacing
        
        # Create the 2D discrete Laplacian operator
        A = sp.diags([4, -1, -1, -1, -1], [0, 1, -1, n, -n], shape=(n*n, n*n)).toarray()
        
        # Create a sine-wave right-hand side
        x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        b = np.sin(np.pi * x.flatten()) * np.sin(np.pi * y.flatten())
        
    else:
        raise ValueError(f"Unknown example type: {example_type}")
        
    return Problem(A, b)

def visualize_solution_evolution(solution_history: List[np.ndarray], problem: Problem, title: str):
    """Visualize how the solution evolves during iterations."""
    n = len(solution_history[0])
    iterations = len(solution_history)
    
    # Get exact solution for reference
    x_exact = np.linalg.solve(problem.A, problem.b)
    
    # If solution is small enough (2D or 3D), show solution trajectory
    if n <= 3:
        fig = plt.figure(figsize=(10, 8))
        
        if n == 2:
            # 2D plot
            ax = fig.add_subplot(111)
            
            # Plot solution trajectory
            x_values = [sol[0] for sol in solution_history]
            y_values = [sol[1] for sol in solution_history]
            
            ax.plot(x_values, y_values, 'b-', marker='o', markersize=4, alpha=0.7)
            
            # Highlight start and end points
            ax.plot(x_values[0], y_values[0], 'go', markersize=8, label='Initial')
            ax.plot(x_values[-1], y_values[-1], 'ro', markersize=8, label='Final')
            ax.plot(x_exact[0], x_exact[1], 'ko', markersize=8, label='Exact')
            
            # Add contour plot of the quadratic form
            x_range = np.linspace(min(x_values) - 0.5, max(x_values) + 0.5, 100)
            y_range = np.linspace(min(y_values) - 0.5, max(y_values) + 0.5, 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    xy = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = 0.5 * xy @ problem.A @ xy - xy @ problem.b
            
            ax.contour(X, Y, Z, 20, cmap='viridis', alpha=0.5)
            
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            
        elif n == 3:
            # 3D plot
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot solution trajectory
            x_values = [sol[0] for sol in solution_history]
            y_values = [sol[1] for sol in solution_history]
            z_values = [sol[2] for sol in solution_history]
            
            ax.plot(x_values, y_values, z_values, 'b-', marker='o', markersize=4, alpha=0.7)
            
            # Highlight start and end points
            ax.scatter(x_values[0], y_values[0], z_values[0], c='g', s=100, label='Initial')
            ax.scatter(x_values[-1], y_values[-1], z_values[-1], c='r', s=100, label='Final')
            ax.scatter(x_exact[0], x_exact[1], x_exact[2], c='k', s=100, label='Exact')
            
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.set_zlabel('x[2]')
        
        ax.set_title(f'Solution Trajectory - {title}')
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    # Error norm evolution
    plt.figure(figsize=(12, 6))
    
    # Calculate error for each iteration
    errors = [np.linalg.norm(sol - x_exact) / np.linalg.norm(x_exact) for sol in solution_history]
    
    plt.semilogy(range(len(errors)), errors, 'b-', marker='o')
    plt.xlabel('Checkpoint Iteration')
    plt.ylabel('Relative Error (log scale)')
    plt.title(f'Error Evolution - {title}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Component-wise evolution
    if n > 3:
        # Too many components to show all, select a few
        num_components = min(5, n)
        indices = np.linspace(0, n-1, num_components, dtype=int)
        
        plt.figure(figsize=(12, 6))
        
        for idx in indices:
            component_values = [sol[idx] for sol in solution_history]
            plt.plot(range(len(component_values)), component_values, marker='o', label=f'x[{idx}]')
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Checkpoint Iteration')
        plt.ylabel('Component Value')
        plt.title(f'Component Evolution - {title}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Create animation
    if iterations > 1 and n <= 10:  # Only for reasonable sizes
        create_solution_animation(solution_history, problem, title)

def create_solution_animation(solution_history: List[np.ndarray], problem: Problem, title: str):
    """Create an animation showing how the solution evolves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(solution_history[0])
    
    # Get exact solution for reference
    x_exact = np.linalg.solve(problem.A, problem.b)
    
    # Set up the plot
    bar_container = ax.bar(range(n), solution_history[0], alpha=0.7)
    ax.set_ylim([min(min(x_exact), min([min(sol) for sol in solution_history])) - 0.1,
                 max(max(x_exact), max([max(sol) for sol in solution_history])) + 0.1])
    
    # Add exact solution as a line
    line, = ax.plot(range(n), x_exact, 'r--', label='Exact Solution')
    
    # Initialize title
    title_text = ax.set_title(f'Iteration 0 / {len(solution_history) - 1}')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Value')
    ax.legend()
    
    def update(frame):
        """Update function for the animation."""
        # Update the bar heights
        for i, bar in enumerate(bar_container):
            bar.set_height(solution_history[frame][i])
        
        # Update title
        title_text.set_text(f'Iteration {frame} / {len(solution_history) - 1}')
        
        return bar_container
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(solution_history), 
                       interval=200, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Linear System Solver Visualization')
    parser.add_argument('--method', choices=SOLVERS.keys(), default='jacobi',
                      help='Solver method to use')
    parser.add_argument('--example', choices=['diagonally-dominant', 'ill-conditioned', 'sparse', '2d-poisson'],
                      default='diagonally-dominant', help='Example problem type')
    parser.add_argument('--size', type=int, default=10,
                      help='Size of the linear system')
    parser.add_argument('--max-iter', type=int, default=100,
                      help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-6,
                      help='Convergence tolerance')
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                      help='How often to save intermediate solutions (every N iterations)')
    
    args = parser.parse_args()
    
    # Create the problem
    print(f"Creating {args.example} problem of size {args.size}...")
    problem = create_example_problem(args.example, args.size)
    
    # Create the instrumented solver
    solver_class = SOLVERS[args.method]
    print(f"Initializing {args.method} solver with checkpointing...")
    solver = InstrumentedSolver(
        solver_class, 
        max_iter=args.max_iter, 
        tol=args.tol,
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Solve the system
    print("Solving the linear system and tracking intermediate solutions...")
    timer = Timer()
    timer.start('solve')
    x, stats = solver.solve(problem.A, problem.b)
    solve_time = timer.stop('solve')
    
    # Print results
    print(f"\nResults for {args.method} solver:")
    print(f"Problem size: {args.size}x{args.size}")
    print(f"Example type: {args.example}")
    print(f"Time taken: {solve_time:.3f} seconds")
    print(f"Iterations: {stats['iterations']}")
    print(f"Final residual: {stats['final_residual']:.2e}")
    print(f"Converged: {stats['converged']}")
    print(f"Stored solution checkpoints: {stats['solution_history_count']}")
    
    # Get the solution history
    solution_history = solver.get_solution_history()
    
    # Visualize the solution evolution
    print("\nVisualizing solution evolution...")
    visualize_solution_evolution(
        solution_history, 
        problem, 
        f"{args.method.upper()} - {args.example.capitalize()}"
    )

if __name__ == '__main__':
    main() 