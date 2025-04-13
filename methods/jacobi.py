import numpy as np
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from linear_solver_project.core.base_solver import BaseSolver

class JacobiSolver(BaseSolver):
    """Implementation of the Jacobi iterative method for solving Ax = b."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, num_threads: int = 4):
        """
        Initialize the solver with common parameters.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            num_threads: Number of threads to use for parallel computation
        """
        super().__init__(max_iter, tol)
        self.num_threads = num_threads
    
    def _compute_component(self, i: int, A: np.ndarray, b: np.ndarray, x: np.ndarray, D_inv: np.ndarray) -> float:
        """Compute a single component of the new iterate."""
        return D_inv[i] * (b[i] - np.dot(A[i, :], x) + A[i, i] * x[i])
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Jacobi method with parallel computation.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple containing:
            - Solution vector x
            - Dictionary with solver statistics
        """
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Extract diagonal and create D^(-1)
        D = np.diag(A)
        D_inv = 1.0 / D
        
        self.residual_history = []
        self.iterations = 0
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            while self.iterations < self.max_iter:
                # Compute new iterate in parallel
                futures = [
                    executor.submit(self._compute_component, i, A, b, x, D_inv)
                    for i in range(n)
                ]
                x_new = np.array([f.result() for f in futures])
                
                # Compute residual
                residual = self._compute_residual(A, b, x_new)
                self.residual_history.append(residual)
                
                # Check convergence
                if self._check_convergence(residual):
                    break
                    
                x = x_new
                self.iterations += 1
            
        stats = {
            'iterations': self.iterations,
            'final_residual': residual,
            'converged': self.iterations < self.max_iter,
            'num_threads': self.num_threads
        }
        
        return x, stats 