import numpy as np
from typing import Optional, Tuple, Dict
from core.base_solver import BaseSolver

class JacobiSolver(BaseSolver):
    """Implementation of the Jacobi iterative method for solving Ax = b."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize the solver with common parameters.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        super().__init__(max_iter, tol)
    
    def _compute_component(self, i: int, A: np.ndarray, b: np.ndarray, x: np.ndarray, D_inv: np.ndarray) -> float:
        """Compute a single component of the new iterate."""
        return D_inv[i] * (b[i] - np.dot(A[i, :], x) + A[i, i] * x[i])
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Jacobi method.
        
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
        
        while self.iterations < self.max_iter:
            # Compute new iterate
            x_new = np.zeros(n)
            for i in range(n):
                x_new[i] = self._compute_component(i, A, b, x, D_inv)
            
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
        }
        
        return x, stats 