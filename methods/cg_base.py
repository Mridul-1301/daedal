import numpy as np
from typing import Optional, Tuple, Dict
from linear_solver_project.core.base_solver import BaseSolver

class ConjugateGradientSolver(BaseSolver):
    """Implementation of the Conjugate Gradient method for solving Ax = b."""
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Conjugate Gradient method.
        
        Args:
            A: Coefficient matrix (must be symmetric positive definite)
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple containing:
            - Solution vector x
            - Dictionary with solver statistics
        """
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Compute initial residual
        r = b - A @ x
        p = r.copy()  # Initial search direction
        
        self.residual_history = []
        self.iterations = 0
        
        while self.iterations < self.max_iter:
            # Compute residual norm
            residual = np.linalg.norm(r)
            self.residual_history.append(residual)
            
            # Check convergence
            if self._check_convergence(residual):
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
            self.iterations += 1
            
        stats = {
            'iterations': self.iterations,
            'final_residual': residual,
            'converged': self.iterations < self.max_iter
        }
        
        return x, stats 