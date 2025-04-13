import numpy as np
from typing import Optional, Tuple, Dict
from linear_solver_project.core.base_solver import BaseSolver

class GaussSeidelSolver(BaseSolver):
    """Implementation of the Gauss-Seidel iterative method for solving Ax = b."""
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Gauss-Seidel method.
        
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
        
        self.residual_history = []
        self.iterations = 0
        
        while self.iterations < self.max_iter:
            # Compute new iterate using forward substitution
            x_new = np.zeros_like(x)
            for i in range(n):
                x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - 
                           np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            
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
            'converged': self.iterations < self.max_iter
        }
        
        return x, stats 