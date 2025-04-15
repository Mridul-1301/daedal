import numpy as np
from typing import Optional, Tuple, Dict
from core.base_solver import BaseSolver

class CGOptimizationSolver(BaseSolver):
    """
    Implementation of the Conjugate Gradient method from the optimization perspective.
    
    This implementation views CG as minimizing the quadratic function:
    f(x) = 1/2 * x^T A x - b^T x
    
    The method generates conjugate directions and minimizes f(x) along each direction.
    """
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Conjugate Gradient method (optimization view).
        
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
        
        # Compute initial gradient (negative residual)
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
            
            # Compute step length (minimizes f(x + alpha*p) along direction p)
            alpha = np.dot(r, r) / np.dot(p, Ap)
            
            # Update solution and gradient
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Compute new conjugate direction
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
        
    def compute_quadratic_value(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        """
        Compute the value of the quadratic function f(x) = 1/2 * x^T A x - b^T x.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            x: Point to evaluate
            
        Returns:
            Value of the quadratic function at x
        """
        return 0.5 * np.dot(x, A @ x) - np.dot(b, x) 