import numpy as np
from typing import Optional, Tuple, Dict
from core.base_solver import BaseSolver

class MultilevelSolver(BaseSolver):
    """Implementation of a vanilla Multilevel method for solving Ax = b.
    
    This is a simple two-level method that uses restriction to a coarse grid,
    solves the coarse system, and then interpolates back to the fine grid.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, 
                 pre_smoothing_steps: int = 2, post_smoothing_steps: int = 2,
                 coarse_solver: Optional[BaseSolver] = None):
        """
        Initialize the multilevel solver.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            pre_smoothing_steps: Number of pre-smoothing steps
            post_smoothing_steps: Number of post-smoothing steps
            coarse_solver: Solver to use for the coarse grid (if None, use direct solve)
        """
        super().__init__(max_iter, tol)
        self.pre_smoothing_steps = pre_smoothing_steps
        self.post_smoothing_steps = post_smoothing_steps
        self.coarse_solver = coarse_solver
    
    def _smooth(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Apply Gauss-Seidel smoothing.
        
        Args:
            A: System matrix
            b: Right-hand side
            x: Current solution
            num_steps: Number of smoothing steps
            
        Returns:
            Updated solution after smoothing
        """
        n = len(b)
        for _ in range(num_steps):
            for i in range(n):
                x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        return x
    
    def _restrict(self, A: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restrict the residual and system matrix to a coarser grid.
        For simplicity, we use a simple restriction by taking every other point.
        
        Args:
            A: Fine grid system matrix
            r: Fine grid residual
            
        Returns:
            Tuple containing:
            - Coarse grid system matrix
            - Coarse grid residual
        """
        n = len(r)
        nc = n // 2 if n % 2 == 0 else n // 2 + 1
        
        # Simple restriction operator (taking every other point)
        R = np.zeros((nc, n))
        for i in range(nc):
            R[i, 2*i] = 1.0
        
        # Restrict residual and matrix
        r_c = R @ r
        A_c = R @ A @ R.T
        
        return A_c, r_c
    
    def _interpolate(self, e_c: np.ndarray, n_fine: int) -> np.ndarray:
        """
        Interpolate the coarse grid error to the fine grid.
        For simplicity, we use linear interpolation.
        
        Args:
            e_c: Coarse grid error
            n_fine: Size of the fine grid
            
        Returns:
            Fine grid error
        """
        n_coarse = len(e_c)
        
        # Simple interpolation operator 
        P = np.zeros((n_fine, n_coarse))
        
        # Fill even-indexed rows directly
        for i in range(0, n_fine, 2):
            if i // 2 < n_coarse:
                P[i, i // 2] = 1.0
        
        # Interpolate for odd-indexed rows
        for i in range(1, n_fine, 2):
            if i // 2 < n_coarse - 1:
                P[i, i // 2] = 0.5
                P[i, i // 2 + 1] = 0.5
            else:
                P[i, i // 2] = 1.0  # At boundary
        
        return P @ e_c
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Multilevel method.
        
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
            # Pre-smoothing
            x = self._smooth(A, b, x.copy(), self.pre_smoothing_steps)
            
            # Compute residual
            r = b - A @ x
            residual_norm = np.linalg.norm(r)
            self.residual_history.append(residual_norm)
            
            # Check convergence
            if self._check_convergence(residual_norm):
                break
            
            # Restrict to coarse grid
            A_c, r_c = self._restrict(A, r)
            
            # Solve coarse grid problem
            if self.coarse_solver is not None:
                e_c, _ = self.coarse_solver.solve(A_c, r_c)
            else:
                # Direct solve on coarse grid
                e_c = np.linalg.solve(A_c, r_c)
            
            # Interpolate error and correct solution
            e = self._interpolate(e_c, n)
            x = x + e
            
            # Post-smoothing
            x = self._smooth(A, b, x.copy(), self.post_smoothing_steps)
            
            self.iterations += 1
        
        # Compute final residual
        final_residual = np.linalg.norm(b - A @ x)
        
        stats = {
            'iterations': self.iterations,
            'final_residual': final_residual,
            'converged': final_residual < self.tol,
        }
        
        return x, stats 