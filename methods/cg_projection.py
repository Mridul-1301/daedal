import numpy as np
from typing import Optional, Tuple, Dict, List
from core.base_solver import BaseSolver

class CGProjectionSolver(BaseSolver):
    """
    Implementation of the Conjugate Gradient method from the Galerkin projection perspective.
    
    This implementation views CG as a Galerkin projection method that finds the solution
    in the Krylov subspace K_k(A, r_0) = span{r_0, Ar_0, A^2r_0, ..., A^(k-1)r_0}
    by imposing the Galerkin condition that the residual is orthogonal to this subspace.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize the solver with common parameters."""
        super().__init__(max_iter, tol)
        self.krylov_basis: List[np.ndarray] = []
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Conjugate Gradient method (projection view).
        
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
        
        # Store basis vectors for the Krylov subspace
        self.krylov_basis = [p.copy()]
        
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
            
            # Compute new search direction (conjugate to previous directions)
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            
            # Store the new basis vector
            self.krylov_basis.append(p.copy())
            
            r = r_new
            self.iterations += 1
            
        stats = {
            'iterations': self.iterations,
            'final_residual': residual,
            'converged': self.iterations < self.max_iter,
            'krylov_dimension': len(self.krylov_basis)
        }
        
        return x, stats
        
    def get_krylov_basis(self) -> List[np.ndarray]:
        """
        Return the basis vectors for the Krylov subspace.
        
        Returns:
            List of basis vectors spanning the Krylov subspace
        """
        return self.krylov_basis
        
    def compute_projection_error(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        """
        Compute the projection error, which is the norm of the component of the residual
        that lies outside the Krylov subspace.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            x: Current solution
            
        Returns:
            Projection error
        """
        r = b - A @ x
        
        # If we haven't generated any basis vectors yet, return the residual norm
        if not self.krylov_basis:
            return np.linalg.norm(r)
            
        # Compute the projection of r onto the Krylov subspace
        proj_r = np.zeros_like(r)
        for v in self.krylov_basis:
            proj_r += np.dot(r, v) / np.dot(v, v) * v
            
        # Return the norm of the component outside the subspace
        return np.linalg.norm(r - proj_r) 