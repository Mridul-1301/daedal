import numpy as np
from typing import Optional, Tuple, Dict, List
from core.base_solver import BaseSolver

class GMRESSolver(BaseSolver):
    """
    Implementation of the Generalized Minimal Residual Method (GMRES).
    
    GMRES is a Krylov subspace method for solving non-symmetric linear systems.
    It minimizes the 2-norm of the residual over a Krylov subspace.
    This implementation provides the standard GMRES without restarts.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize the solver with common parameters."""
        super().__init__(max_iter, tol)
        self.arnoldi_vectors: List[np.ndarray] = []
        self.hessenberg: Optional[np.ndarray] = None
        self.cs: np.ndarray = np.zeros(max_iter)  # Cosine values for Givens rotations
        self.sn: np.ndarray = np.zeros(max_iter)  # Sine values for Givens rotations
        
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using GMRES.
        
        Args:
            A: Coefficient matrix (can be non-symmetric)
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple containing:
            - Solution vector x
            - Dictionary with solver statistics
        """
        n = len(b)
        x = x0 if x0 is not None else np.zeros(n)
        
        # Compute initial residual and its norm
        r = b - A @ x
        beta = np.linalg.norm(r)
        
        # Initialize residual history
        self.residual_history = [beta]
        
        # Check for trivial case (b = 0 or already converged)
        if beta < self.tol:
            return x, {'iterations': 0, 'final_residual': beta, 'converged': True}
        
        # Initialize the Arnoldi basis with the normalized residual
        v = r / beta
        self.arnoldi_vectors = [v]
        
        # Initialize the Hessenberg matrix (upper triangular part)
        self.hessenberg = np.zeros((self.max_iter + 1, self.max_iter))
        
        # Initialize the RHS of the least squares problem
        g = np.zeros(self.max_iter + 1)
        g[0] = beta
        
        self.iterations = 0
        
        # Begin GMRES iterations
        for k in range(self.max_iter):
            self.iterations = k + 1
            
            # Arnoldi process to compute the next basis vector
            w = A @ self.arnoldi_vectors[k]
            
            # Orthogonalize w against previous Arnoldi vectors (Modified Gram-Schmidt)
            for j in range(k + 1):
                self.hessenberg[j, k] = np.dot(w, self.arnoldi_vectors[j])
                w = w - self.hessenberg[j, k] * self.arnoldi_vectors[j]
            
            # Get the norm of the orthogonalized vector
            h_next = np.linalg.norm(w)
            
            # If h_next is too small, we have reached invariant subspace
            if abs(h_next) < 1e-12:
                break
                
            self.hessenberg[k + 1, k] = h_next
            
            # Add the new normalized vector to the Arnoldi basis
            self.arnoldi_vectors.append(w / h_next)
            
            # Apply previous Givens rotations to the new column of the Hessenberg matrix
            for i in range(k):
                # Apply the i-th Givens rotation to (i, k) and (i+1, k) elements
                temp = self.hessenberg[i, k]
                self.hessenberg[i, k] = self.cs[i] * temp + self.sn[i] * self.hessenberg[i + 1, k]
                self.hessenberg[i + 1, k] = -self.sn[i] * temp + self.cs[i] * self.hessenberg[i + 1, k]
            
            # Apply Givens rotation for current iteration
            self._apply_givens_rotation(k, g)
            
            # Update the residual norm
            residual = abs(g[k + 1])
            self.residual_history.append(residual)
            
            # Check convergence
            if self._check_convergence(residual):
                break
        
        # Solve the upper triangular system to get coefficients of the solution
        y = self._solve_upper_triangular(self.hessenberg[:k + 1, :k + 1], g[:k + 1])
        
        # Compute the solution
        for i in range(k + 1):
            x = x + y[i] * self.arnoldi_vectors[i]
        
        stats = {
            'iterations': self.iterations,
            'final_residual': self.residual_history[-1],
            'converged': self.residual_history[-1] < self.tol,
            'arnoldi_dimension': len(self.arnoldi_vectors)
        }
        
        return x, stats
    
    def _apply_givens_rotation(self, k: int, g: np.ndarray) -> None:
        """
        Apply Givens rotation to the Hessenberg matrix at iteration k to create
        an upper triangular matrix.
        
        Args:
            k: Current iteration index
            g: Right-hand side vector of the least squares problem
        """
        # Compute the Givens rotation coefficients
        h1 = self.hessenberg[k, k]
        h2 = self.hessenberg[k + 1, k]
        denom = np.sqrt(h1**2 + h2**2)
        
        # If values are too small, avoid division by zero
        if denom < 1e-14:
            self.cs[k] = 0.0
            self.sn[k] = 1.0
        else:
            self.cs[k] = h1 / denom  # cosine
            self.sn[k] = h2 / denom  # sine
        
        # Apply the Givens rotation to the Hessenberg matrix and RHS vector
        self.hessenberg[k, k] = self.cs[k] * h1 + self.sn[k] * h2
        self.hessenberg[k + 1, k] = 0.0
        
        # Apply the rotation to the RHS vector
        temp = g[k]
        g[k] = self.cs[k] * temp + self.sn[k] * g[k + 1]
        g[k + 1] = -self.sn[k] * temp + self.cs[k] * g[k + 1]
    
    def _solve_upper_triangular(self, H: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Solve the upper triangular system Hy = g.
        
        Args:
            H: Upper triangular part of the Hessenberg matrix
            g: Right-hand side vector
            
        Returns:
            Solution vector y
        """
        k = H.shape[1]
        y = np.zeros(k)
        
        # Back-substitution
        for i in range(k - 1, -1, -1):
            y[i] = g[i]
            for j in range(i + 1, k):
                y[i] -= H[i, j] * y[j]
            y[i] /= H[i, i]
            
        return y
    
    def get_arnoldi_vectors(self) -> List[np.ndarray]:
        """
        Return the Arnoldi vectors generated during the solution process.
        
        Returns:
            List of orthonormal Arnoldi vectors that form a basis for the Krylov subspace
        """
        return self.arnoldi_vectors
    
    def get_hessenberg_matrix(self) -> np.ndarray:
        """
        Return the Hessenberg matrix generated during the Arnoldi process.
        
        Returns:
            Upper Hessenberg matrix
        """
        k = self.iterations
        return self.hessenberg[:k + 1, :k] 