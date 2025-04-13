import numpy as np
from typing import Optional, Tuple, Dict, List
from linear_solver_project.core.base_solver import BaseSolver

class CGKrylovSolver(BaseSolver):
    """
    Implementation of the Conjugate Gradient method from the Krylov subspace perspective.
    
    This implementation views CG as a method that generates an orthogonal basis for the
    Krylov subspace K_k(A, r_0) = span{r_0, Ar_0, A^2r_0, ..., A^(k-1)r_0} and finds
    the solution that minimizes the A-norm of the error in this subspace.
    """
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize the solver with common parameters."""
        super().__init__(max_iter, tol)
        self.krylov_vectors: List[np.ndarray] = []
        self.a_inner_products: List[float] = []
    
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve the linear system using the Conjugate Gradient method (Krylov view).
        
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
        
        # Store the Krylov vectors and their A-inner products
        self.krylov_vectors = [p.copy()]
        self.a_inner_products = []
        
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
            
            # Compute A-inner product of p with itself
            pAp = np.dot(p, Ap)
            self.a_inner_products.append(pAp)
            
            # Compute step length
            alpha = np.dot(r, r) / pAp
            
            # Update solution and residual
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Compute new search direction (A-conjugate to previous directions)
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            
            # Store the new Krylov vector
            self.krylov_vectors.append(p.copy())
            
            r = r_new
            self.iterations += 1
            
        stats = {
            'iterations': self.iterations,
            'final_residual': residual,
            'converged': self.iterations < self.max_iter,
            'krylov_dimension': len(self.krylov_vectors)
        }
        
        return x, stats
        
    def get_krylov_vectors(self) -> List[np.ndarray]:
        """
        Return the Krylov vectors generated during the solution process.
        
        Returns:
            List of Krylov vectors
        """
        return self.krylov_vectors
        
    def compute_a_norm_error(self, A: np.ndarray, x: np.ndarray, x_exact: np.ndarray) -> float:
        """
        Compute the A-norm of the error: ||x - x_exact||_A = sqrt((x - x_exact)^T A (x - x_exact))
        
        Args:
            A: Coefficient matrix
            x: Computed solution
            x_exact: Exact solution
            
        Returns:
            A-norm of the error
        """
        e = x - x_exact
        return np.sqrt(np.dot(e, A @ e))
        
    def compute_krylov_approximation(self, A: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
        """
        Compute the solution in the k-dimensional Krylov subspace.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            k: Dimension of the Krylov subspace
            
        Returns:
            Solution in the k-dimensional Krylov subspace
        """
        if k > len(self.krylov_vectors):
            raise ValueError(f"Requested dimension {k} exceeds the generated Krylov subspace dimension {len(self.krylov_vectors)}")
            
        # Use the first k Krylov vectors to form a basis
        basis = self.krylov_vectors[:k]
        
        # Form the matrix of A-inner products
        V = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                V[i, j] = np.dot(basis[i], A @ basis[j])
                
        # Form the right-hand side vector
        c = np.zeros(k)
        for i in range(k):
            c[i] = np.dot(b, basis[i])
            
        # Solve the reduced system
        coeffs = np.linalg.solve(V, c)
        
        # Form the solution
        x = np.zeros_like(b)
        for i in range(k):
            x += coeffs[i] * basis[i]
            
        return x 