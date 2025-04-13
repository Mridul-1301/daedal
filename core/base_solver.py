from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List

class BaseSolver(ABC):
    """Abstract base class for all linear system solvers."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize the solver with common parameters.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.residual_history: List[float] = []
        self.iterations: int = 0
        
    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Solve the linear system Ax = b.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            x0: Initial guess (optional)
            
        Returns:
            Tuple containing:
            - Solution vector x
            - Dictionary with solver statistics
        """
    
    def _compute_residual(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
        """Compute the 2-norm of the residual."""
        return np.linalg.norm(b - A @ x)
    
    def _check_convergence(self, residual: float) -> bool:
        """Check if the solver has converged."""
        return residual < self.tol
    
    def get_residual_history(self) -> List[float]:
        """Return the history of residual norms."""
        return self.residual_history 