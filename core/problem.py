import numpy as np

class Problem:
    """Class representing a linear system problem Ax = b."""
    
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Initialize a linear system problem.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
        """
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Dimensions of A and b must match")
            
    def get_dimension(self) -> int:
        """Return the dimension of the system."""
        return self.A.shape[0]
    
    def is_symmetric(self) -> bool:
        """Check if the matrix A is symmetric."""
        return np.allclose(self.A, self.A.T)
    
    def is_positive_definite(self) -> bool:
        """Check if the matrix A is positive definite."""
        if not self.is_symmetric():
            return False
        try:
            np.linalg.cholesky(self.A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def get_condition_number(self) -> float:
        """Compute the condition number of matrix A."""
        return np.linalg.cond(self.A)
    
    def get_diagonal_dominance(self) -> float:
        """
        Compute the measure of diagonal dominance.
        Returns the minimum ratio of diagonal element to sum of off-diagonal elements.
        """
        D = np.diag(np.abs(self.A))
        S = np.sum(np.abs(self.A), axis=1) - D
        return np.min(D / (S + 1e-10))  # Add small epsilon to avoid division by zero 