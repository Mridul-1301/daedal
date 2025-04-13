from typing import List, Dict
import matplotlib.pyplot as plt

def plot_residual_history(residuals: List[float], title: str = "Residual History", 
                         log_scale: bool = True) -> None:
    """
    Plot the history of residual norms.
    
    Args:
        residuals: List of residual norms
        title: Plot title
        log_scale: Whether to use logarithmic scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(residuals) + 1)
    
    if log_scale:
        plt.semilogy(iterations, residuals, 'b-', label='Residual')
        plt.ylabel('Residual Norm (log scale)')
    else:
        plt.plot(iterations, residuals, 'b-', label='Residual')
        plt.ylabel('Residual Norm')
        
    plt.xlabel('Iteration')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_comparison(residual_histories: Dict[str, List[float]], 
                   title: str = "Solver Comparison") -> None:
    """
    Plot residual histories for multiple solvers on the same graph.
    
    Args:
        residual_histories: Dictionary mapping solver names to their residual histories
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for solver_name, residuals in residual_histories.items():
        iterations = range(1, len(residuals) + 1)
        plt.semilogy(iterations, residuals, label=solver_name)
        
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm (log scale)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_performance_comparison(times: Dict[str, float], 
                              memory: Dict[str, float],
                              title: str = "Performance Comparison") -> None:
    """
    Create a bar plot comparing solver performance metrics.
    
    Args:
        times: Dictionary mapping solver names to their execution times
        memory: Dictionary mapping solver names to their peak memory usage
        title: Plot title
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time comparison
    solvers = list(times.keys())
    time_values = [times[s] for s in solvers]
    ax1.bar(solvers, time_values)
    ax1.set_xlabel('Solver')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Memory comparison
    memory_values = [memory[s] for s in solvers]
    ax2.bar(solvers, memory_values)
    ax2.set_xlabel('Solver')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Memory Usage')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 