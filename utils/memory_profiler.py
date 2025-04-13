import os
from typing import Dict, Generator
import psutil
from contextlib import contextmanager

@contextmanager
def memory_tracker() -> Generator[Dict[str, float], None, None]:
    """
    Context manager for tracking memory usage.
    
    Usage:
        with memory_tracker() as mem:
            # code to track
        print(f"Memory used: {mem['peak']} MB")
    """
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    peak_mem = start_mem
    
    def get_current_mem() -> float:
        return process.memory_info().rss / 1024 / 1024
    
    yield {
        'start': start_mem,
        'current': get_current_mem,
        'peak': peak_mem
    }
    
    end_mem = get_current_mem()
    return {
        'start': start_mem,
        'end': end_mem,
        'peak': max(peak_mem, end_mem)
    }

class MemoryProfiler:
    """Class for tracking memory usage across multiple points."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements = {}
        
    def measure(self, name: str) -> float:
        """
        Take a memory measurement at a specific point.
        
        Args:
            name: Identifier for the measurement point
            
        Returns:
            Current memory usage in MB
        """
        mem_usage = self.process.memory_info().rss / 1024 / 1024
        self.measurements[name] = mem_usage
        return mem_usage
        
    def get_measurements(self) -> Dict[str, float]:
        """Return all memory measurements."""
        return self.measurements.copy()
        
    def get_peak(self) -> float:
        """Return the peak memory usage across all measurements."""
        return max(self.measurements.values()) if self.measurements else 0.0
        
    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear() 