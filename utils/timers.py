import time
from contextlib import contextmanager
from typing import Generator

@contextmanager
def timer() -> Generator[None, None, float]:
    """
    Context manager for timing code execution.
    
    Usage:
        with timer() as t:
            # code to time
        print(f"Time taken: {t} seconds")
    """
    start = time.time()
    yield lambda: time.time() - start
    end = time.time()
    return end - start

class Timer:
    """Class for tracking multiple timing intervals."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
        
    def start(self, name: str) -> None:
        """Start timing an interval."""
        self.start_times[name] = time.time()
        
    def stop(self, name: str) -> float:
        """Stop timing an interval and return the elapsed time."""
        if name not in self.start_times:
            raise KeyError(f"No timer started for {name}")
            
        elapsed = time.time() - self.start_times[name]
        self.times[name] = elapsed
        return elapsed
        
    def get_times(self) -> dict:
        """Return all recorded times."""
        return self.times.copy()
        
    def reset(self) -> None:
        """Reset all timers."""
        self.times.clear()
        self.start_times.clear() 