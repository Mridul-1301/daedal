# Linear System Solver

A modular, extensible Python codebase for solving linear systems of the form Ax = b using various iterative methods.

## Features

- Implementation of three major iterative methods:
  - Jacobi Method
  - Gauss-Seidel Method
  - Conjugate Gradient Method (with multiple interpretations)
- Performance benchmarking (time + memory)
- Visualization tools for convergence analysis
- Example problems demonstrating different matrix properties

## Project Structure

```
linear_solver_project/
├── core/
│   ├── base_solver.py        # Abstract Solver class
│   └── problem.py            # Linear system problem class
├── methods/
│   ├── jacobi.py            # Jacobi method
│   ├── gauss_seidel.py      # Gauss-Seidel method
│   ├── cg_base.py           # CG algorithm (core loop)
│   ├── cg_optimization.py   # CG as minimizer of quadratic
│   ├── cg_projection.py     # CG as Galerkin projection
│   └── cg_krylov.py         # CG using Krylov subspace logic
├── utils/
│   ├── timers.py            # Timing utilities
│   ├── memory_profiler.py   # Memory profiling
│   └── plotters.py          # Visualization tools
├── examples/
│   ├── ill_conditioned_example.py
│   ├── diagonally_dominant_example.py
│   └── sparse_system_example.py
└── solve.py                 # CLI entry point
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main entry point is `solve.py`, which provides a command-line interface for selecting the solver and problem type:

```bash
python solve.py --method [jacobi|gauss-seidel|cg|cg-opt|cg-proj|cg-krylov] \
                --example [diagonally-dominant|ill-conditioned|sparse] \
                --size 100 \
                --max-iter 1000 \
                --tol 1e-6 \
                --plot
```

To compare all solvers:
```bash
python solve.py --compare --example sparse --size 200 --plot
```

### Conjugate Gradient Implementations

The project includes four different implementations of the Conjugate Gradient method, each highlighting a different mathematical interpretation:

1. **Base CG** (`cg_base.py`): Standard implementation focusing on the algorithm's core loop.
2. **Optimization CG** (`cg_optimization.py`): Views CG as minimizing the quadratic function f(x) = 1/2 * x^T A x - b^T x.
3. **Projection CG** (`cg_projection.py`): Implements CG as a Galerkin projection method that finds the solution in the Krylov subspace.
4. **Krylov CG** (`cg_krylov.py`): Emphasizes the Krylov subspace generation and A-norm minimization properties.

### Example Problems

1. **Diagonally Dominant**: Demonstrates Gauss-Seidel's faster convergence
2. **Ill-Conditioned**: Shows CG's robustness
3. **Sparse SPD**: Highlights CG's efficiency for large sparse systems

## Performance Comparison

Each solver has its strengths:

- **Jacobi**: Most parallelizable, good for distributed systems
- **Gauss-Seidel**: Faster convergence for diagonally dominant systems
- **Conjugate Gradient**: Optimal for large, sparse, symmetric positive definite systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 