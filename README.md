# Measuring Irreversibility by Counting — Figure 2 Data Generator

This repository provides the C++ code used to generate the data behind **Figure 2** of  
**“Measuring irreversibility by counting: a random coarse-graining framework.”**

The simulation integrates a 2D Langevin dynamics of interacting particles with:
- WCA repulsion + spring interactions
- A time-dependent electric field \( E(t) = E + E\sin(2\pi t/T) \)
- Periodic boundary conditions, coarse-grained into a `grid_size × grid_size` partition

It outputs CSV files containing the “true entropy production” surrogate and the **random coarse-graining** estimator built from cross-correlations between region counts.

---

## Build

### Linux (g++ with OpenMP)
```bash
g++ -O3 -std=c++17 -fopenmp -o figure2 main.cpp
