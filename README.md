# Blendenpik

This is a Python implementation of Blendenpik, a solver of an over-determined system `Ax=b` with a random sampling strategy on `A`.

Haim Avron, Petar Maymounkov, and Sivan Toledo  
Blendenpik: Supercharging LAPACK's Least-Squares Solver   
SIAM J. Sci. Comput., 32(3), 1217–1236. (2010)  
Read More: http://epubs.siam.org/doi/abs/10.1137/090767911

# Environment 
- Python 3.5
- Numpy
- Scipy
- xalglib (Option: Used for Discrete Hartley transform)
- PySPQR (Option: Used for QR decomposition)

# Usage

```
blendenpik = Blendenpik(A, b)
x = blendenpik.solve()
```

`A` is a scipy sparse matrix or numpy matrix, and `b` is a numpy array. You can specify `gamma` for sampling rate from `A`.

# Result

A: 2000-by-1000 matrix with density 0.1. (not so fast)

- Residual (L2-norm): 0.000342662818291
- Computational time (sec.): 4.8125

