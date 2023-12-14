# fgmreswlsir
MATLAB codes for performing FGMRES-WLSIR algorithm for solving weighted least squares problems.

This code can be used to reproduce the experiments in Mixed Precision FGMRES-Based Iterative Refinement for Weighted Least Squares (http://arxiv.org/abs/)


## Included MATLAB files
* **_fgmres_sd.m, fgmres_dq.m_** are functions that run split-preconditioned FGMRES using precisions single/double, and double/quad. Application of the preconditioned coefficient matrix to a vector and the preconditioner to the right-hand-side vector are performed in the higher precision; other computations performed all use the lower precision.  

* **_fgmreswlsir_bidiag.m, fgmreswlsir_qr.m_** is a function that performs FGMRES-based iterative refinement in three precisions using block split diagonal and left QR preconditioners.

* **_test_ss.m_** is an example script for comparing FGMRES-WLSIR (with 3 precisions) using block split diagonal and left preconditioners on matrices in SuiteSparse collection.


## Requirements
* The codes have been developed and tested with MATLAB 2022a.
* The codes require some functions from libraries https://github.com/SrikaraPranesh/Multi_precision_NLA_kernels, https://github.com/higham/chop, and https://github.com/SrikaraPranesh/LowPrecision\_Simulation for half precision QR factorization.
* The codes require the Advanpix Multiprecision Computing Toolbox for extended precision computations. 
A free trial of Advanpix is available for download from https://www.advanpix.com/.


