#Context-Dependent Space Filling Curves

- Why: 2D gridded data are harder to work with than 1D equally spaced data.
- How: A space-filling curve algorithm inspired by the work of Dafner, Cohen-Or and Matias (2000).
- What: Unlike Hilbert and Peano curves, which are universal (i.e., context-independent), the algorithm proposed here adapts to the data provided as an input matrix. This matrix can be rectangular, and may even contain missing values (under certain constraints), as shown below. Results are both mesmerizing (IMO…) and potentially useful to model 2D gridded data as if it were 1D, while preserving some locality (spatial autocorrelation).
