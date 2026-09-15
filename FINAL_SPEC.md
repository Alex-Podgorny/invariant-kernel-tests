# Final-code decisions relative to the previous scripts

1. The final simulation code exposes only the four model labels `M1`--`M4`.
2. Harmonized M1--M4 intrinsic parameters across R, S^1 and Aff+ using the rounded values in the final translation implementation.
3. Made M1 class probabilities explicit as `(4,3,2)/9`; the old code passed `(.4,.3,.2)` to `sample()`, which silently normalized them to exactly these probabilities.
4. Fixed `S_GRID` to `4,8,16,32` by default.
5. Fixed all spectral caps to `(5,5,5)`.
6. Removed early stopping, monitoring subsets, checkpoint selection and per-replication diagnostics. Training is exactly 30 epochs.
7. Removed the finite-grid equivariance safeguard from the translation objective.
8. Added independent translation ablations for halo and valid convolutions.
9. Removed competitors absent from the manuscript (translation Weighted-Haar and medoid-L2scale, plus all legacy learned-V1 machinery).
10. Replaced clipped affine Gaussian nuisances by rejection-sampled truncated Gaussians.
11. Standardized runtime accounting across all groups and removed cache double-counting.
12. Kept independent RFF realizations for training and final testing; all test transformations/features are frozen before label permutations.

13. Restored the fast D=256 training path in all three geometries: vectorized candidate-RFF construction and one-time Torch transfer of every theta-independent training tensor.
