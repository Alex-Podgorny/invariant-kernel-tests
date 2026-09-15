# Manuscript changes required to match the clean final code

The current `simulation_results_final (1).tex` is not modified here. The clean code intentionally implements the requested final choices, so the manuscript simulation section should be updated before the next final run/results insertion.

## Optimization

- Remove the statement that early-stopping patience is 12 and that an 8-per-group monitoring subset is evaluated every three epochs.
- Replace it by: all learned models are optimized for exactly 30 epochs; there is no early stopping, intermediate model selection, or per-replication diagnostic pass.
- Keep Adam, batch size 20/group, learning rate 0.01, weight decay 1e-4, MMD-power regularizer 1e-3.
- Spectral caps are `(5,5,5)` for all three learned architectures.
- Remove the translation finite-grid equivariance safeguard weight 0.25. The final objective has no such penalty.

## Final finite-S grid

Use `S in {4,8,16,32}` for all three groups.

## Rounded intrinsic model parameters

The code uses one common parameter block across all three groups (with periodic distance on the circle):

- M1 templates: uni sd `0.5`; bi locations `+-1`, sd `0.4`; tri locations `+-1.5,0,+1.5`, sd `0.3`, central amplitude `0.8`. Baseline class probabilities are exactly `(4,3,2)/9`, and `0.25 delta` mass is moved from class 1 to class 3 in group Y.
- M2: centers `+-1.5`, both component sd `0.5`, coefficient sd `0.25`, correlations `+0.8 delta` / `-0.8 delta`.
- M3: positions `(-2,-1,-0.2,1,2)`, all sd `0.2`, amplitudes `(1,0.5,1.5,0.5,1)`, fourth-peak multiplier `1+0.5 delta` in Y.
- M4: 3 distractors; min separation and motif guard `0.5`; nuisance amplitude mean `1`, log-sd `0.2`; nuisance widths `U(0.15,0.30)`; motif amplitude mean `0.5`, log-sd `0.15`; center sd `0.2`; motif peak sd `0.1`; half-separations `0.20 +- 0.10 delta`.

## Affine nuisance

The code now samples genuine truncated Gaussian nuisance variables by rejection sampling. The manuscript wording “truncated Gaussian” can therefore be kept literally; the old implementation clipped normal draws at the endpoints.

## Translation boundary ablation

Main final setting: enlarged halo ON and valid convolutions ON. The code additionally exposes two ablation flags:

- `TRANSLATION_USE_HALO=0`
- `TRANSLATION_USE_VALID_CONV=0`

These should only be described if the ablation is reported.

## Competitors

Only the methods already retained in the manuscript are run. In particular, the old translation Weighted-Haar, medoid-L2scale and learned-V1 implementations are absent from the clean campaign.

## Runtime convention

For learned methods, training time includes bandwidth selection, RFF-map construction, candidate cache and all 30 optimizer epochs. Evaluation time includes the frozen learned-probability forward pass, finite-S representative construction, RFF evaluation and permutation statistic. Each S-specific row reports its own end-to-end cost `train + evaluation`; training is not double-counted inside a row. Fixed competitors include their label-free preprocessing and bandwidth/RFF setup in their evaluation cost. Permutation-index generation is excluded uniformly from all methods.

## Final model labels

Use only `M1`, `M2`, `M3`, `M4` throughout the final manuscript, tables, figures and code references. The historical simulation labels are intentionally absent from the final project and output files.
