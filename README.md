## Installation

The reference implementation used for the simulations in the paper runs on CPU.

Clone the repository and open the project root in R.

First, install `renv` if needed:

```r
install.packages("renv")
```

Then restore the project environment and install the Torch backend:

```r
source("setup.R")
```

This will:

* restore the exact R package versions recorded in `renv.lock`;
* check the R `torch` package;
* install the required LibTorch/LibLantern backend if needed;
* use the CPU backend as the reference configuration.

You can verify that Torch is correctly installed with:

```r
torch::torch_is_installed(recheck = TRUE)
```

A successful installation should return:

```r
TRUE
```

## Running the simulations

From the project root, the three simulation experiments can be run with:

```r
source("run/run_translation.R")
source("run/run_circular.R")
source("run/run_affine.R")
```

The default configuration is defined in the project configuration files. Simulation settings can also be overridden through environment variables before running a script.

For example, a small smoke test can be run with:

```r
Sys.setenv(
  QUICK = "1",
  NREP = "1",
  BPERM = "19",
  S_GRID = "4",
  DELTAS = "0",
  SCENARIOS = "M1",
  CORES = "1"
)

source("run/run_translation.R")
```

To return to the default configuration afterwards:

```r
Sys.unsetenv(c(
  "QUICK",
  "NREP",
  "BPERM",
  "S_GRID",
  "DELTAS",
  "SCENARIOS",
  "CORES"
))
```

## Generating the figures

After the translation, circular and affine simulations have been run, the figures used in the paper can be generated with:

```r
source("analysis/make_final_results_figures.R")
```

By default, runtime figures use median computation times to reduce sensitivity to occasional system-level timing outliers.

To use mean runtimes instead:

```r
Sys.setenv(RUNTIME_STAT = "mean")
source("analysis/make_final_results_figures.R")
```





# Invariant-kernel simulations

Clean implementation of the final simulation study for three nuisance groups:

- `R`: aperiodic translations on `[-5,5]`;
- `S^1`: circular translations on 128 grid points;
- `Aff`: positive affine transformations.

The code intentionally contains only the four final intrinsic alternatives and only the competitors reported in the manuscript.

## Layout

- `R/config.R`: single source of truth for shared parameters.
- `R/models.R`: the four final intrinsic models M1--M4.
- `R/generators.R`: nuisance-group data generators; noise is generated before the nuisance action.
- `R/utils.R`: RFF, MMD/permutation, seeds, spectral projection, output helpers.
- `R/methods/translation_methods.R`: low-level translation implementation.
- `R/methods/circular_methods.R`: low-level circular implementation.
- `R/methods/affine_methods.R`: low-level affine implementation.
- `R/methods/*.R`: one small public wrapper per reported method.
- `run/run_translation.R`, `run/run_circular.R`, `run/run_affine.R`: final campaigns.
- `run/run_all.R`: sequentially launches all three campaigns.

## Final shared choices

- intrinsic models: `M1`, `M2`, `M3`, `M4`; these are the only model labels used by the final code and output files;
- `delta = 0,.2,.4,.6,.8,1`;
- learned split: 20 train + 40 test observations per group;
- non-learned competitors: all 60 observations per group;
- multiplicative noise `N(1,0.5^2)` before the nuisance transformation;
- test RFF dimension = training RFF dimension = 256;
- `S = 4,8,16,32`;
- 500 permutations, alpha = .05;
- Adam, batch size 20/group, exactly 30 epochs, learning rate .01, weight decay 1e-4;
- no early stopping, no monitoring subset, no per-replication diagnostics;
- no equivariance safeguard/penalty;
- spectral caps `(5,5,5)` after every optimizer step.

## Shared nuisance parameters

The location-type nuisance is controlled by the same two parameters in all three experiments:

- `SHIFT_GAP = 0.5`: difference between the X and Y nuisance means;
- `SIGMA_SHIFT = 0.8`: nuisance standard deviation within each group.

These values are used for the aperiodic translation on `R`, the circular phase shift on `S^1`, and the translation coordinate `b` of `Aff^+(1)`. The affine translation is truncated to the interval `[-1.6,1.6]`.

The affine scale coordinate has two additional shared-configuration parameters:

- `ALPHA_GAP = 0.1`;
- `ALPHA_SD = 0.1`.

The affine log-scale nuisance is truncated to `[-0.4,0.4]`.

## Rounded intrinsic parameters

The same values are used in all three group experiments. The periodic case only replaces Euclidean Gaussian distance by wrapped distance.

- M1: normalized uni/bi/trimodal templates with widths `0.5`, `0.4`, `0.3`, locations `0`, `+-1`, `+-1.5`, and central trimodal amplitude `0.8`. Historical weights `(.4,.3,.2)` are written explicitly as `(4,3,2)/9`, so the actual distribution is unambiguous.
- M2: two separately L2-normalized Gaussians centered at `+-1.5`, both with sd `0.5`; coefficient sd `.25`; correlations `+-.8 delta`.
- M3: locations `(-2,-1,-.2,1,2)`, all widths `.2`, baseline amplitudes `(1,.3,1.5,.3,1)`, fourth-peak effect `.5 delta`, and multiplicative peak log-sd `.2`.
- M4: three distractors; minimum spacing/guard `.5`; distractor amplitude log-sd `.2`, widths `U(.15,.30)`; motif amplitude `.5` with log-sd `.15`; motif peak width `.1`; half-separation `.20 +- .10 delta`.

## Translation boundary ablations

Two independent environment flags are available:

- `TRANSLATION_USE_HALO=1/0`: enables/disables the enlarged score/representative halo.
- `TRANSLATION_USE_VALID_CONV=1/0`: uses valid convolutions with a receptive-field input guard, or zero-padded same-width convolutions.

Default final setting is `1` for both. Turning them off is an ablation, not the main final specification.

## Timing convention

Every result row uses the same convention.

- Learned methods: `train_seconds` contains all training-side work (training bandwidth, train/test RFF-map construction, candidate cache, 30 optimization epochs). `eval_seconds` contains the complete frozen-model evaluation at that `S`: probability forward pass, sampled representatives, RFF features and permutation statistic. `total_seconds = train_seconds + eval_seconds`.
- Non-learned methods: `train_seconds=0`; `eval_seconds` contains all method-specific label-free preprocessing, bandwidth/RFF setup, representation construction and permutation evaluation. For finite-S fixed probabilistic methods the common bandwidth/RFF setup is attributed to every S-specific end-to-end row.
- The random permutation index plan itself is shared experimental infrastructure and is not timed.

This avoids the previous cache double-counting and the inconsistent treatment of learned probability-forward costs across groups.

## Running from the R console / RStudio

From the project directory:

```r
setwd("C:/path/to/final_simulations_final")
Sys.setenv(QUICK="1", NREP="1", BPERM="19", S_GRID="4", CORES="1")
source("run/run_translation.R")
source("run/run_circular.R")
source("run/run_affine.R")
```

For the final settings, clear the smoke-test overrides first:

```r
Sys.unsetenv(c("QUICK","NREP","BPERM","S_GRID"))
```

`CORES` can then be set independently. Default is one core. The output directory can be overridden with `OUTDIR`.

## Training-speed implementation

The final learned implementations use the same performance strategy in all three groups:

- all candidate representatives are constructed once per fit;
- their RFF features are evaluated in one vectorized RFF call;
- candidate RFF tensors and all fixed CNN/G-CNN inputs are transferred to LibTorch once before the epoch loop;
- with the final `n_train=20` and batch size `20` per group, the complete training fold is a single cached Torch batch;
- there are no R-to-Torch conversions of the full candidate cache inside the 30 optimization epochs.

This restores the optimization pattern used by the earlier fast `D=256` implementation.

To benchmark that path directly from the R console without switching to the reduced QUICK dimensions:

```r
Sys.setenv(
  QUICK="0", NREP="1", BPERM="19", S_GRID="4",
  DELTAS="0", SCENARIOS="M1", CORES="1",
  RFF_DIM="256", RFF_TRAIN_DIM="256"
)
source("run/run_translation.R")
```

The same environment settings can be used with the circular and affine runners.

## Reported competitors

Translation: Base-RFF, Energy-prob-RFF, Align-energy-median-RFF, Align-xcorr-RFF, Learned-prob-CNN-RFF.

Circle: Base-RFF, Circular-Haar-RFF, Circular-Energy-prob, Align-energy-circular-median, Align-xcorr-circular, Learned-Circular-GCNN-RFF.

Affine: Base-RFF, Affine-MovingChart-Wavelet-energy-prob, Align-wavelet-max-affine, Align-xcorr-affine, Learned-Affine-GCNN-RFF.
