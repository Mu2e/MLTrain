# CrvCosmic

XGBoost-based classifier for rejecting cosmic-ray-induced backgrounds at the Mu2e experiment. The model takes per-coincidence CRV and tracker features as input and outputs a probability that an event is cosmic-induced. It is trained on `CosmicCRYSignalAllOnSpillTriggered` (pure cosmics) versus `CeEndpointMix2BBTriggered` (beam+pileup) datasets.

This code was ported from [`sam-grant/mu2e-cosmic`](https://github.com/sam-grant/mu2e-cosmic).

## Layout

```
CrvCosmic/
├── config/
│   └── cuts.yaml              # Cutset definition
├── src/
│   ├── core/
│   │   ├── analyse.py         # Cut flow + feature helpers
│   │   └── postprocess.py     # combine_cut_flows / combine_hists / combine_arrays
│   ├── utils/
│   │   ├── io_manager.py      # pkl + cuts.yaml load, pkl/csv/h5/parquet write
│   │   ├── hist_manager.py    # histogram booking/filling for the summary plots
│   │   ├── draw.py            # plot_summary_ml
│   │   └── mu2e.mplstyle
│   └── ml/
│       ├── process.py         # MLProcessor: extracts features from ROOT files
│       ├── load.py            # LoadML: read processed data and trained models
│       ├── assemble.py        # AssembleDataset: label, shuffle, fold-split
│       ├── train.py           # Train: fit XGBoost, K-fold CV, save UBJ
│       ├── validate.py        # Validate: ROC, threshold scan, money table
│       └── optimise.py        # Optimise: grid search over hyperparameters
├── run/
│   └── run_ml_prep.py         # entrypoint: process ROOT files into parquet/pkl
└── notebooks/
    ├── assemble.ipynb         # Validate assemble module
    ├── feature_engineering.ipynb # Feature evaluation
    ├── optimise.ipynb         # Hyperparameter tuning
    ├── train.ipynb            # Training
    └── validate.ipynb         # Validation
```

The non-ML pieces of the upstream cosmic-background framework are not included here — only the cuts and helpers that the ML preselection actually exercises.

## Pipeline

1. **Process**: read ROOT EventNtuple files, apply the `MLPreprocess` cutset,
   flatten per-coincidence features into awkward/parquet output.
2. **Assemble**: load processed CRY and CE-mix outputs, label, combine, and
   produce K-fold indices for nested cross-validation.
3. **Train**: fit XGBoost on each fold, find the per-fold operating threshold at
   a target veto efficiency, then retrain on the full set with the
   CV-averaged threshold and export to `.ubj`.
4. **Validate**: compute ROC AUC, score distributions, feature importance, and
   a "money table" comparing the ML model against the simple `dT` cut baseline.
5. **Optimise** (optional): grid search over XGBoost hyperparameters using
   K-fold CV; minimises deadtime at the target veto efficiency.

## Setup

On a Mu2e VM or EAF:

```bash
mu2einit
pyenv ana
```

The code depends on [`Mu2e/pyutils`](https://github.com/Mu2e/pyutils) (provided by `pyenv ana`) — `pyutils.pycut.CutManager` is used directly — plus `xgboost` `scikit-learn`, `awkward`, `pandas`, `hist`, `pyarrow`, `h5py`, `joblib`, and `pyyaml`.

## Running the pipeline

### 1. Preprocess ROOT files into ML-ready features

```bash
cd run
python run_ml_prep.py            # production: full datasets via xrootd
python run_ml_prep.py --test     # test: two local files, if they exist
```

This populates `output/ml/{run}/data/{tag}/` with `events.parquet`, `results.pkl`, `hists.h5`, `stats.csv`, and `cut_flow.csv`. The default production `run_str` is set inside `run_ml_prep.py` (currently `"k"`); change it when starting a new training round.

### 2. Train and validate

```bash
cd notebooks
jupyter notebook
```


The final model is saved to `output/ml/{run}/results/final/model_{version}.ubj`, suitable for inference with the standard XGBoost C API.

## Configuring cuts

Cutsets are defined in `config/cuts.yaml`. The ML preprocessing uses the `MLPreprocess` cutset by default (set in `src/ml/process.py`). To modify the preselection, edit thresholds or active flags under `cutsets.MLPreprocess`. Defining a new cutset is also possible by adding an entry under `cutsets:` and overriding `cutset_name` when constructing `MLProcessor`.

## Output paths

All outputs land under `CrvCosmic/output/`:

- `output/ml/{run}/data/{tag}/` &mdash; processed per-coincidence features
- `output/ml/{run}/results/{tag}/` &mdash; trained models, threshold scans,
  money tables, CSV summaries
- `output/images/ml/{run}/` &mdash; ROC curves, score distributions, feature
  importance plots

These directories are created on demand. Symlink them to another disk if space is a concern.
