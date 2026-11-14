# MERIT: Characterizing and Replaying the ADS-B Observation Process for Robust Aircraft Trajectory Prediction

Code, checkpoints and evaluation outputs for

> C. Liu, P. Wu, S. He, and J. H. Park, *MERIT: Characterizing and Replaying the ADS-B Observation Process for Robust Aircraft Trajectory Prediction*, submitted to IEEE Transactions on Instrumentation and Measurement.

This core reproduction contains processed-data construction, measurement of ADS-B residuals and report ages,
the measured-error replay generator, ten predictors and their controls, training and raw-replay evaluation,
the measured residual pool and age banks, and the per-seed results used in the paper. On KBTP strict-unseen
windows, MERIT reduces LSTM ADE from 136.5 to 120.6 m. At KAGC, the same weights give 253.6 to 210.9 m.
Baseline configurations and the evaluation protocol are in `CONFIGS.md`.

## Contents

| Path | What |
|---|---|
| `merit/data.py`, `merit/windows.py` | official processed-scene construction, normalization and chronological raw windows with overlap labels (Sections III-A and V-A) |
| `merit/data_manifest.json` | SHA-256 for the processed cache, normalized windows, six raw-replay caches and derived statistics |
| `merit/causal.py`, `merit/raw.py`, `merit/kagc.py` | past-only replay, position age, bounded-interpolation references and second-site preparation |
| `merit/measurement.py`, `merit/synthetic.py` | local quadratic residuals, along/cross decomposition, estimator calibration and recovery, day-cluster measurement CIs (Section III) |
| `merit/replay.py`, `merit/training.py` | Eq. (1) replay, clean batches, isotropic/matched Gaussian and Bernoulli controls (Section IV) |
| `merit/lstm.py`, `merit/inputs.py` | LSTM, Fixed, NARF, filter-then-predict and age-input controls |
| `merit/patchtst.py`, `merit/transformer.py`, `merit/trajairnet.py` | attention and single-aircraft generative predictors |
| `merit/residual_pool.npz`, `merit/ages_*.npz` | 578,438 paired residuals and complete KBTP/KAGC training-period age banks |
| `scripts/prepare_data.py` | rebuilds from public sources and checks container and array hashes |
| `scripts/measure_error.py` | regenerates measurement summaries, the pool, recovery checks and Table I CIs |
| `scripts/train.py` | paper training settings for all ten predictors and the training/input controls |
| `scripts/evaluate.py` | released-checkpoint evaluation, CV/KF-CV, per-seed metric checks, day bootstrap and Wilcoxon |
| `scripts/reproduce_tables.py` | source-data-free reproduction of 430 printed cells: Tables I-V, paired CIs, and the validation, date-disjoint, timing, tail-error and figure results quoted in the text |
| `scripts/draw_figures.py` | redraws Figures 1, 3 and 4 from bundled derived statistics, errors and recentered case data |
| `checkpoints/` | 175 checkpoints, 457.8 MB, with settings, SHA-256 and reference metrics in `index.json` |
| `results/` | full-precision per-seed raw/processed metrics, validation ablations, test factorials, input processing, smoothing sensitivity and window-error arrays |
| `results/runs.json` | 1,551 raw-evaluation records with explicit study, site, split, seed and subset labels |
| `results/paper_tables.json` | numerical cells as printed in the manuscript |
| `results/figure_data.json`, `results/figure_cases.*` | age bins, figure definitions and de-identified cases with origin-relative tracks and 60-step displayed-position ages |
| `tests/` | model, replay, causality, data, hash, statistical and table checks |

## Usage

Validated with Python 3.11, PyTorch 2.7.1 and CUDA 12.8 on an RTX 5050.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
python scripts/reproduce_tables.py
python scripts/reproduce_tables.py --bootstrap
python scripts/draw_figures.py --output outputs/figures
```

`reproduce_tables.py` exits with status 0 when every printed cell agrees. The optional bootstrap regenerates
all paired prediction CIs from seed-averaged window errors, including intervals otherwise read from saved reports. Table I uses the frozen measurement summaries
and measurement bootstrap outputs. Raw-log measurement, including its day bootstrap, has a separate entry point.
Table II checks trainable parameter counts. Tables III and IV pair isotropic and MERIT training for ten predictors,
with a separate relative-change CI for each predictor and subset; Table IV also reports P95. Table V contains
seven residual/timing combinations and five alternative remedies, with three-seed cells marked by a dagger.
All printed values use direct one-step rounding of the full-precision results. The JSON results retain full precision.

### Preparing source data

Obtain TrajAir's `111_days` directory, containing `processed_data/train`, `processed_data/test` and
`raw_data`, from the official release. Obtain the KAGC ADS-B date directories from TartanAviation.
These source files are user-supplied inputs. [TrajAir](https://theairlab.org/trajair/), [TartanAviation](https://github.com/castacks/TartanAviation).

```bash
python scripts/prepare_data.py --trajair <111_days> --kagc-extract <KAGC_date_folders> --output data
# Alternatively supply KAGC logs already converted by merit.kagc.convert
python scripts/prepare_data.py --trajair <111_days> --kagc-raw <KAGC_raw_data> --output data
python scripts/prepare_data.py --verify-only data
```

Choose one build command and a fresh output directory. The builder converts eligible KAGC days, constructs
the processed cache and raw windows, and verifies the published hashes. A full local reconstruction matched
the processed cache bytes, the 30,000/5,208 normalized training/validation windows, and every container and array
in all six KBTP/KAGC replay caches exactly. KAGC reconstruction in this check started from already converted
logs. The original extracted KAGC inputs were unavailable for a full conversion replay.

### Evaluating the released models

```bash
python scripts/evaluate.py --data data --site kbtp --groups iso merit --rows direct --subsets strict_unseen --output outputs/kbtp
python scripts/evaluate.py --data data --site kbtp --groups iso merit --rows direct projection gate_analytic patchtst transformer --kf 4 15 --output outputs/kbtp_all
python scripts/evaluate.py --data data --site kagc --groups iso merit --rows direct projection gate_analytic patchtst transformer --kf 1 60 --output outputs/kagc
python scripts/evaluate.py --data data --site kbtp --groups ftp_iso ftp_merit merit_age iso_stale_age --output outputs/input_controls
python scripts/evaluate.py --data data --site kbtp --groups merit near0 matched_stale --output outputs/timing_controls
python scripts/evaluate.py --data data --site kbtp --groups dd_iso dd_merit --target "dd_merit direct" --output outputs/date_disjoint
```

Evaluation reads input data and writes to the selected output directory. It verifies each checkpoint and replay
cache before inference. `--device` selects CPU or CUDA. Re-evaluation of all 50 main released checkpoints on
both sites reproduced 900 per-seed/subset metrics with a maximum relative difference of 0. The 20 input-control
checkpoints reproduced another 360 metrics across both sites exactly. These checks include the five-seed
LSTM isotropic/MERIT strict-unseen comparison. Retraining can vary with the numerical environment.

### Measurement and training

```bash
python scripts/measure_error.py --raw <KBTP_raw_data> --output outputs/measurement
python scripts/measure_error.py --mode recovery --output outputs/recovery
python scripts/measure_error.py --mode day-ci --raw <KBTP_raw_data> --train-replay data/raw_replay/train.npz --output outputs/measurement_ci
python scripts/train.py --data data --row direct --noise merit --seed 42 --output runs/merit_direct_s42
python scripts/train.py --data data --row direct --noise iso --seed 42 --output runs/iso_direct_s42
python scripts/train.py --data data --row trajairnet --noise merit --seed 42 --output runs/merit_trajairnet_s42
python scripts/train.py --data data --row direct --noise merit --date-disjoint --seed 42 --output runs/dd_merit_direct_s42
```

The residual-pool reconstruction matched the released NPZ byte for byte. Recovery can regenerate its known
synthetic tracks locally, or read `--synthetic-cache`. `CONFIGS.md` gives the remaining control settings.
Run each training setting for seeds 42, 123, 456, 0 and 1, using the first three for the factorial/validation ablations.

## Checkpoint coverage

The 175 weights comprise 50 main checkpoints for LSTM, LSTM+Fixed, LSTM+NARF, PatchTST and Transformer,
20 input-processing checkpoints, 45 training/scale/age-bank controls, and 10 date-disjoint LSTM checkpoints. Main and selected training controls
have five seeds, while factorial, scale and fine-tuning controls have three. Files are unchanged from the
experiment artifacts and individually smaller than 100 MB. No large-file extension is required.

The ten TrajAirNet checkpoints, totaling 701.9 MB, are excluded to keep this core repository compact.
Its complete five-seed results at both sites, window errors, model implementation and training recipe are
included. Retrain those weights with `scripts/train.py` for inference reproduction of that row.
The other controls are selected by their `group`, `row` and `seed` in `checkpoints/index.json`.

The nine PDF panels reproduce Figures 1, 3 and 4. Figure 1 shows KBTP residual structure and contrasts
displayed-position ages at the two sites. Figure 3 shows KBTP horizon errors, the KAGC window-error tail and age-dependent gains
with 95% day-cluster bootstrap bars. Figure 4 shows each case's 60 s displayed-position age history and
30-step displacement errors. Drawing uses only bundled statistics, window errors and de-identified cases.

## Third-party implementation

TrajAirNet follows the official implementation at commit `5a5bc751dabba09a668ca9acb6b05ae4112c5b3d`,
adapted to horizontal single-aircraft 60 s histories and 30 s forecasts. The adaptation removes weather and
neighbor inputs, evaluates the one-node attention algebra, uses prior mean z=0, rebuilds six-channel kinematics,
and initializes Verlet velocity from the last two 1 Hz observations. Dilated convolutions use equivalent
unfold/linear operations. This product includes software developed by [project]. Its BSD-4-Clause copyright
and license are retained in `merit/trajairnet.py` and `THIRD_PARTY_NOTICES`. [TrajAirNet source](https://github.com/castacks/trajairnet).

PatchTST uses the supervised implementation at commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`,
with package-relative imports and task-specific input/output handling outside the vendored layers.
The Apache-2.0 license and original RevIN attribution are retained under `merit/vendor_patchtst/`.
[PatchTST source](https://github.com/yuqinie98/PatchTST).

## License

MIT, copyright 2026 Chunhao Liu, for the MERIT code. Third-party terms are listed in `THIRD_PARTY_NOTICES`.

### Added predictor implementations

`merit/modern_models.py` contains the Seq2Seq LSTM adapter (`seq2seq`), the faithful WTFTP wavelet path (`wtftp`), iTransformer (`itransformer`), DLinear (`dlinear`), and PatchTST with the last-observation position residual (`patchtst_r`). WTFTP uses a 60-step history followed by 29 zero placeholders to reconstruct 90 steps and returns the last 30 steps; positions are represented relative to the last observed position. Its Haar DWT, attention blocks, coefficient-domain loss, and inverse reconstruction are implemented locally without an extra wavelet dependency.

The added weights are the five-seed locked artifacts for conventional and MERIT training. `checkpoints/index.json` records each SHA-256, configuration, scaler and per-site metrics.
