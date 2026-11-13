# Protocol and configurations

*MERIT: Characterizing and Replaying the ADS-B Observation Process for Robust Aircraft Trajectory Prediction*,
submitted to IEEE Transactions on Instrumentation and Measurement.

## Data and references

| Item | Setting |
|---|---|
| Interface | [B,60,6] to [B,30,6], 1 Hz, horizontal position/velocity/acceleration |
| Processed training | official TrajAir train/test scenes, validation from 10% of shuffled training scenes, seed 42 |
| Processed trajectories | split each aircraft at missing frames, retain at least 90 points, km to m, central-gradient cached kinematics |
| Windows | 90 points, stride 30; at most 30,000 train and 6,000 validation windows, selected with seeds 43/44 |
| Actual processed windows | 30,000 train, 5,208 validation; scaler fitted to all valid training-trajectory points |
| KBTP chronological split | 111 source days, 77 train / 11 validation / 23 test |
| KAGC chronological split | 185 millisecond-timestamp days, 2021-10-31 to 2022-06-05, 128 / 19 / 38 days |
| Raw inputs | last logged position available at each integer second, same-time ties use last row, maximum logged-report age 30 s |
| Source eligibility | range below 5 km, altitude below 6,000 ft, segment gaps at most 30 s; reversed aircraft-days excluded |
| Raw reference | offline bounded linear interpolation of future ADS-B reports, exact times accepted, no extrapolation |
| Reference brackets / stride | KBTP at most 5 s / 30 s; KAGC at most 15 s / 10 s |
| Overlap | same aircraft and distance below 5 m to any official processed training point, over history and future |
| Subsets | strict_unseen overlap <10%, unseen overlap <=50%, all windows |
| KBTP raw validation / test | 7,824 / 13,247 windows; strict-unseen test 949 windows over 20 days |
| KAGC test | 504 windows from 47 aircraft over 33 days, all subsets coincide |
| Transfer | frozen KBTP weights and normalization at KAGC |

The KAGC converter strips null bytes and incomplete final rows, reformats timestamps, derives range/bearing
from latitude/longitude about the receiver, and retains days with at least 90% millisecond timestamps.
Coordinates use each airport's runway frame. Position age tracks the most recent *changed* position.
Bounded future interpolation remains an observation reference with unresolved measurement uncertainty.
Strict-unseen applies to trajectory overlap. Aircraft identity may recur. The exploratory KBTP test exposure
preceded the final replay design. Subsequent selections used validation, and later comparisons were fixed
before their test evaluations. KAGC test data entered no design choice.

## Error measurement, Section III and Table I

First appearances of changed latitude/longitude, altitude above 1,500 ft, gaps <=10 s, segments with >=40
updates. Local quadratic fits use half widths 5/10/15 s, at least seven observations, and coverage extending
at least 0.6 half widths on both sides. Fitted speed must exceed 20 m/s. Along/cross residuals follow the fitted
velocity. MAD is scaled by 1.4826. Autocorrelation uses lag tolerance +/-0.5 s and pairs with |residual| <50 m.
The primary 10 s estimator covers 12,254 segments and 2,173,755 updates. Table I CIs cluster by source day,
with 1,000 resamples and seed 0, covering 74 residual days and 65 replay days with usable observations.

The pool retains paired along/cross residuals from the first 20 training days in directory-name order after
the chronological training-period selection, half width 10 s and both components below 500 m in magnitude.
It contains 578,438 pairs. The KBTP/KAGC age banks retain all 32,249/863 training windows and 60 ages per window.
They preserve original dtype and values. The pool is stored unchanged, and age-only NPZs use lossless compression.
Synthetic calibration uses 300 seed-0 tracks with sigma/rho pairs (0,0), (5,0), (5,0.8), (25,0), (25,0.8).
The recovery check injects paired measured errors into 300 known tracks and reapplies the same estimator.
For KAGC interval statistics use `--gap 30 --min-updates 10 --halves 30 40 60 --pool-half 40`.

## Replay, Section IV and Eq. (1)

Each training batch is clean with probability 0.35. Otherwise sigma is drawn uniformly in [0,100], giving
k = sigma/100 times kmax. MERIT uses kmax=2, a sampled age row, fractional interpolation at
tau=max(1,t-age) in one-based indexing, and paired along/cross residuals. Repeated source times retain the
same residual draw. Heading uses the clean velocity at the floor of the source time, with an x-axis fallback
below 1e-3 m/s. Kinematics are rebuilt by backward differences inside the history. At the left boundary,
source times clamp to the first point while the age channel retains the sampled age in seconds.

## Training and predictor settings, Section V-A

Adam, batch 256, cosine decay, patience 5, FP32, TF32 disabled, seeds 42/123/456/0/1. Checkpoints minimize
the mean validation forecast-state loss across replay levels sigma=0/25/50/100. MERIT applies its observation
model at these same levels, yielding k=0/0.5/1/2. The recurrent learned gate additionally receives position
reconstruction MSE and a SmoothL1 analytic blend-target loss during training. Forecast loss weights position
channels by 2 and adds 0.5 times endpoint position MSE. TrajAirNet training uses its own RMSE plus latent-summed
KL objective, while checkpoint selection uses the shared validation forecast loss.

| Predictor | Configuration | Learning rate / epoch budget |
|---|---|---|
| LSTM (`direct`) | two layers, hidden 256, dropout 0.12, direct 30x6 decoder, last-position residual | 0.002 / 25 |
| LSTM+Fixed (`projection`) | same predictor, second-difference projection lambda=25, correction cap 0.75 | 0.002 / 25 |
| LSTM+NARF (`gate_analytic`) | same projection and cap, sequence-scalar reliability MLP 12-32-32-1, GELU, dropout 0.12, sharpness 2 | 0.002 / 25 |
| PatchTST (`patchtst`) | channel-independent, patch 16, stride 8, 3 layers, d=128, 16 heads, FF=256, dropout/fc dropout 0.05, BatchNorm, RevIN without affine | 0.001 / 25 |
| Transformer (`transformer`) | step tokens, d=128, 4 heads, 3 pre-norm layers, FF=256, dropout 0.1, learned positions, last/mean pooling, endpoint residual | 0.002 / 120 |
| TrajAirNet (`trajairnet`) | TCN [256,256,30], kernel 4, dilations 1/2/4, dropout 0.05, singleton GAT 16x256, CVAE two 128-wide layers, z=128, coordinate MLP=32, prior-mean inference | 0.001 / 50 |

Some selected checkpoints reach the epoch budget. The fixed-budget results retain this stopping behavior.

## Training and input controls, Tables III-V and text

Each row below uses `python scripts/train.py --data data --row direct --seed <seed> --output <fresh_run>`
plus the listed arguments. The index records each released checkpoint's exact settings.

| Setting / checkpoint group | Arguments |
|---|---|
| Conventional isotropic (`iso`) | `--noise iso` |
| MERIT (`merit`) | `--noise merit --kmax 2` |
| Matched isotropic (`matched`) | `--noise measured --pool gaussian:12.35 --kmax 2` |
| Matched isotropic plus measured staleness (`matched_stale`) | `--noise merit --pool gaussian:12.35 --kmax 2` |
| Measured structure (`structure`) | `--noise measured --kmax 2` |
| Scale variants (`merit_k1`, `merit_k4`, `merit_k10`) | `--noise merit --kmax 1`, `4` or `10` |
| Raw-data training (`raw`) | `--data-mode raw --noise none` |
| Raw fine-tuning (`raw_finetune`) | `--data-mode raw --noise none --init <MERIT_checkpoint>` |
| Tuned heuristic (`heur`) | `--noise merit --pool gaussian:0.01 --ages bernoulli:0.25 --kmax 2` |
| KAGC age sensitivity (`merit_kagc`) | `--noise merit --ages merit/ages_kagc.npz --kmax 2` |
| Isotropic plus KF (`ftp_iso`) | `--noise iso --filter-cv 8 15` |
| MERIT plus KF (`ftp_merit`) | `--noise merit --filter-cv 8 15` |
| MERIT plus age (`merit_age`) | `--noise merit --report-age` |
| Matched residuals, measured ages, age input (`iso_stale_age`) | `--noise merit --pool gaussian:12.35 --report-age` |
| Near-zero residuals plus measured ages (`near0`) | `--noise merit --pool gaussian:0.01 --kmax 2` |
| Date-disjoint isotropic / MERIT (`dd_iso`, `dd_merit`) | `--noise iso --date-disjoint` / `--noise merit --kmax 2 --date-disjoint` |

Raw-data training uses 28,644 input/reference windows and 3,605 selection windows from the final eight
eligible training-period days. It retains the processed-training scaler. The factorial and scale/fine-tuning
ablations use seeds 42/123/456. Selected main/raw/heuristic/age-bank/input comparisons use all five seeds,
as do the `near0`, `matched_stale` and date-disjoint groups (`results/p8_*`).

Table V has seven residual/timing rows and five alternative remedies. Its no-timing matched-isotropic KBTP
cells use the five-seed `gauss direct` results in `raw_eval_test_final`; its KAGC cells use the three-seed
`iso_matched direct` results in `kagc_eval_test_2x2`. Measured residuals without timing use the three-seed
`measured direct` results in `raw_eval_test_2x2` / `kagc_eval_test_2x2`. These cells carry a dagger.
The paired effect of adding measured ages to matched isotropic residuals uses the same three seeds on
both sides (`iso_matched_stale direct` versus `iso_matched direct`), separately from the five-seed table rows.
Validation ablations, scale bounds, raw fine-tuning and the KAGC-age sensitivity remain text controls.

Date-disjoint retraining keeps the official training scenes whose tracks match same-aircraft raw reports on a
unique day inside the 77 measurement days: 707 training and 83 selection scenes, 10,699 and 1,236 windows,
listed by cache trajectory index in `merit/date_disjoint.json`. The scaler is refitted on these training
trajectories. Raw validation and test windows, checkpoints selection and all other settings are unchanged.

The 13 heuristic candidates are (s,p) = (0.01,0.25), (0.01,0.5), (3,0.25), (3,0.5), (6,0.1), (6,0.25),
(6,0.5), (12.35,0.1), (12.35,0.25), (12.35,0.5), (25,0.1), (25,0.25), (25,0.5), selected using raw validation.
The selected nominal s=0.01 m is effectively zero. Bernoulli ages use 20,000 rows, seed 0, with a fresh first step.

## Kinematic, filtering and statistical evaluation

CV uses the final 5 s displacement. KF-CV has dt=1 s, white acceleration process SD q, position SD r,
zero initial velocity and a 100 m/s velocity prior. Validation-selected (q,r) are (4,15) at KBTP and (1,60)
at KAGC. Filter-then-predict chooses (8,15) on KBTP from q in {1,2,4,8}, r in {15,30,60}, and transfers
that setting unchanged. The front end is a causal linear operator built in float64 and applied at input dtype.
Age input adds the seventh channel, preserving six-channel targets and giving zero ages to clean batches.

ADE/FDE are horizontal Euclidean distances in meters. Means and sample SDs are over seeds. P95 first averages
each window's errors across seeds, then takes the 95th percentile of window ADE. Relative-change CIs use
2,000 day-cluster resamples, seed 0. P95 relative-change CIs resample days in the same way (`compare_p95`). Wilcoxon uses paired, seed-averaged window errors, without multiplicity
adjustment. A stored p=0 indicates floating-point underflow. The measurement CIs use the separate 1,000-draw
procedure above. Reference-smoothing sensitivity uses Savitzky-Golay width 7, order 2, on future references only.

Tables III and IV compare each MERIT predictor with its own isotropic counterpart. Saved comparisons in
the main reports target the direct LSTM, so the other predictors' paired CIs are recomputed from their
window errors. Table II counts trainable parameters, in millions, at the manuscript's displayed precision.
Figure 3(c) uses 1,000 day-cluster draws with `RandomState(0)`, shared by the two predictors within each bin
and advanced across bins; quantile bins have inclusive bounds. Figure 4 retains the seed-42 case selection
and uses the bundled 3x60 displayed-position ages and 3x30 prediction/reference positions; source window
indices, timestamps, dates and aircraft identifiers are omitted.
