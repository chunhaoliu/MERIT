"""Frozen displacement summaries and day-cluster paired comparisons."""
import numpy as np

def kf_cv_forecast(history, process_std, measurement_std, horizon=30):
    """Constant-velocity Kalman filter (dt = 1 s) on the replayed history, then CV extrapolation."""
    f = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    h = np.eye(2, 4)
    g = np.array([[.5, 0], [0, .5], [1, 0], [0, 1]])
    q, r = g @ g.T * process_std ** 2, np.eye(2) * measurement_std ** 2
    p = np.diag([measurement_std ** 2] * 2 + [100. ** 2] * 2)
    state = np.concatenate([history[:, 0], np.zeros_like(history[:, 0])], axis=1)
    for t in range(1, history.shape[1]):  # covariance is shared: same sample times for all windows
        state, p = state @ f.T, f @ p @ f.T + q
        gain = p @ h.T @ np.linalg.inv(h @ p @ h.T + r)
        state = state + (history[:, t] - state @ h.T) @ gain.T
        p = (np.eye(4) - gain @ h) @ p
    steps = np.arange(1, horizon + 1)[None, :, None]
    return state[:, None, :2] + state[:, None, 2:] * steps


def summarize(errors, day, mask):
    days = np.unique(day[mask])
    return dict(ADE=float(errors[mask].mean()), FDE=float(errors[mask][:, -1].mean()),
                day_equal_ADE=float(np.mean([errors[mask & (day == d)].mean() for d in days])), windows=int(mask.sum()))


def compare(target, other, day, mask, draws=2000, seed=0):
    """Relative ADE change of target vs other: day-cluster bootstrap 95% CI and window-level Wilcoxon p."""
    from scipy.stats import wilcoxon
    a, b = target[mask].mean(1), other[mask].mean(1)
    days = np.unique(day[mask])
    groups = [np.flatnonzero(day[mask] == d) for d in days]
    rng = np.random.RandomState(seed)
    ratios = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.randint(0, len(groups), len(groups))])
        ratios.append(a[idx].mean() / b[idx].mean() - 1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return dict(change=float(a.mean() / b.mean() - 1), ci95=[float(lo), float(hi)],
                wilcoxon_p=float(wilcoxon(a, b).pvalue), days_better=int(sum(a[g].mean() < b[g].mean() for g in groups)),
                days=len(groups))
