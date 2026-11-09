"""Train one paper predictor using processed or training-period raw windows."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit.evaluation import MODELS, make_model
from merit.losses import forecast_loss
from merit.training import ObservationReplay, load_data, validate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--row", choices=MODELS, default="direct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", choices=["iso", "measured", "merit", "none"], default="merit")
    p.add_argument("--kmax", type=float, default=2.0)
    p.add_argument("--pool", default="measured", help="measured or gaussian:STD")
    p.add_argument("--ages", help="NPZ age bank or bernoulli:P")
    p.add_argument("--data-mode", choices=["processed", "raw"], default="processed")
    p.add_argument("--init", type=Path, help="checkpoint for raw-data fine-tuning")
    p.add_argument("--filter-cv", nargs=2, type=float)
    p.add_argument("--report-age", action="store_true")
    p.add_argument("--budget", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.report_age and a.noise != "merit":
        p.error("Age input requires held-report replay (--noise merit)")
    if (a.filter_cv or a.report_age) and (a.row != "direct" or a.data_mode != "processed" or a.init):
        p.error("Input processing requires a processed-data direct LSTM without initialization")
    if a.output.resolve().is_relative_to(a.data.resolve()):
        raise ValueError("Training output must be outside the source data directory")
    if not math.isfinite(a.kmax) or a.kmax < 0:
        p.error("--kmax must be finite and nonnegative")
    budget = 1 if a.smoke else a.budget or {"transformer": 120, "trajairnet": 50}.get(a.row, 25)
    lr = a.lr or (.001 if a.row in ("patchtst", "trajairnet") else .002)
    if budget <= 0 or not math.isfinite(lr) or lr <= 0:
        p.error("Training budget and learning rate must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors, scaler = load_data(a.data, a.device, raw=a.data_mode == "raw", smoke=a.smoke)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    model = make_model(a.row, scaler.mean_, scaler.scale_, filter_cv=a.filter_cv, report_age=a.report_age).to(a.device)
    if a.init:
        model.load_state_dict(torch.load(a.init, weights_only=True, map_location=a.device))
    noise = ObservationReplay(a.noise, kmax=a.kmax, pool=a.pool, ages=a.ages, report_age=a.report_age)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    train, val = tensors
    a.output.mkdir(parents=True, exist_ok=False)
    config = {k: v for k, v in vars(a).items() if k not in ("data", "output", "init", "ages")}
    config.update(budget=budget, lr=lr, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                  ages="external" if a.ages else "released KBTP", initialization=bool(a.init))
    (a.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, best_epoch, bad, history = math.inf, 0, 0, []
    for epoch in range(budget):
        model.train()
        rng = np.random.RandomState(a.seed + epoch * 10000)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(a.seed + epoch * 10000)).to(train.device)
        for batch_id, offset in enumerate(range(0, len(train), 256)):
            batch = train[order[offset:offset + 256]]
            clean, target = batch[:, :60], batch[:, 60:]
            sigma = 0. if rng.rand() < .35 else float(rng.uniform(0, 100))
            observed = noise(clean, scaler, sigma, a.seed + epoch * 100000 + batch_id)
            optimizer.zero_grad(set_to_none=True)
            if a.row == "trajairnet":
                loss = model.training_loss(observed, target)
            else:
                prediction, auxiliary = model(observed, clean)
                loss = forecast_loss(prediction, target) + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
        score, metrics = validate(model, val, scaler, noise)
        history.append(dict(epoch=epoch + 1, validation_loss=score, validation=metrics))
        if score < best:
            best, best_epoch, bad = score, epoch + 1, 0
            torch.save(model.state_dict(), a.output / "checkpoint.pth")
        else:
            bad += 1
        (a.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch + 1} validation loss {score:.6f} best epoch {best_epoch}", flush=True)
        if bad >= 5:
            break
        for group in optimizer.param_groups:
            group["lr"] = lr * .5 * (1 + math.cos(math.pi * (epoch + 1) / budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
