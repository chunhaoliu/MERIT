import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from merit.causal import causal_replay
from merit.evaluation import MODELS, make_model
from merit.inputs import cv_filter_history
from merit.kagc import convert
from merit.losses import forecast_loss
from merit.raw import bounded_reference, build_frame
from merit.replay import inject_adsb_noise
from merit.statistics import compare, kf_cv_forecast
from merit.training import ObservationReplay, bernoulli_ages
from merit.windows import RECIPE
from scripts.reproduce_tables import recompute

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class Models(unittest.TestCase):
    def test_all_predictors_forward_gradient_and_mode(self):
        for row in MODELS:
            with self.subTest(row=row):
                torch.manual_seed(0)
                model = make_model(row, np.zeros(6), np.ones(6))
                x = torch.randn(2, 60, 6)
                model.train()
                prediction, auxiliary = model(x, x)
                self.assertEqual(prediction.shape, (2, 30, 6))
                self.assertEqual(prediction.dtype, x.dtype)
                loss = (model.training_loss(x, torch.zeros_like(prediction)) if row == "trajairnet"
                        else forecast_loss(prediction, torch.zeros_like(prediction)) + auxiliary)
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertTrue(any(p.grad is not None for p in model.parameters() if p.requires_grad))
                model.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(x)[0], model(x)[0]))

    def test_checkpoint_hashes_and_reload(self):
        index = json.loads((ROOT / "checkpoints/index.json").read_text())
        self.assertEqual(len(index), 125)
        seen = set()
        for entry in index:
            p = ROOT / "checkpoints" / entry["file"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), entry["sha256"])
            self.assertLess(p.stat().st_size, 100_000_000)
            settings = entry["settings"]
            kind = (entry["row"], bool(settings.get("filter_cv")), bool(settings.get("report_age")))
            if kind in seen:
                continue
            seen.add(kind)
            model = make_model(entry["row"], entry["mean"], entry["scale"],
                               filter_cv=settings.get("filter_cv"), report_age=settings.get("report_age", False))
            model.load_state_dict(torch.load(p, weights_only=True, map_location="cpu"))
            x = torch.zeros(2, 60, 7 if settings.get("report_age") else 6)
            self.assertTrue(torch.isfinite(model.eval()(x)[0]).all())

    def test_filter_and_age_lstm_shapes(self):
        for age, filt in ((True, None), (False, [8, 15])):
            model = make_model("direct", np.zeros(6), np.ones(6), report_age=age, filter_cv=filt)
            self.assertEqual(model(torch.zeros(2, 60, 7 if age else 6))[0].shape, (2, 30, 6))

    def test_parameter_counts(self):
        expected = {"direct": 845993, "trajairnet": 17541226}
        for row, count in expected.items():
            model = make_model(row, np.zeros(6), np.ones(6))
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)


class Replay(unittest.TestCase):
    def setUp(self):
        self.scaler = SimpleNamespace(mean_=np.zeros(6), scale_=np.ones(6))
        self.x = torch.zeros(2, 60, 6)
        self.x[:, :, 0] = torch.arange(60)
        self.x[:, :, 2] = 1

    def test_held_report_keeps_position_error(self):
        ages = torch.arange(60, dtype=torch.float32)[None]
        ages[:, 1:] -= 1
        result, actual_age = inject_adsb_noise(self.x, 1, self.scaler, torch.tensor([5.]), torch.tensor([2.]),
                                              staleness=ages, return_age=True,
                                              generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(result[:, 1:, :2], result[:, 1:2, :2].expand(-1, 59, -1)))
        self.assertTrue(torch.equal(actual_age, ages.expand(2, -1)))

    def test_left_boundary_and_zero_scale(self):
        result, age = inject_adsb_noise(self.x, 0, self.scaler, torch.zeros(1), torch.zeros(1),
                                        staleness=torch.full((1, 60), 100.), return_age=True)
        self.assertTrue(torch.equal(result[:, :, :2], torch.zeros(2, 60, 2)))
        self.assertTrue(torch.all(age == 100))

    def test_clean_batch_and_heuristic(self):
        noise = ObservationReplay("merit", report_age=True)
        x = noise(self.x, self.scaler, 0, 42)
        self.assertTrue(torch.equal(x[:, :, :6], self.x))
        self.assertTrue(torch.all(x[:, :, 6] == 0))
        np.testing.assert_array_equal(bernoulli_ages(0, n=2), np.zeros((2, 60)))
        np.testing.assert_array_equal(bernoulli_ages(1, n=2), np.tile(np.arange(60), (2, 1)))

    def test_replay_reproducibility_and_gradient(self):
        x = self.x.requires_grad_()
        noise = ObservationReplay("merit")
        a, b = noise(x, self.scaler, 50, 3), noise(x, self.scaler, 50, 3)
        self.assertTrue(torch.equal(a, b))
        a.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class DataAndStatistics(unittest.TestCase):
    def test_kagc_conversion_and_timestamp_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.csv", Path(folder) / "converted.csv"
            rows = pd.DataFrame({"ID": ["123", "123"], "Time": ["12:00:00.123", "12:00:01.234"],
                                 "Date": ["2022-01-01", "2022-01-01"], "Altitude": [2000, 2000],
                                 "Lat": [40.351405, 40.352405], "Lon": [-79.924081, -79.924081]})
            rows.to_csv(source, index=False)
            self.assertEqual(convert(source, target), 2)
            result = pd.read_csv(target)
            self.assertEqual(result.Date.iloc[0], "01/01/2022")
            self.assertEqual(result.Time.iloc[0], "12:00:00.123")
            self.assertAlmostEqual(result.Range.iloc[0], 0)
            self.assertGreater(result.Range.iloc[1], .1)
            rows["Time"] = ["12:00:00", "12:00:01"]
            rows.to_csv(source, index=False)
            self.assertTrue(str(convert(source, target)).startswith("skipped"))

    def test_causal_ties_and_future_independence(self):
        t = np.array([0., 1., 1., 3.])
        p = np.array([[0., 0.], [1., 1.], [2., 2.], [9., 9.]])
        r = causal_replay(t, p, [0, 1, 2])
        np.testing.assert_array_equal(r["position_m"], [[0, 0], [2, 2], [2, 2]])
        self.assertTrue(r["same_timestamp_position_conflict"][1])
        p[-1] += 10000
        np.testing.assert_array_equal(r["position_m"], causal_replay(t, p, [0, 1, 2])["position_m"])
        with self.assertRaises(ValueError):
            causal_replay([1, 0], np.zeros((2, 2)), [0])

    def test_bounded_reference(self):
        r = bounded_reference([0., 1., 10.], np.array([[0., 0.], [1., 1.], [10., 10.]]), [-1., .5, 5., 10., 11.], 5)
        np.testing.assert_array_equal(r["valid"], [False, True, False, True, False])

    def test_raw_frame_windows(self):
        t = pd.date_range("2021-01-19", periods=120, freq="s")
        f = pd.DataFrame({"ID": ["123"] * len(t), "Date": t.strftime("%m/%d/%Y"),
                          "Time": t.strftime("%H:%M:%S.%f"), "Range": np.linspace(1, 2, len(t)),
                          "Bearing": np.zeros(len(t)), "Altitude": np.full(len(t), 2000)})
        arrays, counts = build_frame(f, RECIPE)
        self.assertEqual(counts["accepted_windows"], 2)
        self.assertEqual(arrays["history_xy_m"].shape, (2, 60, 2))

    def test_filter_is_causal_and_matches_forecast_state(self):
        x = torch.randn(3, 60, 2, dtype=torch.float64)
        filtered = cv_filter_history(x, 8, 15)
        changed = x.clone()
        changed[:, 30:] += 1000
        torch.testing.assert_close(filtered[:, :30], cv_filter_history(changed, 8, 15)[:, :30], rtol=0, atol=0)
        forecast = kf_cv_forecast(x.numpy(), 8, 15)
        velocity = forecast[:, 1] - forecast[:, 0]
        np.testing.assert_allclose(forecast[:, 0] - velocity, filtered[:, -1].numpy(), rtol=1e-10, atol=1e-10)

    def test_bootstrap_is_deterministic(self):
        days = np.repeat(["d1", "d2", "d3"], 3)
        b = np.arange(1., 28.).reshape(9, 3)
        a = b * .8
        result = compare(a, b, days, np.ones(9, bool), draws=50)
        np.testing.assert_allclose(result["ci95"], [-.2, -.2])

    def test_derived_statistic_hashes(self):
        manifest = json.loads((ROOT / "merit/data_manifest.json").read_text())
        for filename, spec in manifest["derived"].items():
            p = ROOT / "merit" / filename
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), spec["sha256"])
            with np.load(p) as z:
                for key, array in spec["arrays"].items():
                    self.assertEqual(hashlib.sha256(z[key].tobytes()).hexdigest(), array["sha256"])

    def test_all_printed_cells(self):
        expected = json.loads((ROOT / "results/paper_tables.json").read_text())
        self.assertEqual(recompute(), expected)
        self.assertEqual(recompute(bootstrap=True), expected)


if __name__ == "__main__":
    unittest.main()
