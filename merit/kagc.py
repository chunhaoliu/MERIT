"""Convert millisecond-resolution TartanAviation KAGC logs to the replay layout."""
import io
from pathlib import Path
import numpy as np
import pandas as pd

R_EARTH = 6371008.8
RECEIVER = (40.351405, -79.924081)  # same as raw_coordinates.SITES["kagc"]


def convert(src, dst):
    text = Path(src).read_bytes().replace(b"\0", b"")
    text = text[:text.rfind(b"\n") + 1]  # some logs end in a truncated row
    try:
        frame = pd.read_csv(io.BytesIO(text), usecols=["ID", "Time", "Date", "Altitude", "Lat", "Lon"], dtype=str,
                            on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return "skipped: empty"
    t = frame.Time.str.findall(r"[\d.]+")
    d = frame.Date.str.findall(r"\d+")
    ok = (t.str.len() == 3) & (d.str.len() == 3)
    frame, t, d = frame[ok].copy(), t[ok], d[ok]
    millisecond = np.mean(["." in x[2] for x in t]) if len(t) else 0.
    if millisecond < .9:
        return f"skipped: millisecond fraction {millisecond:.2f}"
    frame["Time"] = [f"{int(h):02d}:{int(m):02d}:{float(s):06.3f}" for h, m, s in t]
    frame["Date"] = [f"{int(mo):02d}/{int(dd):02d}/{int(y):04d}" for y, mo, dd in d]
    for k in ("Altitude", "Lat", "Lon"):
        frame[k] = pd.to_numeric(frame[k], errors="coerce")
    frame = frame.dropna()
    lat0, lon0 = np.radians(RECEIVER)
    x = R_EARTH * (np.radians(frame.Lon.to_numpy()) - lon0) * np.cos(lat0)
    y = R_EARTH * (np.radians(frame.Lat.to_numpy()) - lat0)
    frame["Range"] = np.hypot(x, y) / 1000.0
    frame["Bearing"] = np.degrees(np.arctan2(x, y))
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame[["ID", "Time", "Date", "Altitude", "Lat", "Lon", "Range", "Bearing"]].to_csv(dst, index=False)
    return len(frame)
