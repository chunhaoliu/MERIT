"""Fixed KBTP runway frame, following official TrajAir airport geometry.

Source: https://github.com/castacks/trajairnet/blob/main/adsb_preprocess/utils.py
GeographicLib 2.1 is an explicit diagnostic dependency; distances are meters.
"""
import numpy as np
from geographiclib.geodesic import Geodesic

RECEIVER = (40.777888, -79.949864)
RUNWAY_08 = (40.774548, -79.959237)
RUNWAY_26 = (40.778630, -79.942803)
# Second site (TartanAviation KAGC): receiver fitted from logged Range/Bearing; runway 10/28 ends from OurAirports.
SITES = {"kbtp": (RECEIVER, RUNWAY_08, RUNWAY_26),
         "kagc": ((40.351405, -79.924081), (40.354301, -79.941101), (40.354301, -79.917801))}


def use_site(name):
    """Switch receiver and runway frame (origin at the low-numbered threshold, x along the main runway)."""
    global RECEIVER, RUNWAY_08, RUNWAY_26
    RECEIVER, RUNWAY_08, RUNWAY_26 = SITES[name]


def geodetic_polar(lat, lon):
    g = Geodesic.WGS84.Inverse(*RECEIVER, float(lat), float(lon))
    return g["s12"], g["azi1"]


def polar_to_en(range_m, bearing_deg):
    angle = np.deg2rad(bearing_deg)
    return np.stack((range_m * np.sin(angle), range_m * np.cos(angle)), axis=-1)


def runway_geometry():
    origin = polar_to_en(*geodetic_polar(*RUNWAY_08))
    far = polar_to_en(*geodetic_polar(*RUNWAY_26))
    along = (far - origin) / np.linalg.norm(far - origin)
    left = np.array([-along[1], along[0]])
    return origin, np.stack((along, left))


def runway_positions(range_km, bearing_deg):
    origin, axes = runway_geometry()
    return (polar_to_en(np.asarray(range_km) * 1000., bearing_deg) - origin) @ axes.T
