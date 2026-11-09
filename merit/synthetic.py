"""Known synthetic tracks for the residual estimator recovery check."""
import os
import pickle
import numpy as np

AIRCRAFT_CACHE_VERSION = "aircraft_state_v2"

def generate_ca_trajectory(length=120, noise_level=0.1, dims=4096):
    x0, y0 = np.random.uniform(100, dims - 100, 2)
    vx0, vy0 = np.random.uniform(-15, 15, 2)
    ax, ay = np.random.uniform(-0.3, 0.3, 2)
    trajectory = np.zeros((length, 6))
    x, y, vx, vy = x0, y0, vx0, vy0
    for step in range(length):
        trajectory[step] = [
            x + np.random.normal(0, noise_level),
            y + np.random.normal(0, noise_level),
            vx,
            vy,
            ax,
            ay,
        ]
        x = np.clip(x + vx + 0.5 * ax, 0, dims)
        y = np.clip(y + vy + 0.5 * ay, 0, dims)
        vx += ax
        vy += ay
    return trajectory


def generate_ctrv_trajectory(length=120, noise_level=0.1, dims=4096):
    x, y = np.random.uniform(100, dims - 100, 2)
    speed = np.random.uniform(5, 25)
    yaw = np.random.uniform(0, 2 * np.pi)
    yaw_rate = np.random.uniform(-0.08, 0.08)
    trajectory = np.zeros((length, 6))
    for step in range(length):
        vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
        ax = -speed * np.sin(yaw) * yaw_rate
        ay = speed * np.cos(yaw) * yaw_rate
        trajectory[step] = [
            x + np.random.normal(0, noise_level),
            y + np.random.normal(0, noise_level),
            vx,
            vy,
            ax,
            ay,
        ]
        if abs(yaw_rate) > 1e-5:
            x += (speed / yaw_rate) * (np.sin(yaw + yaw_rate) - np.sin(yaw))
            y += (speed / yaw_rate) * (-np.cos(yaw + yaw_rate) + np.cos(yaw))
        else:
            x += vx
            y += vy
        yaw = (yaw + yaw_rate) % (2 * np.pi)
        x, y = np.clip(x, 0, dims), np.clip(y, 0, dims)
    return trajectory


def generate_fighter_trajectory(length=120, noise_level=0.15, dims=4096):
    x, y = np.random.uniform(200, dims - 200, 2)
    speed = np.random.uniform(15, 40)
    yaw = np.random.uniform(0, 2 * np.pi)
    yaw_rate = np.random.uniform(-0.03, 0.03)
    trajectory = np.zeros((length, 6))
    for step in range(length):
        if np.random.random() < 0.05:
            yaw_rate = np.random.uniform(-0.2, 0.2)
        if np.random.random() < 0.03:
            speed = np.clip(speed + np.random.uniform(-10, 10), 5, 50)
        vx, vy = speed * np.cos(yaw), speed * np.sin(yaw)
        ax = -speed * np.sin(yaw) * yaw_rate
        ay = speed * np.cos(yaw) * yaw_rate
        trajectory[step] = [
            x + np.random.normal(0, noise_level),
            y + np.random.normal(0, noise_level),
            vx,
            vy,
            ax,
            ay,
        ]
        if abs(yaw_rate) > 1e-5:
            x += (speed / yaw_rate) * (np.sin(yaw + yaw_rate) - np.sin(yaw))
            y += (speed / yaw_rate) * (-np.cos(yaw + yaw_rate) + np.cos(yaw))
        else:
            x += vx
            y += vy
        yaw = (yaw + yaw_rate) % (2 * np.pi)
        x, y = np.clip(x, 0, dims), np.clip(y, 0, dims)
    return trajectory


def generate_aircraft_trajectories(
    num_samples=3000,
    seq_length=120,
    ca_ratio=0.33,
    ctrv_ratio=0.34,
    fighter_ratio=0.33,
    save_path=None,
):
    del fighter_ratio  # The final branch receives the remaining probability mass.
    np.random.seed(42)
    trajectories, types = [], []
    for _ in range(num_samples):
        draw = np.random.random()
        if draw < ca_ratio:
            trajectory, type_id = generate_ca_trajectory(seq_length), 0
        elif draw < ca_ratio + ctrv_ratio:
            trajectory, type_id = generate_ctrv_trajectory(seq_length), 1
        else:
            trajectory, type_id = generate_fighter_trajectory(seq_length), 2
        trajectories.append(trajectory)
        types.append(type_id)

    trajectories = np.asarray(trajectories, dtype=np.float32)
    types = np.asarray(types, dtype=np.int32)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as stream:
            pickle.dump(
                {
                    "cache_version": AIRCRAFT_CACHE_VERSION,
                    "state_channels": ["x", "y", "vx", "vy", "ax", "ay"],
                    "trajectories": trajectories,
                    "types": types,
                },
                stream,
            )
    return trajectories, types
