"""The first seventeen seconds of Doom, scripted: spawn in the hangar, grab the
green armour, run out into the nukage courtyard, kill the first zombieman.

Every processor derives its frame from the same clock and the same tables here,
so the muzzle flash, the pistol report and the status bar's ammo count agree
without any of them talking to each other.
"""
from __future__ import annotations

import math

FPS = 35
DEMO_SECONDS = 17.0
VIEW_HEIGHT = 41  # Doom's eye height above the floor
TICS_PER_SECOND = 35

# (time, x, y, angle_degrees). Position and angle interpolate between rows.
WAYPOINTS = [
    (0.0, 1056, -3616, 90),
    (1.2, 1056, -3616, 90),
    (2.0, 1056, -3500, 180),
    (3.0, 780, -3500, 180),
    (3.4, 780, -3420, 90),
    (3.9, 780, -3232, 90),
    (4.4, 700, -3232, 180),
    (6.0, -180, -3232, 180),
    (6.3, -200, -3232, 180),
    (7.1, -200, -3232, 0),
    (8.9, 700, -3232, 0),
    (9.3, 780, -3232, 0),
    (9.8, 780, -3420, 270),
    (10.2, 780, -3500, 0),
    (11.3, 1376, -3500, 0),
    (13.4, 2700, -3500, 0),
    (13.8, 2860, -3480, 300),
    (14.2, 2860, -3480, 298),
    (15.6, 2860, -3480, 298),
    (16.0, 2860, -3480, 330),
    (DEMO_SECONDS, 2860, -3480, 330),
]

ARMOUR_PICKUP_TIME = 6.1
PISTOL_FIRE_TIMES = [14.3, 14.7, 15.1, 16.2, 16.6]
PISTOL_FLASH_SECONDS = 0.12
ITEM_FLASH_SECONDS = 0.35

# The two live things the demo interacts with, by thing position.
ZOMBIEMAN_ONE = (3056, -3584)
ZOMBIEMAN_ONE_SIGHT_TIME = 13.5
ZOMBIEMAN_ONE_DEATH_TIME = 15.1
IMP_SIGHT_TIME = 13.9
ZOMBIEMAN_ONE_FIRE_TIMES = [14.9]

STARTING_BULLETS = 50


def pose_at(t: float) -> tuple[float, float, float]:
    """(x, y, angle_radians) at demo time t, with ease-in-out between waypoints."""
    t = max(0.0, min(DEMO_SECONDS, t))
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u)
            a0r, a1r = math.radians(a0), math.radians(a1)
            da = (a1r - a0r + math.pi) % (2 * math.pi) - math.pi
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, a0r + da * u
    t_end, x, y, a = WAYPOINTS[-1]
    return float(x), float(y), math.radians(a)


def is_moving(t: float) -> bool:
    x0, y0, _ = pose_at(t - 0.05)
    x1, y1, _ = pose_at(t + 0.05)
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 > 4.0


def pistol_flash_active(t: float) -> bool:
    return any(0.0 <= t - f < PISTOL_FLASH_SECONDS for f in PISTOL_FIRE_TIMES)


def bullets_at(t: float) -> int:
    return STARTING_BULLETS - sum(1 for f in PISTOL_FIRE_TIMES if t >= f)


def armour_at(t: float) -> int:
    return 100 if t >= ARMOUR_PICKUP_TIME else 0


def palette_at(t: float) -> int:
    """Which of PLAYPAL's fourteen palettes this frame is shown through."""
    since_pickup = t - ARMOUR_PICKUP_TIME
    if 0.0 <= since_pickup < ITEM_FLASH_SECONDS:
        return 12 - min(3, int(since_pickup / ITEM_FLASH_SECONDS * 4))
    return 0
