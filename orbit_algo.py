"""Calculate future boot times (UTC) for target passes and store in a list

#params
Epoch: '2026-03-05T12:00:00'
semimajor_axis = 1887.4 #km (Moon radius 1737.4 + 150 km altitude)
eccentricity = 0.0 #circular 
inclination = 90    #deg (polar orbit)
ascending_long = 45 #deg
perigee_arg = 120   #deg
true_anom = 30      #deg

#config
#TODO: integrate with GUI
target_long = 0.0    
target_lat = 0.0     

num_clustered_passes = 5

#read current satellite pos from tele
Epoch = '2026-03-05T12:00:00 UTC'
#Frame: Moon-centered inertial (MCI)

#Position (km):
x = 1835.2
y = -112.5
z = 98.4

#Velocity (km/s):
vx = 0.032
vy = 1.622
vz = -0.015

#calculate next 100 orbits and the min distance to target center (assuming sphere)

#check if any passes ensure smaller than 30 degress offset from top-down view, if not, return message to state out of range

#if yes, record fastest and clostest pass, and load times of passes around it with number of num_clustered passes into a new array

#print array of boot times


Compute future camera boot times for lunar surface passes.

This script follows the pseudocode originally written as comments.  No
external APIs are used; the orbit is propagated using simple two–body
Keplerian motion and a spherical Moon model.  Given classical orbital elements
(or an initial state vector) the code calculates the times of the next 100
orbits, finds the moment during each orbit when the spacecraft is closest to a
specified lat/long on the lunar surface, and then identifies a cluster
of passes around the best one.

Configuration occurs at the top of the file; to integrate with a GUI simply
export or set the same parameters programmatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import sin, cos, sqrt, atan2, atan, asin, acos, pi
from typing import List, Tuple

import numpy as np

# constants
MU_MOON = 4902.800066  # km^3 / s^2, standard gravitational parameter for the Moon
MOON_RADIUS = 1737.4   # km


def _latlon_to_vector(lat_deg: float, lon_deg: float, radius: float = MOON_RADIUS) -> np.ndarray:
    """Convert a surface (lat, lon) to a Moon-centred cartesian vector."""
    lat = np.radians(lat_deg)
    long = np.radians(lon_deg)
    x_coord = radius * cos(lat) * cos(long)
    y_coord = radius * cos(lat) * sin(long)
    z_coord = radius * sin(lat)
    return np.array([x_coord, y_coord, z_coord])


def _central_angle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return central angle between two surface points (radians)."""
    # spherical law of cosines
    return acos(
        sin(lat1) * sin(lat2)
        + cos(lat1) * cos(lat2) * cos(lon1 - lon2)
    )


def _solve_kepler(mean_anomaly: float, eccentricity: float, tolerance: float = 1e-10) -> float:
    """Solve Kepler's equation E - e*sin(E) = M for eccentric anomaly E."""
    # initial guess
    if eccentricity < 0.8:
        eccentric_anomaly = mean_anomaly
    else:
        eccentric_anomaly = pi
    for _ in range(50):
        residual = eccentric_anomaly - eccentricity * sin(eccentric_anomaly) - mean_anomaly
        derivative = 1 - eccentricity * cos(eccentric_anomaly)
        anomaly_correction = -residual / derivative
        eccentric_anomaly += anomaly_correction
        if abs(anomaly_correction) < tolerance:
            break
    return eccentric_anomaly


def _orbital_position(semimajor_axis: float, eccentricity: float, inclination: float, raan: float, arg_perigee: float,
                      initial_mean_anomaly: float, time_seconds: float) -> np.ndarray:
    """Compute ECI position after ``time_seconds`` seconds from epoch.

    All angles are in radians.  Returns a 3‑vector in the Moon‑centred inertial
    frame (same as MCI in the comments).
    """
    # mean motion
    mean_motion = sqrt(MU_MOON / semimajor_axis ** 3)
    current_mean_anomaly = initial_mean_anomaly + mean_motion * time_seconds
    eccentric_anomaly = _solve_kepler(current_mean_anomaly % (2 * pi), eccentricity)
    # true anomaly
    true_anomaly = 2 * atan2(sqrt(1 + eccentricity) * sin(eccentric_anomaly / 2), sqrt(1 - eccentricity) * cos(eccentric_anomaly / 2))
    orbital_radius = semimajor_axis * (1 - eccentricity * cos(eccentric_anomaly))
    # position in perifocal frame
    perifocal_position = np.array([orbital_radius * cos(true_anomaly), orbital_radius * sin(true_anomaly), 0.0])
    # rotation matrices
    rotation_z = lambda angle: np.array([
        [ cos(angle), -sin(angle), 0],
        [ sin(angle),  cos(angle), 0],
        [         0,           0, 1],
    ])
    rotation_x = lambda angle: np.array([
        [1,         0,          0],
        [0, cos(angle), -sin(angle)],
        [0, sin(angle),  cos(angle)],
    ])
    # transform from perifocal to inertial
    rotation = rotation_z(-raan) @ rotation_x(-inclination) @ rotation_z(-arg_perigee)
    return rotation @ perifocal_position


def compute_boot_times(
    epoch_str: str,
    semimajor_axis: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    arg_perigee_deg: float,
    true_anomaly_deg: float,
    target_lat: float,
    target_lon: float,
    num_orbits: int = 100,
    num_clustered: int = 5,
) -> List[datetime]:
    """Return candidate boot times around the best pass.

    Parameters match the variables in the source comments; the epoch string
    should include a timezone (e.g. 'UTC').
    """
    epoch = datetime.fromisoformat(epoch_str)
    epoch = epoch.replace(tzinfo=timezone.utc)
    # convert angles to radians
    inclination = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    arg_perigee = np.radians(arg_perigee_deg)
    initial_true_anomaly = np.radians(true_anomaly_deg)
    # compute initial mean anomaly
    initial_eccentric_anomaly = 2 * atan2(sqrt(1 - eccentricity) * sin(initial_true_anomaly / 2), sqrt(1 + eccentricity) * cos(initial_true_anomaly / 2))
    initial_mean_anomaly = initial_eccentric_anomaly - eccentricity * sin(initial_eccentric_anomaly)
    # period
    orbital_period = 2 * pi * sqrt(semimajor_axis ** 3 / MU_MOON)
    target_vector = _latlon_to_vector(target_lat, target_lon)
    target_lat_rad = np.radians(target_lat)
    target_lon_rad = np.radians(target_lon)

    pass_times: List[Tuple[datetime, float]] = []

    # sample each orbit at 360 points to find minimum distance
    samples = 360
    for orbit_index in range(num_orbits):
        time_base = orbit_index * orbital_period
        best_distance = float("inf")
        best_pass_time = None
        for sample_index in range(samples):
            elapsed_time = time_base + (sample_index / samples) * orbital_period
            position = _orbital_position(semimajor_axis, eccentricity, inclination, raan, arg_perigee, initial_mean_anomaly, elapsed_time)
            # convert to lat/lon for central-angle check
            position_magnitude = np.linalg.norm(position)
            lat = asin(position[2] / position_magnitude)
            long = atan2(position[1], position[0])
            # central angle
            central_angle = acos(
                sin(lat) * sin(target_lat_rad)
                + cos(lat) * cos(target_lat_rad) * cos(long - target_lon_rad)
            )
            # distance to surface point
            position_difference = position - target_vector
            distance = np.linalg.norm(position_difference)
            if distance < best_distance:
                best_distance = distance
                best_pass_time = epoch + timedelta(seconds=elapsed_time)
        if best_pass_time is not None:
            pass_times.append((best_pass_time, best_distance))

    # filter passes by 30° central-angle at the time of closest approach
    qualified: List[Tuple[datetime, float]] = []
    for pass_time, pass_distance in pass_times:
        # recompute angle at that time
        elapsed_time = (pass_time - epoch).total_seconds()
        position = _orbital_position(semimajor_axis, eccentricity, inclination, raan, arg_perigee, initial_mean_anomaly, elapsed_time)
        position_magnitude = np.linalg.norm(position)
        lat = asin(position[2] / position_magnitude)
        long = atan2(position[1], position[0])
        central_angle = acos(
            sin(lat) * sin(target_lat_rad)
            + cos(lat) * cos(target_lat_rad) * cos(long - target_lon_rad)
        )
        if central_angle <= np.radians(30):
            qualified.append((pass_time, pass_distance))

    if not qualified:
        raise ValueError("No passes within 30° of the target available in the next"
                         f" {num_orbits} orbits.")

    # pick the pass with smallest distance
    qualified.sort(key=lambda time_distance_pair: time_distance_pair[1])
    best_pass_index = 0
    # now construct a time-sorted list and take neighbours
    times_only = [time_distance_pair[0] for time_distance_pair in sorted(qualified, key=lambda time_distance_pair: time_distance_pair[0])]
    # locate the best pass in the time-ordered list
    best_pass_time = qualified[best_pass_index][0]
    best_pass_idx = times_only.index(best_pass_time)
    half = num_clustered // 2
    start_idx = max(0, best_pass_idx - half)
    end_idx = min(len(times_only), start_idx + num_clustered)
    return times_only[start_idx:end_idx]


if __name__ == "__main__":
    # simple demonstration using the original commented values
    epoch = "2026-03-05T12:00:00+00:00"
    boot_times = compute_boot_times(
        epoch, 1837, 0.002, 90, 45, 120, 30,
        target_lat=90.0, target_lon=180,
    )
    print("Boot times around best pass:")
    for boot_time in boot_times:
        print(boot_time.isoformat()) 