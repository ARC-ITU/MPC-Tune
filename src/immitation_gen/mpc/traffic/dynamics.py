from __future__ import annotations
import numpy as np
from shapely.geometry import LineString, Point

from immitation_gen.mpc.traffic.mpc_config import default_vehicle_constants as vehicle_constants
from immitation_gen.mpc.traffic.mpc_config import mpc_constants

import casadi as ca


def symbolic_vehicle_model(state, u):
    x = state[0]; y = state[1]; si = state[2]; vx = state[3]; vy = state[4]; omega = state[5]
    delta = u[0]; acceleration = u[1]

    Ffy = vehicle_constants["Caf"] * (delta - ca.atan2(vy + vehicle_constants["lf"] * omega, vx))
    Fry = -vehicle_constants["Car"] * ca.atan2((vy - vehicle_constants["lr"] * omega), vx)

    dx = vx * ca.cos(si) - vy * ca.sin(si)
    dy = vx * ca.sin(si) + vy * ca.cos(si)

    dsi = omega

    dvx = omega * vy + acceleration
    dvy = -omega * vx + (2 / vehicle_constants["mass"]) * (Ffy * ca.cos(delta) + Fry)

    domega = (2 / vehicle_constants["Iz"]) * (vehicle_constants["lf"] * Ffy - vehicle_constants["lr"] * Fry)

    return ca.vertcat(dx, dy, dsi, dvx, dvy, domega)



def rk4(x, u, dt):
    k1 = dt * symbolic_vehicle_model(x, u)
    k2 = dt * symbolic_vehicle_model(x + 0.5 * k1, u)
    k3 = dt * symbolic_vehicle_model(x + 0.5 * k2, u)
    k4 = dt * symbolic_vehicle_model(x + k3, u)
    return x + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def build_reference_sequence(initial_state: np.ndarray, ref_track: LineString, N: int, dt: float):
    current_pos = Point(initial_state[0], initial_state[1])
    current_distance = ref_track.project(current_pos)
    vx = initial_state[3]
    vy = initial_state[4]
    current_speed = ca.sqrt(vx**2 + vy**2) * (vx / ca.sqrt(vx**2 + 1e-8))
    
    refs = []
    
    # move forward along track at current speed
    for i in range(N):
        future_distance = current_distance + (i + 1) * dt * current_speed
        future_distance = min(future_distance, ref_track.length - 0.1)
        
        ref_point = ref_track.interpolate(future_distance)
        lane_heading = get_lane_heading(ref_point.x, ref_point.y, ref_track)
        
        refs.append([ref_point.x, ref_point.y, lane_heading[0], lane_heading[1]])
    #print("refs",refs)
    return np.array(refs, dtype=float)


def goal_speed_cost(vx, vy,goal_speed):
    current_speed = ca.sqrt(vx**2 + vy**2) * (vx / ca.sqrt(vx**2 + 1e-8))
    return (goal_speed - current_speed) ** 2


def reference_cost(x, y, Px, Py):
    return (x - Px) ** 2 + (y - Py) ** 2


def orientation_cost(si, lane_heading_vector):
    car_vector = [ca.cos(si), ca.sin(si)]

    dot_product = (
        lane_heading_vector[0] * car_vector[0] + lane_heading_vector[1] * car_vector[1]
    )
    lane_curve_vector_magnitude =ca.sqrt(
        lane_heading_vector[0] ** 2 + lane_heading_vector[1] ** 2
    )
    car_vector_magnitude =ca.sqrt(car_vector[0] ** 2 + car_vector[1] ** 2)

    x = dot_product / (lane_curve_vector_magnitude * car_vector_magnitude)
    return (1 - x) ** 2


def acceleration_cost(acceleration):
    return acceleration**2

def get_lane_heading(X, Y, reference_track: LineString):
    distance = reference_track.project(Point(X, Y))
    reference_point = reference_track.interpolate(distance)

    threshold = 0.1
    next_point = reference_track.line_interpolate_point(distance + threshold)

    # at end of the track
    if next_point == reference_point:
        lane_curve_vector = (
            reference_track.coords[-1][0] - reference_track.coords[-2][0],
            reference_track.coords[-1][1] - reference_track.coords[-2][1],
        )
    else:
        lane_curve_vector = (
            next_point.x - reference_point.x,
            next_point.y - reference_point.y,
        )

    return lane_curve_vector

def symbolic_cost_function(input_state, u, ref_pos, lane_heading, weights):

    x = input_state[0] 
    y = input_state[1]
    si = input_state[2]
    vx = input_state[3]
    vy = input_state[4]
    omega = input_state[5]

    goal_speed = weights["goal_speed"] * goal_speed_cost(vx, vy, weights["goal_speed"])
    acc = weights["acceleration"] * acceleration_cost(u[1])
    tracking = weights["tracking"] * reference_cost(
        x, y, ref_pos[0], ref_pos[1]
    )
    orientation = weights["orientation"] * orientation_cost(
        si, lane_heading
    )

    cost = goal_speed + tracking + orientation

    cost_a = acc

    cost_t = cost + cost_a

    return cost_t
