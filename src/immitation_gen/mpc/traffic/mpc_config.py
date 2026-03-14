import numpy as np
from typing import TypedDict


class VehicleConstants(TypedDict):
    mass: float
    Iz: float
    lf: float
    lr: float
    Car: float
    Caf: float
    ax: float
    car_l: float
    car_w: float
    max_steering_angle: float
    max_acceleration: float
    time_step: float
    scan_radius: float
    goal_speed: float
    lookahead_distance: float


L = 2.9  # [m] Wheel base of vehicle
dt = 0.2  # [s] Time step
Lr = L / 2.0  # [m]
Lf = L - Lr  # [m]
Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg
max_steering_angle = np.deg2rad(50)  # [rad] Maximum steering angle
max_steering_turn_rate = np.deg2rad(30) / dt  # [rad/s] Maximum steering turn rate
max_acceleration = 2.0  # [m/s²] Maximum acceleration
max_acceleration_change_rate = 1.0  # [m/s³] Maximum acceleration change rate

scan_radius = 30  # [m] Scan radius
goal_speed = 2  # [m/s] Goal speed
lookahead_distance = 0  # [m] Lookahead distance

default_vehicle_constants: VehicleConstants = {
    "mass": m,
    "Iz": Iz,
    "lf": Lf,
    "lr": Lr,
    "Car": Cr,
    "Caf": Cf,
    "ax": 0.1,
    "car_l": L * 1.7,  # 1.7 is a scaling factor
    "car_w": L * 0.8,
    "max_steering_angle": max_steering_angle,
    "max_steering_turn_rate": max_steering_turn_rate,
    "max_acceleration": max_acceleration,
    "max_acceleration_change_rate": max_acceleration_change_rate,
    "time_step": dt,
    "scan_radius": scan_radius,
    "goal_speed": goal_speed,
    "lookahead_distance": lookahead_distance,
}

class MPCWeights(TypedDict):
    tracking: float
    steering_angle: float
    acceleration: float
    goal_speed: float
    orientation: float


class MPCConstants(TypedDict):
    prediction_horizon: int
    control_interval: int
    dt: list[float]
    weights: MPCWeights
    


control_interval = 3  # [s] Control interval
prediction_horizon = 6  # [s] Prediction horizon
closest_distance_threshold = 6  # [m] Closest distance threshold
dt = [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]  # [s] Time step for each prediction step

mpc_constants = {
    "prediction_horizon": prediction_horizon,
    "control_interval": control_interval,
    "closest_distance_threshold": closest_distance_threshold,
    "dt":dt,
    "weights": {
       
        "tracking": 0.0,
        "acceleration": 0.0,
        "goal_speed": 0.0,
        "orientation":0,
        "seperation": 0.0,
        "local_speed_deviation": 0.5,
    },
}
