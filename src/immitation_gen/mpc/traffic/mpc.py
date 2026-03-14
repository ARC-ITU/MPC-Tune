from __future__ import annotations 

import casadi as ca
import numpy as np
import logging

from immitation_gen.mpc.traffic.mpc_config import mpc_constants, default_vehicle_constants as vehicle_constants
from immitation_gen.mpc.traffic.dynamics import *

class MPC:
    def __init__(self, timestep: float, weights):
        self.prediction_horizon = mpc_constants["prediction_horizon"]
        self.time_step = float(timestep)
        self.control_interval = mpc_constants["control_interval"]
        self.control_inputs = 2
        self.minimum_distance = mpc_constants["closest_distance_threshold"]
        self.weights = weights
        self.k_neighbours = 2

        self._prev_solution = None
        self.build_solver()


    def build_solver(self):
        N = int(self.prediction_horizon)
        dt = float(self.time_step)

        # Decision variables
        U = [ca.SX.sym(f"u_{i}", self.control_inputs) for i in range(N)]
        U_flat = ca.vertcat(*U)

        # Parameters: [x0(6), refs(4*N)]
        self.nx = 6
        self.nref = 4 * N
        self.goal_speed = 1   # for goal speed weight
        p = ca.SX.sym("p", self.nx + self.nref + self.goal_speed)
        #p = ca.SX.sym("p", self._nx + self._nref)

        x0 = p[0:6]
        refs = p[6:6 + self.nref]
        goal_speed_weight = p[-1]

        X = [x0]
        total_cost = ca.SX(0)

        for i in range(N):
            x_next = rk4(X[i], U[i], dt)
            X.append(x_next)

            ref_x = refs[4 * i + 0]
            ref_y = refs[4 * i + 1]
            ref_heading_x = refs[4 * i + 2]
            ref_heading_y = refs[4 * i + 3]
            ref_pos = (ref_x, ref_y)
            lane_heading = (ref_heading_x, ref_heading_y)
            
            stage_cost = symbolic_cost_function(X[i+1], U[i], ref_pos, lane_heading,self.weights)
            total_cost += stage_cost
     
        g_list = []

        for i in range(1, N):
            # steering difference constraint
            g_list.append((U[i][0] - U[i-1][0]) / dt)
            # acceleration difference constraint
            g_list.append((U[i][1] - U[i-1][1]) / dt)

        # Build NLP
        nlp = {"x": U_flat, "f": total_cost, "g": ca.vertcat(*g_list), "p": p}

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-2,
            "ipopt.dual_inf_tol": 1e-2,
            "ipopt.constr_viol_tol": 1e-2,
            "ipopt.acceptable_tol": 5e-2,
            "ipopt.acceptable_iter": 10,
            "ipopt.linear_solver": "mumps",
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }

        self._solver = ca.nlpsol("mpc_ipopt", "ipopt", nlp, opts)
        self.g_dim = len(g_list)
        self._N = N

 
    def optimize(self, u, reference_track: LineString, current_vehicle, road_network=None):
        N = self._N
        u = np.asarray(u, dtype=float).reshape(N, self.control_inputs)

        x_init = np.copy(current_vehicle.get_state()).astype(float)

        refs = build_reference_sequence(x_init, reference_track, N, self.time_step)
        
        refs_flat = refs.flatten()
        goal_speed = current_vehicle.goal_speed
        
        p_num = np.concatenate([x_init, refs_flat, [goal_speed]])
        # Input bounds only
        max_steer = vehicle_constants["max_steering_angle"]  
        max_accel = vehicle_constants["max_acceleration"]  
        
        lbx = [-max_steer, -max_accel] * N
        ubx = [max_steer, max_accel] * N
        lbx = np.array(lbx, dtype=float)
        ubx = np.array(ubx, dtype=float)
 
        x0 = self.get_initial_guess(current_vehicle, reference_track, N)
        #x0 = np.zeros(N * self.control_inputs, dtype=float)
        x0 = np.clip(x0, lbx, ubx)
        
        max_steer_rate = vehicle_constants["max_steering_turn_rate"]
        max_accel_rate = vehicle_constants["max_acceleration_change_rate"]
        lbg_list = []
        ubg_list = []
        for i in range(1, N):
            # steering rate bound for step i
            lbg_list.append(-max_steer_rate)
            ubg_list.append( max_steer_rate)
            # acceleration rate bound for step i
            lbg_list.append(-max_accel_rate)
            ubg_list.append( max_accel_rate)
            
        lbg = np.array(lbg_list, dtype=float)
        ubg = np.array(ubg_list, dtype=float)
        
        sol = self._solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_num)
        
        solver_stats = self._solver.stats()
        if solver_stats['return_status'] not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
            #logging.warning(f"Solver status: {solver_stats['return_status']}")
            return self._get_safe_fallback()
        
        u_opt = np.array(sol["x"]).squeeze().astype(float)
        
        if np.any(np.isnan(u_opt)) or np.any(np.isinf(u_opt)):
            return self._get_safe_fallback()
        
        self._prev_solution = u_opt.copy()
        class _Result: 
            pass
        result = _Result()
        result.x = u_opt
        result.f = float(sol["f"]) if "f" in sol else None
        result.success = True
        
        return result

        

    def get_initial_guess(self, current_vehicle, reference_track, N):
       
        state = current_vehicle.get_state()

        lane_vec = get_lane_heading(state[0], state[1], reference_track)
        target_heading = np.arctan2(lane_vec[1], lane_vec[0])
        heading_error = target_heading - state[2]

        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        target_steering = heading_error * 0.5
   
        current_speed = np.sqrt(state[3]**2 + state[4]**2)
        speed_error = current_vehicle.goal_speed - current_speed
        target_accel = speed_error * 0.2

        u_init = np.zeros(N * 2)
        u_init[0::2] = target_steering  
        u_init[1::2] = target_accel    
        return u_init



    def _get_safe_fallback(self):
        N = self._N
        
        fallback_u = np.zeros(N * 2)
        fallback_u[1::2] = 0.5  
        
        class _Result: 
            pass
        result = _Result()
        result.x = fallback_u
        result.f = 1e6
        result.success = False
        
        return result

    def simulate(self, u, reference_track: LineString, current_vehicle, road_network):
        result = self.optimize(u, reference_track, current_vehicle, road_network)
        
        if not result.success:
            #logging.warning("Using fallback controls")
            pass
        
        control_outputs = result.x.reshape(self.prediction_horizon, self.control_inputs)
       
        return control_outputs