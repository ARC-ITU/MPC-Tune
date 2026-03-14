import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from immitation_gen.mpc.traffic.mpc import MPC
from immitation_gen.mpc.traffic.mpc_config import default_vehicle_constants as vehicle_constants
from immitation_gen.mpc.traffic.dynamics import rk4
import casadi as ca

class SimpleVehicle:
    def __init__(self, initial_state):
        self.state = np.array(initial_state, dtype=float)
        self.goal_speed = vehicle_constants["goal_speed"]
        
    def get_state(self):
        return self.state.copy()
    
    def update_state(self, u, dt):
        # Simple integration using RK4 from dynamics

        
        # Convert to CasADi format for dynamics
        state_ca = ca.DM(self.state)
        u_ca = ca.DM(u)
        
        # Update state
        next_state = rk4(state_ca, u_ca, dt)
        self.state = np.array(next_state).flatten()

def create_reference_track():
    points = [
        # First horizontal segment (going right)
        (0, 0),
        (10, 0),
        (20, 0),
        (30, 0),
        
        # Smooth corner transition (right to up)
        (35, 0.5),
        (38, 1.5),
        (40, 3),
        (41, 5),
        (41.5, 7),
        (42, 10),
        
        # Vertical segment (going up)
        (42, 15),
        (42, 20),
        (42, 25),
        (42, 30),
        
        # Smooth corner transition (up to right)
        (42.5, 33),
        (43.5, 35),
        (45, 37),
        (47, 38),
        (50, 38.5),
        
        # Second horizontal segment (going right)
        (55, 38.5),
        (60, 38.5),
        (70, 38.5),
        (80, 38.5),
    ]
    return LineString(points)

def run_mpc_simulation(_initial_state, _weights, dt=vehicle_constants["time_step"]):
    # Initialize MPC
    #dt = vehicle_constants["time_step"]
    mpc = MPC(dt, weights=_weights)

    # Create reference track
    ref_track = create_reference_track()
    
    # Initial vehicle state: [x, y, heading, vx, vy, omega]
    initial_state = _initial_state
    vehicle = SimpleVehicle(initial_state)
    
    # Simulation parameters
    sim_time = 20.0  # seconds
    steps = int(sim_time / dt)
    
    # Storage for results
    states = []
    controls = []
    ref_points = []
    
    # Initial control input guess
    N = mpc.prediction_horizon
    u = np.zeros((N, 2))  # [steering, acceleration]
    
    #print(f"Starting simulation for {sim_time}s ({steps} steps)")
    
    for step in range(steps):
        current_time = step * dt
        current_state = vehicle.get_state()
        
        # Store current state
        states.append(current_state.copy())
        
        # Get reference point for plotting
        from shapely.geometry import Point
        current_pos = Point(current_state[0], current_state[1])
        
        ref_distance = ref_track.project(current_pos)
        ref_point = ref_track.interpolate(ref_distance)
        ref_points.append([ref_point.x, ref_point.y])
        
        # Run MPC optimization
        try:
            result = mpc.optimize(u, ref_track, vehicle)
            if result.success:
                u_opt = result.x.reshape(N, 2)
                control_input = u_opt[0]  # Apply first control
            else:
                #print(f"MPC failed at step {step}, using zero control")
                control_input = np.array([0.0, 0.0])
        except Exception as e:
            print(f"Error at step {step}: {e}")
            control_input = np.array([0.0, 0.0])
        
        controls.append(control_input.copy())
        
        # Update vehicle state
        vehicle.update_state(control_input, dt)
        
        # Prepare next iteration control guess
        if 'u_opt' in locals():
            u = np.roll(u_opt, -1, axis=0)
            u[-1] = u_opt[-1]  # Repeat last control
        
        #if step % 10 == 0:
        #    print(f"Step {step}/{steps}, Time: {current_time:.1f}s")
    
    return np.array(states), np.array(controls), np.array(ref_points), ref_track

def plot_results(states, controls, ref_points, ref_track):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trajectory
    ax1 = axes[0, 0]
    track_coords = np.array(ref_track.coords)
    ax1.plot(track_coords[:, 0], track_coords[:, 1], 'k--', label='Reference Track', linewidth=2)
    ax1.plot(states[:, 0], states[:, 1], 'b-', label='Vehicle Path', linewidth=2)
    ax1.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
    ax1.plot(states[-1, 0], states[-1, 1], 'ro', markersize=8, label='End')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Vehicle Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Controls
    ax2 = axes[0, 1]
    time = np.arange(len(controls)) * vehicle_constants["time_step"]
    ax2.plot(time, controls[:, 0], 'r-', label='Steering [rad]')
    ax2.plot(time, controls[:, 1], 'b-', label='Acceleration [m/s²]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Control Input')
    ax2.set_title('Control Inputs')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Speed
    ax3 = axes[1, 0]
    speeds = np.sqrt(states[:, 3]**2 + states[:, 4]**2)
    ax3.plot(time, speeds, 'g-', label='Speed')
    ax3.axhline(y=2.0, color='r', linestyle='--', label='Goal Speed')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Speed [m/s]')
    ax3.set_title('Vehicle Speed')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Tracking Error
    ax4 = axes[1, 1]
    tracking_errors = np.sqrt((states[:, 0] - ref_points[:, 0])**2 + 
                             (states[:, 1] - ref_points[:, 1])**2)
    ax4.plot(time, tracking_errors, 'm-', label='Tracking Error')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Error [m]')
    ax4.set_title('Tracking Error')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    initial_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # [x, y, heading, vx, vy, omega]
    weights = {
        "goal_speed": 5,
        "tracking": 1,
        "orientation": 1,
        "acceleration": 0.1,
    }
    states, controls, ref_points, ref_track = run_mpc_simulation(initial_state, weights)
    
    # Plot results
    plot_results(states, controls, ref_points, ref_track)
    
    # Print final statistics
    final_error = np.sqrt((states[-1, 0] - ref_points[-1, 0])**2 + 
                         (states[-1, 1] - ref_points[-1, 1])**2)
    mean_error = np.mean(np.sqrt((states[:, 0] - ref_points[:, 0])**2 + 
                                (states[:, 1] - ref_points[:, 1])**2))
    
    print(f"\nSimulation Complete!")
    print(f"Final tracking error: {final_error:.3f} m")
    print(f"Mean tracking error: {mean_error:.3f} m")

if __name__ == "__main__":
    # Run simulation

    initial_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # [x, y, heading, vx, vy, omega]
    weights = {
        "goal_speed": 1,
        "tracking": 1,
        "orientation": 1,
        "acceleration": 0.1,
    }
    states, controls, ref_points, ref_track = run_mpc_simulation(initial_state, weights)

    # Plot results
    plot_results(states, controls, ref_points, ref_track)
    
    # Print final statistics
    final_error = np.sqrt((states[-1, 0] - ref_points[-1, 0])**2 + 
                         (states[-1, 1] - ref_points[-1, 1])**2)
    mean_error = np.mean(np.sqrt((states[:, 0] - ref_points[:, 0])**2 + 
                                (states[:, 1] - ref_points[:, 1])**2))
    
    print(f"\nSimulation Complete!")
    print(f"Final tracking error: {final_error:.3f} m")
    print(f"Mean tracking error: {mean_error:.3f} m")