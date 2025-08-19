import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import random
import json 
import math
import csv
import os
import ast
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import io

st.set_page_config(layout="wide")
st.title("Lap Simulation & Strategy Optimizer - Shell Eco-Marathon Battery Electric Vehicle")

st.subheader("Lap Simulation & Strategy Optimization")
st.info("I developed this lap simulation program during my Group Design Business Project in the first semester of 2025. The main goal of the simulation is to identify the optimal strategy at a given circuit for the Shell Eco-Marathon")

with st.expander("Key Features"):
    st.write("""
            - The simulation is modular and adaptable to any vehicle or circuit. These are easily changed by altering parameters in the code.

            - The simulation              

             """) 

# ================ SIMULATION PROGRAM ================
# Vehicle Parameters
vehicle_mass = 79 #71.25 
vehicle_reaction_force = vehicle_mass * 9.81
cdA = 0.0345
coefficient_friction_tire = 0.5
air_density = 1.19 #kg/m^3
frontal_area = 0.8 #m^2
tire_pressure = 413685 #Pa
wheel_diam =  0.48 #0.50 # meters
wheel_radius = 0.25
wheel_outer_radius = 0.25
wheel_inner_radius = 0.2
wheel_mass = 1.5
motor_torque_constant = 0.136
motor_speed_constant = 70.2
height_cog = 0.19 # meters
track_width = 0.5 # meters
gear_ratio = 6.856
wheelbase = 0.5
rollover_limit = (9.81 * track_width) / (2 * height_cog)
wheel_inertia_value = 0.5 * wheel_mass * (wheel_outer_radius**2 - wheel_inner_radius**2)

def import_track_data(filename):
    # Extract file extension
    global lat_ref, long_ref, latitude, longitude
    global y_coordinates, x_coordinates
    global distance_between_points, total_dist, elevation

    # Output containers
    latitude, longitude, elevation = [], [], []

    # Determine file type
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        # Read plain text file (skip header)
        with open(filename, "r") as file:
            next(file)
            for line in file:
                data = line.strip().split()
                latitude.append(float(data[1]))
                longitude.append(float(data[2]))
                elevation.append(float(data[3]))
    elif ext == ".csv":
        # Read CSV using pandas
        #print("Processing as CSV file.")
        df = pd.read_csv(filename)
        latitude = df["Latitude"].tolist()
        longitude = df["Longitude"].tolist()
        elevation = df["Elevation"].tolist()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Reference point for coordinate transformation
    lat_ref = latitude[0]
    long_ref = longitude[0]

    # Convert latitude/longitude to local x, y using equirectangular projection
    x_coordinates, y_coordinates = [], []
    for lat, lon in zip(latitude, longitude):
        x = 6371000 * np.cos(np.radians(lat_ref)) * np.radians(lon - long_ref)
        y = 6371000 * np.radians(lat - lat_ref)
        x_coordinates.append(x)
        y_coordinates.append(y)

    # Calculate distance between consecutive points
    distance_between_points = [
        math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        for (x1, y1), (x2, y2) in zip(zip(x_coordinates, y_coordinates), zip(x_coordinates[1:], y_coordinates[1:]))
    ]

    total_dist = sum(distance_between_points)

    # Calculate smoothed elevation gradient (lookahead = 100 points)
    elevation_gradient = []
    for current_index in range(len(elevation)):
        lookahead = min(current_index + 100, len(elevation) - 1)
        delta_elevation = elevation[lookahead] - elevation[current_index]
        delta_dist = sum(distance_between_points[current_index:lookahead]) if lookahead > current_index else 0
        gradient_rad = math.atan(delta_elevation / delta_dist) if delta_dist != 0 else 0
        elevation_gradient.append(gradient_rad)

    return {
        "x_coordinates": x_coordinates,
        "y_coordinates": y_coordinates,
        "total_distance": total_dist,
        "elevation_gradient": elevation_gradient
    }

# Define raw motor performance data (empirically determined or estimated)
rpm_data = np.array([0, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
torque_data = np.array([1.5, 1.25, 1, 0.75, 0.5, 0.5, 0.5, 0.5])
efficiency_data = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

# Create interpolation functions for torque and efficiency vs RPM
raw_torque = interp1d(rpm_data, torque_data, kind="cubic", fill_value="extrapolate")
raw_eff = interp1d(rpm_data, efficiency_data, kind="cubic", fill_value="extrapolate")

def torque_function(rpm):

    rpm = np.asarray(rpm)
    torque = raw_torque(rpm)
    torque[rpm > 4000] = torque_data[-1]  # Clamp beyond known data
    return torque

def eff_function(rpm):
    rpm = np.asarray(rpm)
    eff = raw_eff(rpm)
    eff[rpm > 4000] = efficiency_data[-1]  # Clamp beyond known data
    return eff

def torque_output(current_rpm):
    # Interpolated values
    torque = torque_function(current_rpm)
    efficiency = eff_function(current_rpm)

    # Convert RPM to angular velocity in rad/s
    angular_velocity = current_rpm * 0.10472  # (2 * pi / 60)

    # Calculate mechanical output power (Nm * rad/s = W)
    mechanical_output = torque * angular_velocity

    # Method 1: Electrical input based on output & efficiency
    current_energy_input = mechanical_output / efficiency

    # Method 2: Estimate current and voltage, then calculate electrical power
    voltage = angular_velocity / motor_speed_constant  # V = ω / K_v
    current = torque / motor_torque_constant           # I = τ / K_t
    current_energy_input_method2 = voltage * current

    return torque, efficiency, current_energy_input, current_energy_input_method2

def calculate_curvature(x, y, step):
    global kappa

    n = len(x)
    kappa = np.zeros(n)  # Initialize curvature array

    for i in range(step, n - step):
        # Extract three spaced points around the current point
        x1, y1 = x[i - step], y[i - step]
        x2, y2 = x[i],         y[i]
        x3, y3 = x[i + step], y[i + step]

        # Calculate side lengths of triangle
        a = np.hypot(x2 - x1, y2 - y1)
        b = np.hypot(x3 - x2, y3 - y2)
        c = np.hypot(x3 - x1, y3 - y1)

        # Heron's formula (safeguarded against floating point errors)
        s = 0.5 * (a + b + c)
        area_squared = max(s * (s - a) * (s - b) * (s - c), 0.0)

        if area_squared == 0.0:
            kappa[i] = 0.0  # Points are collinear
            continue

        area = np.sqrt(area_squared)

        # κ = 1 / R = 4 * area / (a * b * c)
        kappa[i] = 4.0 * area / (a * b * c)

        # Determine curvature sign using 2D cross product
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        kappa[i] *= np.sign(cross)

    # Pad the ends to avoid NaNs or outliers due to edge effects
    kappa[:step] = kappa[step]
    kappa[-step:] = kappa[-step - 1]

    return kappa

def generate_strategies(total_distance, num_strategies, force_start, min_events, max_events, event_length_range, accel_range, safety_margin):
    num_events = random.randint(min_events, max_events)
    events = []

    iteration_limit = 10000
    iterations = 0

    # Optionally add a guaranteed event at the very start of the lap
    if force_start:
        first_length = random.uniform(*event_length_range)
        first_accel = random.uniform(*accel_range)
        events.append((0.0, first_length, first_accel))

    while len(events) < num_events and iterations < iteration_limit:
        iterations += 1

        # Randomly choose a start point within bounds
        start = random.uniform(0.0, total_distance - event_length_range[0])

        # Calculate how much room is left from this point to track end
        max_valid_length = min(event_length_range[1], total_distance - start)
        if max_valid_length < event_length_range[0]:
            continue  # Not enough space for even the shortest valid event

        # Randomly select a valid event length and acceleration
        event_length = random.uniform(event_length_range[0], max_valid_length)
        accel_value = random.uniform(*accel_range)

        # Check for overlaps with already-placed events (using safety margin)
        conflict = False
        for existing_start, existing_length, _ in events:
            existing_end = existing_start + existing_length
            proposed_end = start + event_length

            if (start < existing_end + safety_margin) and (proposed_end + safety_margin > existing_start):
                conflict = True
                break

        if conflict:
            continue  # Try again

        # If no conflict, add the new event
        events.append((start, event_length, accel_value))

    # Sort all events by their start position
    events.sort(key=lambda e: e[0])
    return events

def simulate_lap_with_initial_velocity(strategy, initial_velocity, dt=0.01, max_time=15000, max_iterations=50000):
    global time_elapsed, energy_consumed, current_distance, velocity_profile, distance_profile, time_profile, finished

    # --- Initialisation ---
    current_distance = 0
    current_velocity = initial_velocity
    time_elapsed = 0
    energy_consumed = 0

    lap_distance = sum(distance_between_points)
    cumulative_distances = np.cumsum(distance_between_points)
    tire_pressure_bar = tire_pressure / 100000  # Convert to bar

    # --- Output buffers ---
    velocity_profile = []
    distance_profile = []
    time_profile = []
    resistive_force_profile = []
    net_force_profile = []
    motor_torque_list = []
    rpm_list = []
    energy_consumption = []
    motor_status = []
    lat_acc = []
    lon_acc = []
    power_profile = []
    motor_status = []

    # --- Simulation exit criteria ---
    stall_threshold = 0.1  # m/s
    stall_iterations = 0
    max_stall_iterations = 1000000
    progress_epsilon = 1e-9
    small_progress_iterations = 0
    max_small_progress_iterations = 5000
    iteration_counter = 0

    g = 9.81

    # --- Main simulation loop ---
    while current_distance < lap_distance and iteration_counter < max_iterations:
        current_index = np.searchsorted(cumulative_distances, current_distance, side="right")
        current_ke = 0.5 * vehicle_mass * current_velocity**2

        # --- Elevation gradient estimation every few steps ---
        if iteration_counter % 5 == 0 and current_index < len(elevation) - 1:
            lookahead = min(current_index + 10, len(elevation) - 10)
            delta_elev = elevation[lookahead] - elevation[current_index]
            delta_dist = sum(distance_between_points[current_index:lookahead])
            elevation_gradient = math.atan(delta_elev / delta_dist) if delta_dist != 0 else 0
        else:
            elevation_gradient = 0

        # --- Determine if currently accelerating ---
        in_accel_event = any(start <= current_distance < start + length for start, length, _ in strategy)

        # --- Motor & RPM calculation ---
        current_rpm = (current_velocity / (math.pi * wheel_diam)) * 60 * gear_ratio

        # --- Lateral acceleration check (curvature-based) ---
        curvature = kappa[current_index] if current_index < len(kappa) else 0
        a_lat = current_velocity**2 * curvature
        lat_acc.append(a_lat)

        if a_lat > rollover_limit:
            #print("Vehicle Rolled Over")
            break

        # --- Longitudinal acceleration (if motor is ON) ---
        if in_accel_event:
            torque, eff, energy_input, _ = torque_output(current_rpm)
            motor_force = 0.8 * torque * gear_ratio / wheel_radius
            motor_accel = motor_force / vehicle_mass
            a_long = motor_accel
            lon_acc.append(a_long)
        else:
            motor_force = 0
            torque = 0
            energy_input = 0
            a_long = 0
            lon_acc.append(a_long)

        power_profile.append(torque * ((2*math.pi)/60) * current_rpm)

        # --- Resistive Forces ---
        air_resistance = 0.5 * air_density * cdA * current_velocity**2
        rolling_resistance = 0.005 + (1 / tire_pressure_bar) * (0.01 + 0.0095 * ((current_velocity * 3.6) / 100)**2)
        wheel_inertia_force = 0  # Angular accel assumed zero
        resistive_force = air_resistance + rolling_resistance + wheel_inertia_force
        resistive_force_profile.append(resistive_force)

        gravity_force = g * np.sin(elevation_gradient) * vehicle_mass 
        net_force = motor_force - resistive_force - gravity_force
        net_acc = net_force / vehicle_mass

        # --- Energy and motion update ---
        motor_work = motor_force * current_velocity * dt
        gravity_work = gravity_force * current_velocity * dt
        resistive_work = resistive_force * current_velocity * dt
        net_work = motor_work - resistive_work - gravity_work
        new_ke = max(current_ke + net_work, 0)
        new_velocity = math.sqrt(2 * new_ke / vehicle_mass)
        new_distance = current_distance + new_velocity * dt
        time_elapsed += dt

        # --- Save step data ---
        velocity_profile.append(new_velocity)
        distance_profile.append(new_distance)
        time_profile.append(time_elapsed)
        net_force_profile.append(net_force)
        motor_torque_list.append(torque)
        energy_consumed += 1.1 * energy_input * dt if in_accel_event else 0
        energy_consumption.append(energy_consumed)
        motor_status.append(1 if in_accel_event else 0)
        rpm_list.append(current_rpm)

        # --- Progress checks ---
        if new_velocity < stall_threshold:
            stall_iterations += 1
            if stall_iterations >= max_stall_iterations:
                #print("Lap Simulation Aborted: Vehicle Stalled")
                break
        else:
            stall_iterations = 0

        if (new_distance - current_distance) < progress_epsilon:
            small_progress_iterations += 1
            if small_progress_iterations >= max_small_progress_iterations:
                #print("Lap Simulation Aborted: No Significant Progress")
                break
        else:
            small_progress_iterations = 0

        # --- Advance state ---
        current_velocity = new_velocity
        current_distance = new_distance
        iteration_counter += 1

    finished = current_distance >= lap_distance

    # --- Save telemetry to CSV ---
    telemetry_df = pd.DataFrame({
        "Time": time_profile,
        "Velocity_kph": [v * 3.6 for v in velocity_profile],  # convert to km/h
        "Distance_m": distance_profile,
        "Motor_Torque_Nm": motor_torque_list,
        "Motor_Status": motor_status,
        "RPM": rpm_list,
        "Energy_Consumed_J": energy_consumption,
    })
    telemetry_df.to_csv("telemetry_output.csv", index=False)
    #print("Telemetry saved to telemetry_output.csv")

    return (
        time_elapsed,
        energy_consumed,
        current_distance,
        velocity_profile,
        distance_profile,
        time_profile,
        resistive_force_profile,
        power_profile,
        motor_status,
        finished,
        lat_acc,
        lon_acc,
        energy_consumption
    )

def simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=0, n_candidates=1000, optimization=True, events=1, progress_bar=None, status_text=None):
# Initialize race-level tracking variables
    race_time = 0
    race_energy = 0
    current_velocity = initial_velocity

    # Store best strategies and energy results
    best_strategy_lap1 = []
    best_strategy_lap2 = []
    lap1energy = []
    lap2energy = []

    # Data logging
    data = []
    analysis = []

    # Averages (currently unused)
    avg_start1, avg_start2, avg_length1, avg_length2 = [], [], [], []

    strategy_selection_num = 5  # Top N strategies to keep per lap
    
    strategies = []

    # Loop through two laps
    for lap in range(2):
        #print(f"====== Lap {lap + 1} ======")
        candidate_results = []

        accel_events = 2  # Fixed number of acceleration events

        for candidate in range(n_candidates):
            if progress_bar is not None:
                # fraction from 0.0 to 1.0
                done = (lap * n_candidates + candidate + 1) / (2 * n_candidates)
                progress_bar.progress(done)

            if status_text is not None:
                status_text.text(f"Lap {lap+1} / 2 - Candidate {candidate+1} / {n_candidates}")

            if optimization:
                # Generate strategy using your external optimizer
                lap1_strat, lap2_strat = optimize_strategies(filename_lap1, filename_lap2)

                if lap == 0:
                    strategy = lap1_strat
                else:
                    strategy = lap2_strat

            if lap == 0:
                force_start = True
                current_velocity = 1
            else:
                force_start = False
            
            strategy = generate_strategies(total_dist, 1, force_start, acceleration_events, acceleration_events, (10,500), (1.5,2.5), 20)

            # Simulate lap with selected strategy
            time_elapsed, energy_consumed, current_distance, velocity_profile, distance_profile, time_profile, resistive_force_profile, power_profile, motor_status, finished, lat_acc, lon_acc, energy_consumption = simulate_lap_with_initial_velocity(
                strategy, current_velocity, dt, max_time, max_iterations
            )

            # Get lap stats
            lap_time, energy_J, final_distance = time_elapsed, energy_consumed, current_distance
            avg_speed_kph = (final_distance / lap_time) * 3.6 if lap_time > 0 else 0

            # Calculate pseudo-efficiency metric
            if finished and avg_speed_kph > 25:
                efficiency = (final_distance / 1000) * (((energy_J / lap_time) * (lap_time / 3600)) / 1000)
                #print("Valid Strategy")
            else:
                efficiency = 0

            strategies.append({
                        "Lap" : lap,
                        "Candidate": candidate,
                        "Strategy": list(strategy),
                        "EnergyConsumption": energy_consumed,
                        "MaxVelocity": max(velocity_profile),
                        "Minimum Velocity": min(velocity_profile),
                    })

            # Store simulation result
            candidate_results.append((
                strategy, efficiency, lap_time, energy_J, final_distance,
                velocity_profile, time_profile, resistive_force_profile, avg_speed_kph
            ))

        # Filter out failed candidates (efficiency = 0)
        valid_candidates = [c for c in candidate_results if c[1] > 0]

        strategies_df = pd.DataFrame(strategies)
        strategies_df.to_csv("strategy_analysis.csv", index=False)
        
        #print(valid_candidates[0])

        # Sort by energy consumption (ascending) if valid candidates exist
        sorted_candidates = sorted(valid_candidates, key=lambda x: x[3]) if valid_candidates else []

        # Select top N candidates
        top_candidates = sorted_candidates[:strategy_selection_num]
        
        # From the top N, pick the one with highest energy usage (heuristic)
        best_candidate = max(top_candidates, key=lambda x: x[3]) if top_candidates else None

        if best_candidate:
            best_strategy = best_candidate[0]
            #print(best_strategy)
            best_energy = best_candidate[3]
            best_vel_profile = best_candidate[5]
            best_time_profile = best_candidate[6]
            best_force_profile = best_candidate[7]

        telemetry_lap_df = pd.DataFrame({
            "TimeProfile": best_time_profile,
            "VelProfile": best_vel_profile,
            "ForceProfile": best_force_profile 
        })

        telemetry_lap_df.to_csv(f"telemetry_lap{lap+1}.csv", index=False)

        # Save best strategy and accumulate energy
        if lap == 0:
            best_strategy_lap1.append(best_strategy)
            lap1energy.append(best_energy)
            race_energy += best_energy
        else:
            best_strategy_lap2.append(best_strategy)
            lap2energy.append(best_energy)
            race_energy += best_energy * 11  # Multiply lap 2 impact (e.g. weight in final score)

        race_time += best_candidate[2]
        current_velocity = best_vel_profile[-1] if best_vel_profile else current_velocity

        # Log top strategies and performance for analysis
        for idx, candidate in enumerate(top_candidates):
            strategy, efficiency, lap_time, energy_used, distance, v_profile, t_profile, f_profile, avg_speed = candidate

            data.append({
                "Lap": lap + 1,
                "Candidate Rank": idx + 1,
                "Strategy": json.dumps(strategy),
                "Efficiency": efficiency,
                "Lap Time": lap_time,
                "Energy Consumption": energy_used,
                "Distance": distance,
                "Average Speed": avg_speed
            })

            analysis.append({
                "Lap": lap + 1,
                "Velocity Profile": [float(v) for v in v_profile],
                "Time Profile": [float(t) for t in t_profile],
                "Force Profile": [float(f) for f in f_profile]
            })

    # Save strategy ranking and profile analysis per lap
    df = pd.DataFrame(data)
    df_analysis = pd.DataFrame(analysis)

    if optimization:
        for lap in sorted(df["Lap"].unique()):
            df[df["Lap"] == lap].to_csv(f"top_strategies_optimized_lap{lap}.csv", index=False)
    
    else:
        for lap in sorted(df["Lap"].unique()):
            df[df["Lap"] == lap].to_csv(f"top_strategies_lap{lap}.csv", index=False)
        for lap in [1,2]:
            df_lap = df[df["Lap"] == lap]
            filename = (f"top_strategies_optimized_lap{lap}.csv"
                        if optimization
                        else f"top_strategies_lap{lap}.csv")
            df_lap.to_csv(filename, index=False)

    #print("Saved")

    for lap in sorted(df_analysis["Lap"].unique()):
        df_analysis[df_analysis["Lap"] == lap].to_csv(f"analysis_lap{lap}.csv", index=False)

    # Return summary of race performance and best strategies
    return strategies_df, race_time, race_energy, [], [], current_velocity, best_strategy_lap1, best_strategy_lap2, lap1energy, lap2energy, avg_length1, avg_length2, avg_start1, avg_start2

def flatten_strategy(strategy):
    return [value for event in strategy for value in event]

def optimize_strategies(filename_lap1, filename_lap2):
    num_events = 2  # Fixed number of acceleration events per strategy

    # --- Load strategy data from CSV ---
    df_lap1 = pd.read_csv(filename_lap1)
    df_lap2 = pd.read_csv(filename_lap2)

    # Parse strategy JSON strings into Python lists
    strategies_lap1 = [json.loads(s) for s in df_lap1["Strategy"].tolist()]
    strategies_lap2 = [json.loads(s) for s in df_lap2["Strategy"].tolist()]

    # Flatten strategy lists for analysis
    flat_strategies_lap1 = [flatten_strategy(s) for s in strategies_lap1]
    flat_strategies_lap2 = [flatten_strategy(s) for s in strategies_lap2]

    # Separate out start and length values for each event
    start_lap1, length_lap1 = [], []
    start_lap2, length_lap2 = [], []

    for i in range(len(flat_strategies_lap1)):
        start_lap1.append([flat_strategies_lap1[i][3*j]     for j in range(num_events)])
        length_lap1.append([flat_strategies_lap1[i][3*j+1]  for j in range(num_events)])
        start_lap2.append([flat_strategies_lap2[i][3*j]     for j in range(num_events)])
        length_lap2.append([flat_strategies_lap2[i][3*j+1]  for j in range(num_events)])

    # --- Calculate statistical means ---
    avg_start_lap1 = [float(np.mean([start_lap1[i][j] for i in range(len(start_lap1))])) for j in range(num_events)]
    avg_length_lap1 = [float(np.mean([length_lap1[i][j] for i in range(len(length_lap1))])) for j in range(num_events)]
    avg_start_lap2 = [float(np.mean([start_lap2[i][j] for i in range(len(start_lap2))])) for j in range(num_events)]
    avg_length_lap2 = [float(np.mean([length_lap2[i][j] for i in range(len(length_lap2))])) for j in range(num_events)]

    # --- Calculate standard deviations (sample-based) ---
    std_start_lap1 = [np.std([start_lap1[i][j] for i in range(len(start_lap1))], ddof=1) for j in range(num_events)]
    std_length_lap1 = [np.std([length_lap1[i][j] for i in range(len(length_lap1))], ddof=1) for j in range(num_events)]
    std_start_lap2 = [np.std([start_lap2[i][j] for i in range(len(start_lap2))], ddof=1) for j in range(num_events)]
    std_length_lap2 = [np.std([length_lap2[i][j] for i in range(len(length_lap2))], ddof=1) for j in range(num_events)]

    # --- Generate new strategies using normal distribution sampling ---
    lap1_strat, lap2_strat = [], []
    factor = 1.5  # Shrink std to reduce variance in new strategies

    for evt in range(num_events):
        # Sample new values for each lap using reduced variance
        start1 = np.random.normal(loc=avg_start_lap1[evt], scale=std_start_lap1[evt] / factor)
        length1 = np.random.normal(loc=avg_length_lap1[evt], scale=std_length_lap1[evt] / factor)
        start2 = np.random.normal(loc=avg_start_lap2[evt], scale=std_start_lap2[evt] / factor)
        length2 = np.random.normal(loc=avg_length_lap2[evt], scale=std_length_lap2[evt] / factor)

        # Fixed acceleration of 2.0 (can be changed or made variable later)
        lap1_strat.append((start1, length1, 2.0))
        lap2_strat.append((start2, length2, 2.0))

    return lap1_strat, lap2_strat

def plot_strategy_overlay(x_coords, y_coords, strategy_lap1, strategy_lap2, title="Optimized Strategy"):
    # Step 1: Compute cumulative distance along track
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.insert(np.cumsum(np.sqrt(dx**2 + dy**2)), 0, 0)

    def get_indices(strategy):
        indices = []
        for start, length, _ in strategy:
            end = start + length
            idx_range = np.where((distances >= start) & (distances <= end))[0]
            indices.append(idx_range)
        return indices

    lap1_indices = get_indices(strategy_lap1)
    lap2_indices = get_indices(strategy_lap2)

    # Step 2: Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Lap 1 ---
    ax1.plot(x_coords, y_coords, color='lightgray', label='Track')
    for idx in lap1_indices:
        ax1.plot(np.array(x_coords)[idx], np.array(y_coords)[idx], color='red', linewidth=2)
    ax1.set_title(f"Lap 1 – {title}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()

    # --- Lap 2 ---
    ax2.plot(x_coords, y_coords, color='gray', label='Track')
    for idx in lap2_indices:
        ax2.plot(np.array(x_coords)[idx], np.array(y_coords)[idx], color='blue', linewidth=2)
    ax2.set_title(f"Lap 2 – {title}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend()

    plt.show()

st.subheader("Try the Simulation Yourself:")

track_selection = st.selectbox("Select the Circuit", ["None", "Silesia Ring - Poland", "Nogaro - France"], index=0)

if track_selection == "None":
    st.warning("Waiting...")
    st.stop()

if track_selection == "Silesia Ring - Poland":
    # Running Functions
    import_track_data("CircuitSilesiaRingData.csv")
    # corner_calc("SilesiaCornerData.csv", True)
    calculate_curvature(x_coordinates, y_coordinates, step=30)

    st.write(f"You selected {track_selection}, the venue for the 2025 SEM. Below you can see the track layout, and curvature along the track.")


st.write("The dataframes containing these values, alongside the elevation of the track are the foundation of the simulation as we are simulating the car at each individual point.")

coordinate_df = pd.DataFrame({"x": x_coordinates, "y": y_coordinates, "elevation": elevation, "kappa": kappa})

points = np.vstack([coordinate_df.x, coordinate_df.y]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap="viridis",
    linewidth=4
)
lc.set_array(coordinate_df.kappa)

curvature_fig, curvature_ax = plt.subplots(figsize=(7,5))
curvature_ax.add_collection(lc)
curvature_ax.set_title("Track Curvature")
curvature_ax.set_aspect("equal", "box")
curvature_ax.autoscale()
cbar = curvature_fig.colorbar(lc, ax=curvature_ax, label="Curvature (1/m)", shrink=0.8, pad=0.02)
buf_curvature = io.BytesIO()
curvature_fig.savefig(buf_curvature, format="png", dpi=100, bbox_inches="tight")
buf_curvature.seek(0)

track_map_fig, track_map_ax = plt.subplots(figsize=(7,5))
track_map_ax.plot(x_coordinates, y_coordinates)
track_map_ax.set_title("Track Map")
buf_track_map = io.BytesIO()
track_map_fig.savefig(buf_track_map, format="png", dpi=100, bbox_inches="tight")
buf_track_map.seek(0)

track_fig1, track_fig2 = st.columns([1, 1.4])
with track_fig1:
    st.image(buf_track_map, caption="Track Map", clamp=False, channels="RGB", output_format="PNG", width=500)
with track_fig2:
    st.write("**Track Characteristics**")
    st.write("- The Silesia Ring is a 1.317 km long track. In the SEM, as per the regulations, the car must drive 16 km, which in this track equated to 11 laps.")
    st.write("- The **start/finish line** can be found on the straight in the top-right section of the track, and the car follows a clock-wise direction to complete the lap.")
    st.write("**Driving Strategy**")
    st.write("- At first glance, the track doesn't have too much elevation, only a total of 3 metres from the highest to lowest point on the track. However, make no mistake, this elevation profile is quite impactful on the car.")
    st.write("- We will use the simulation to decide on **how many acceleration events we should perform per lap** and also **how long each acceleration event should be**.")

st.subheader("Vehicle Characteristics")
st.write("Enter below the vehicle settings you'd like to test. If you'd like to simulate the concept vehicle designed by our team, leave it as it is!")
veh_1, veh_2, veh_3, veh_4, veh_5, veh_6 = st.columns(6)
with veh_1:
    vehicle_mass = st.number_input("**Vehicle Mass (*kg*)**", min_value=65, max_value=150, value=78, width=200)
with veh_2:
    cdA = st.number_input("**Coefficient of Drag**", min_value=0.0, max_value=2.0, value=0.0345, width=200)
with veh_3:
    tire_pressure = st.number_input("**Tire Pressure (*Pa*)**", min_value=120000, max_value=680000, width=200)
with veh_4:
    wheel_radius = st.number_input("**Wheel Radius (*m*)**", min_value=0.2, max_value=1.0, value=0.48, width=200)
with veh_5:
    wheel_mass = st.number_input("**Wheel Mass (*kg*)**", min_value=1.0, max_value=4.0, value=1.5, width=200)
with veh_6:
    gear_ratio = st.number_input("**Gear Ratio**", min_value=1.0, max_value=10.0, value=6.856, width=200)

veh_21, veh_22, veh_23, veh_24, veh_25, veh_26 = st.columns(6)
with veh_23:
    track_width = st.number_input("**Track Width (*m*)**", min_value=0.5, max_value=1.5, value=0.5, width=200)
with veh_24:
    height_cog = st.number_input("**Height CoG (*m*)**", min_value=0.05, max_value=0.40, value=0.19, width=200)

with st.expander("**Understanding Vehicle Parameter Setup Optimization**"):
    st.write("**Vehicle Mass (kg):** Higher mass slows acceleration, increases braking distance, and raises energy consumption.")
    st.write("**Coefficient of Drag (Cd):** Higher Cd increases aerodynamic drag, reducing top speed and efficiency at high velocities.")
    st.write("**Tire Pressure (Pa):** Higher pressure lowers rolling resistance (better efficiency) but may reduce grip; lower pressure improves grip but increases drag.")
    st.write("**Wheel Radius (m):** Larger radius increases top speed for a given RPM but reduces acceleration; smaller radius improves acceleration but lowers top speed.")
    st.write("**Wheel Mass (kg):** Higher wheel mass increases rotational inertia, making acceleration and deceleration less responsive.")
    st.write("**Gear Ratio:** Higher ratios improve acceleration but reduce top speed; lower ratios increase top speed but slow acceleration.")
    st.write("**Track Width (m):** Wider track improves cornering stability but can increase aerodynamic drag and rolling resistance.")
    st.write("**Height CoG (m):** Higher center of gravity reduces stability in corners, increasing rollover risk and limiting maximum cornering speed.")

st.subheader("Simulation Setup")
st.write("""
         Here you have two options:
         
         - You can either use the simulation to find the optimal strategy for you

         - Define your own strategy and simulate a race to see the results. 
         """)

st.write("What defines a strategy is **the number of acceleration events in a lap, and for how long you accelerate**. At first, you'd think that accelerating as little as possible is the best strategy," \
"however, that might not be the case. If you, for example, get to a high speed at the start of the run, and continuously top up the speed so that you maintain an average speed of 25 km/h, you might consume less energy by accelerating let's say 3-7 km/h, then doing a " 
"big acceleration of 12-15 km/h. **It is a trade off - and this simulation program helps you identify what the best trade off is!**")

simulation_setting = st.radio("**Select your Choice**", ["None", "Find the Optimum Strategy", "Create Your Own Strategy"], index=0, horizontal=True)

if simulation_setting == "None":
    # wipe any previous results so nothing leaks through
    for k in ["simulation_done", "simulation_done_custom", "custom_results"]:
        if k in st.session_state:
            del st.session_state[k]
    st.stop()  # hard stop: no further UI rendered

if simulation_setting == "Find the Optimum Strategy":
    sim_set_1, sim_set_2, sim_set_3, sim_set_4 = st.columns(4)
    with sim_set_1:
        acceleration_events = st.number_input("Number of Acceleration Events/Lap", min_value=1, max_value=10, value=2, width=300)
    with sim_set_2:
        num_strats = st.number_input("Number of Strategies to Evaluate", min_value=10, max_value=10000, value=1000, width=300)
    with sim_set_3:
        if st.button("Run Simulation"):
            with sim_set_4:
                total = 2 * num_strats 
                progress_bar = st.progress(0.0)
                status_text  = st.empty()

                simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=0.5, n_candidates=num_strats, optimization=False, events=acceleration_events, progress_bar=progress_bar, status_text=status_text)
                st.session_state.simulation_done = True

# ---- Place this where your "Create Your Own Strategy" branch is ----
elif simulation_setting == "Create Your Own Strategy":
    # Ensure state keys
    if "custom_strategy_df" not in st.session_state:
        st.session_state.custom_strategy_df = None
    if "custom_results" not in st.session_state:
        st.session_state.custom_results = None
    if "simulation_done_custom" not in st.session_state:
        st.session_state.simulation_done_custom = False

    st.write("Define how many acceleration events you want, then set the **start (m)** and **length (m)** of each event.")

    num_events = st.number_input("Number of Acceleration Events:", min_value=1, max_value=12, value=2, step=1)

    # Seed defaults (evenly spaced)
    if (st.session_state.custom_strategy_df is None) or (len(st.session_state.custom_strategy_df) != num_events):
        # use total_dist from import_track_data()
        default_starts = np.linspace(0, max(1.0, total_dist * 0.8), num_events).round(1) if "total_dist" in globals() else np.linspace(0, 1000, num_events).round(1)
        default_lengths = np.full(num_events, max(5.0, (total_dist / (num_events * 8)) if "total_dist" in globals() else 30.0)).round(1)
        st.session_state.custom_strategy_df = pd.DataFrame({"start_m": default_starts, "length_m": default_lengths})

    edited = st.data_editor(
        st.session_state.custom_strategy_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "start_m": st.column_config.NumberColumn("Start (m)", min_value=0.0, step=0.1),
            "length_m": st.column_config.NumberColumn("Length (m)", min_value=0.1, step=0.1),
        },
        key="custom_strategy_editor",
    )
    st.session_state.custom_strategy_df = edited

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        accel_value = st.number_input("Acceleration Value (arb.)", min_value=0.1, max_value=5.0, value=2.0, step=0.1,
                                      help="This is the accel value used inside each event. Matches your (start, length, accel) triple.")
    with col_b:
        init_v = st.number_input("Initial Velocity (m/s)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col_c:
        dt_val = st.number_input("Time Step dt (s)", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

    # Build strategy [(start, length, accel)] from the table
    df = st.session_state.custom_strategy_df.copy()
    # Basic in-bounds clean-up: end = start + length, clamp to track if available
    if "total_dist" in globals():
        df["length_m"] = np.minimum(df["length_m"], np.maximum(0.1, total_dist - df["start_m"]))
    strategy_custom = [(float(r.start_m), float(r.length_m), float(accel_value)) for r in df.itertuples(index=False)]

    run_custom = st.button("Simulate this custom strategy")

    if run_custom:
        st.session_state.simulation_done_custom = False  # reset

        # Run ONE LAP with your existing function
        (time_elapsed, energy_consumed, current_distance, vprof, dprof, tprof,
         resistive_force_profile, power_profile, motor_status, finished,
         lat_acc, lon_acc, energy_consumption) = simulate_lap_with_initial_velocity(
            strategy_custom, initial_velocity=float(init_v),
            dt=float(dt_val), max_time=15000, max_iterations=50000
        )

        # Store results for display
        st.session_state.custom_results = {
            "time_elapsed": time_elapsed,
            "energy_consumed": energy_consumed,
            "current_distance": current_distance,
            "vprof": vprof,
            "dprof": dprof,
            "tprof": tprof,
            "energy_consumption": energy_consumption,
            "motor_status": motor_status,
        }
        st.session_state.simulation_done_custom = True

# ---- Full-width results for the CUSTOM strategy (single lap) ----
if st.session_state.get("simulation_done_custom", False):
    st.subheader("Custom Strategy — Single-Lap Results")

    r = st.session_state.custom_results
    lap_time = r["time_elapsed"]
    lap_dist = r["current_distance"]
    avg_speed_kph = (lap_dist / lap_time) * 3.6 if lap_time > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Lap Time (s)", f"{lap_time:.2f}")
    with m2:
        st.metric("Lap Distance (m)", f"{lap_dist:.1f}")
    with m3:
        st.metric("Avg Speed (km/h)", f"{avg_speed_kph:.2f}")

    # Side-by-side graphs
    col1, col2 = st.columns(2)

    with col1:
        fig_v, ax_v = plt.subplots(figsize=(4, 3))  # smaller size
        ax_v.plot(r["tprof"], [v*3.6 for v in r["vprof"]], linewidth=2)
        ax_v.set_xlabel("Time (s)")
        ax_v.set_ylabel("Velocity (km/h)")
        ax_v.set_title("Velocity Profile")
        st.pyplot(fig_v, use_container_width=True)

    with col2:
        fig_m, ax_m = plt.subplots(figsize=(4, 3))  # smaller size
        ax_m.step(r["dprof"], r["motor_status"], where="post")
        ax_m.set_xlabel("Distance (m)")
        ax_m.set_ylabel("Motor (0=OFF, 1=ON)")
        ax_m.set_title("Motor On/Off")
        st.pyplot(fig_m, use_container_width=True)
    

if st.session_state.get("simulation_done", False):
    st.subheader("Simulation Output")

    lap1_tel = pd.read_csv("telemetry_lap1.csv")
    lap2_tel = pd.read_csv("telemetry_lap2.csv")

    lap2_tel["TimeProfile"] = lap2_tel["TimeProfile"] + lap1_tel["TimeProfile"].iloc[-1]

    lap_tel_df = pd.concat([lap1_tel, lap2_tel])
    average_speed = lap_tel_df["VelProfile"].mean()

    fig_velocity1, ax_velocity1 = plt.subplots()
    ax_velocity1.plot(lap_tel_df["TimeProfile"], lap_tel_df["VelProfile"])
    ax_velocity1.axvline(x=lap2_tel["TimeProfile"].iloc[0], color="red", linestyle="--", linewidth=1, label="Beginning of Lap 2")
    ax_velocity1.axhline(y=average_speed, color="green", linestyle="--", linewidth=1, label=f"Average Speed ({average_speed*3.6:.2f} km/h)")
    ax_velocity1.axhline(y=6.94444444, color="black", linestyle="--", linewidth=1, label="Minimum Average Speed (25 km/h)")
    ax_velocity1.set_title("Velocity Profile - Lap 1")
    ax_velocity1.set_ylabel("Velocity (km/h)")
    ax_velocity1.legend()
    ax_velocity1.set_xlabel("Time (s)")
    vel1_buff = io.BytesIO()
    fig_velocity1.savefig(vel1_buff, format="png", dpi=100, bbox_inches="tight")
    vel1_buff.seek(0)

    st.info("""
                **Insights**

                - This is the result of a **perfect run**, and it is the combined best of lap 1 and lap 2 strategies. However, it must be noted that this result is only true in this environment, where assumptions and simplifications have been made.

                - One of the biggest features lacking in this simulation is the inclusion of **wind modelling**. With the car being as aerodynamical as possible, the wind has a great effect on it - sometimes even requiring changes in the strategy.
                """)

    vel_col1, vel_col2 = st.columns([1,2])
    with vel_col1:
        st.image(vel1_buff, caption="Velocity Profile Lap 1 & 2", clamp=False, channels="RGB", output_format="PNG", width=450)
    with vel_col2:
        with st.info("**Results**"):
            st.write(f"**Regulations Check:** The average speed was {average_speed*3.6:.2f} km/h, which is above the required minimum average speed of 25 km/h, therefore this strategy passes this constraint.")
        st.write("**Lap 1 Energy Consumption:**")
        st.write("**Lap 2 Energy Consumption:**")

    stats_analysis = pd.read_csv("strategy_analysis.csv")

    lap1 = stats_analysis[stats_analysis["Lap"]==0]
    lap2 = stats_analysis[stats_analysis["Lap"]==1]

    strategies = [json.loads(s) for s in stats_analysis["Strategy"].tolist()]
    flat_strategies = [flatten_strategy(s) for s in strategies]

    starting_pos = [x for i, x in enumerate(strategies) if i % 3 == 0]
    length = [x for i, x in enumerate(strategies) if i % 3 == 1]

    with st.expander("Statistical Analysis of Strategies Generated"):
        st.write("It is interesting to analyse the strategies generated. It is possible to potentially identify patterns between acceleration events and overall energy consupmption. The more strategies evaluated, the better the data!")
        st.write(f"Out of the {num_strats} generated, the mean starting position is {starting_pos.mean()}")


    def parse_strategy(s):
        list_of_lists = ast.literal_eval(s)
        return [tuple(inner) for inner in list_of_lists]

    lap1_df = pd.read_csv("top_strategies_lap1.csv")
    lap2_df = pd.read_csv("top_strategies_lap2.csv")
    with st.expander("In-Depth Telemetry"):
        telemetry_lap = st.selectbox("Select Lap", ["Lap 1", "Lap 2"], index=0)
        if telemetry_lap == "Lap 1":
            strategy = parse_strategy(lap1_df["Strategy"].iloc[0])
            initial_velocity = 1
        elif telemetry_lap == "Lap 2":
            strategy = parse_strategy(lap2_df["Strategy"].iloc[0])
            initial_velocity = lap1_tel["VelProfile"].iloc[-1]

        time_elapsed, energy_consumed, current_distance, velocity_profile, distance_profile, time_profile, resistive_force_profile, power_profile, motor_status, finished, lat_acc, lon_acc, energy_consumption = simulate_lap_with_initial_velocity(strategy, initial_velocity, dt=0.01, max_time=15000, max_iterations=50000)
        
        fig, axes = plt.subplots(
            nrows=5,
            ncols=1,
            sharex=True,
            figsize=(6,15),
            tight_layout=True
        )

        axes[0].plot(distance_profile, velocity_profile, lw=2)
        axes[0].set_ylabel("Velocity (m/s)")
        axes[0].set_title("Silesia Ring Lap 1 - Simulated Telemetry")

        axes[1].plot(distance_profile, resistive_force_profile, lw=2)
        axes[1].set_ylabel("Resistive Force During Lap (N)")

        axes[2].plot(distance_profile, motor_status, lw=2)
        axes[2].set_ylabel("Motor On/Off")
        axes[2].set_yticks([0,1])
        axes[2].set_yticklabels(["OFF", "ON"])

        axes[3].plot(distance_profile, energy_consumption, lw=2)
        axes[3].set_ylabel("Energy Consumption during Lap (J)")

        axes[4].plot(distance_profile, lat_acc, lw=2, label="Lateral Acc")
        axes[4].plot(distance_profile, lon_acc, lw=2, label="Longitudinal Acc")
        axes[4].set_ylabel("Lateral/Longitudinal Acc (m/s^2)")

        plt.subplots_adjust(hspace=0.3)

        tel_col1, tel_col2 = st.columns([1, 2])
        with tel_col1:
            st.pyplot(fig, use_container_width=False)

