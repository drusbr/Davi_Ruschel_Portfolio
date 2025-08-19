from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
import random
import json 
import math
import csv
import os
import ast

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

wind_speed = 0
wind_direction = 0

# Track Data
# This will import a file saved from matlab that contains the x and y coordinates of the track as a list.
def import_track_data(filename):
    # Extract file extension
    global lat_ref, long_ref, latitude, longitude
    global y_coordinates, x_coordinates
    global distance_between_points, total_dist, elevation

    """
    Reads GPS-based track data from a file and converts latitude/longitude into Cartesian coordinates.
    Also calculates segment distances, total track distance, and smoothed elevation gradient.

    Parameters:
        filename (str): Path to the .csv or .txt file containing the track data.

    Returns:
        dict: {
            'x_coordinates': List of x positions (meters),
            'y_coordinates': List of y positions (meters),
            'total_distance': Total track length (meters),
            'elevation_gradient': List of elevation gradients (radians)
        }
    """

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
        print("Processing as CSV file.")
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

# def corner_calc(corner_data_file, corrected=True):
#     global corner_radius_list, sorted_corner_idx_entry, sorted_corner_idx_apex, sorted_corner_idx_exit, corner_velocity

#     corner_x, corner_y = [], []
#     corner_data = pd.read_csv(corner_data_file, names=["Longitude", "Latitude"])

#     if corrected == False:
#         corner_longitude = corner_data["Longitude"].tolist()
#         corner_latitude = corner_data["Latitude"].tolist()

#         for point in range(len(corner_latitude)):
#             corner_x.append(6371000 * np.cos(np.radians(lat_ref)) * np.radians(corner_longitude[point] - long_ref))
#             corner_y.append(6371000 * np.radians(corner_latitude[point] - lat_ref))
#     else:
#         corner_longitude = corner_data["Longitude"].tolist()
#         corner_latitude = corner_data["Latitude"].tolist()
#         corner_x.extend(corner_longitude)
#         corner_y.extend(corner_latitude)

#     corner_radius_list = [] 

#     current_corner = 0
#     for corner in range(0, len(corner_x)//3):
#         entry_x, entry_y = corner_x[corner], corner_y[corner]
#         apex_x, apex_y = corner_x[corner+1], corner_y[corner+1]
#         exit_x, exit_y = corner_x[corner+2], corner_y[corner+2]

#         distance_a = math.sqrt((apex_x - entry_x)**2 + (apex_y - entry_y)**2)
#         distance_b = math.sqrt((exit_x - apex_x)**2 + (exit_y - apex_y)**2)
#         distance_c = math.sqrt((exit_x - entry_x)**2 + (exit_y - entry_y)**2)

#         angle_A_cos = (distance_b**2 + distance_c**2 - distance_a**2) / (2 * distance_b * distance_c)
#         angle_A = math.acos(angle_A_cos)
#         angle_P = math.radians(180) - angle_A

#         radius = distance_a / (2*math.sin(angle_P))

#         corner_radius_list.append(radius)
#         current_corner += 3

#     corner_idx_entry = []
#     corner_idx_apex = []
#     corner_idx_exit = []

#     for corner in range(0, len(corner_x)//3):
#         entry_x, entry_y = corner_x[corner*3], corner_y[corner*3]
#         apex_x, apex_y = corner_x[corner*3+1], corner_y[corner*3+1]
#         exit_x, exit_y = corner_x[corner*3+2], corner_y[corner*3+2]

#         distances_entry = [math.sqrt((entry_x - x_coordinates[i])**2 + (entry_y - y_coordinates[i])**2) 
#                      for i in range(len(x_coordinates))]
        
#         distances_apex = [math.sqrt((apex_x - x_coordinates[i])**2 + (apex_y - y_coordinates[i])**2) 
#                      for i in range(len(x_coordinates))]
        
#         distances_exit = [math.sqrt((exit_x - x_coordinates[i])**2 + (exit_y - y_coordinates[i])**2) 
#                      for i in range(len(x_coordinates))]
        
#         closest_index_entry = np.argmin(distances_entry)
#         closest_index_apex = np.argmin(distances_apex)
#         closest_index_exit = np.argmin(distances_exit)

#         corner_idx_entry.append(int(closest_index_entry))
#         corner_idx_apex.append(int(closest_index_apex))
#         corner_idx_exit.append(int(closest_index_exit))

#     sorted_corner_idx_entry = sorted(corner_idx_entry)
#     sorted_corner_idx_apex = sorted(corner_idx_apex)
#     sorted_corner_idx_exit = sorted(corner_idx_exit)

#     corner_velocity = []
#     for corner in range(0, len(corner_radius_list)):
#         max_corner_velocity = math.sqrt(math.sqrt((coefficient_friction_tire * vehicle_reaction_force)**2)/(vehicle_mass/corner_radius_list[corner])**2 + (0.5*air_density*cdA)**2)
#         corner_velocity.append(max_corner_velocity)
    
#     return corner_radius_list, sorted_corner_idx_entry, sorted_corner_idx_apex, sorted_corner_idx_exit, corner_velocity

# Last Year's Motor
# rpm_data = np.array([0, 1000, 1500, 2000, 2500, 3000, 3500])
# torque_data = np.array([1.5, 1.25, 1, 0.75, 0.5, 0.5, 0.5])
# efficiency_data = np.array([0.5, 0.72, 0.81, 0.84, 0.84, 0.84, 0.84])


# Define raw motor performance data (empirically determined or estimated)
rpm_data = np.array([0, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
torque_data = np.array([1.16, 1.16, 1.16, 1.16, 1.16, 1.16, 1.16, 1.16]) #np.array([1.5, 1.25, 1, 0.75, 0.5, 0.5, 0.5, 0.5])
efficiency_data = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

# Create interpolation functions for torque and efficiency vs RPM
raw_torque = interp1d(rpm_data, torque_data, kind="cubic", fill_value="extrapolate")
raw_eff = interp1d(rpm_data, efficiency_data, kind="cubic", fill_value="extrapolate")

def torque_function(rpm):
    """
    Interpolates torque at a given RPM using a cubic spline.
    Caps torque at the maximum defined RPM (4000).
    """
    rpm = np.asarray(rpm)
    torque = raw_torque(rpm)
    torque[rpm > 4000] = torque_data[-1]  # Clamp beyond known data
    return torque

def eff_function(rpm):
    """
    Interpolates efficiency at a given RPM using a cubic spline.
    Caps efficiency at the maximum defined RPM (4000).
    """
    rpm = np.asarray(rpm)
    eff = raw_eff(rpm)
    eff[rpm > 4000] = efficiency_data[-1]  # Clamp beyond known data
    return eff

def torque_output(current_rpm):
    """
    Calculates torque, efficiency, and energy input at a given RPM.
    
    Parameters:
        current_rpm (float): Motor shaft speed in RPM
    
    Returns:
        tuple:
            - torque (Nm)
            - efficiency (0–1)
            - energy_input_method1 (W) = mechanical_output / efficiency
            - energy_input_method2 (W) = voltage * current (theoretical electrical input)
    """

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
    """
    Calculates the signed curvature κ (1/m) for a 2D path defined by x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of x and y coordinates (same length), representing a path in metres.
    step : int
        Number of points before and after each point to form the 3-point arc.
        Larger step = smoother curvature, but less local detail.

    Returns
    -------
    kappa : np.ndarray
        Array of signed curvature values:
            - Positive for left turns
            - Negative for right turns
            - Zero for straight lines
    """
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
    """
    Generates a list of randomly placed acceleration events along a track.

    Parameters
    ----------
    total_distance : float
        Total length of the track (in meters).
    num_strategies : int
        Number of strategies to generate (used externally, not within function).
    force_start : bool
        If True, ensures the first event always starts at position 0.
    min_events, max_events : int
        Range of number of acceleration events per strategy.
    event_length_range : tuple (float, float)
        Minimum and maximum length of each acceleration event (in meters).
    accel_range : tuple (float, float)
        Range of acceleration values to assign to each event.
    safety_margin : float
        Minimum buffer distance (in meters) between events.

    Returns
    -------
    events : list of tuples
        Each tuple is (start_position, event_length, acceleration_value)
    """

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
    
    """
    Simulates a vehicle lap using a given acceleration strategy and initial velocity.

    Parameters
    ----------
    strategy : list of tuples
        Each tuple is (start_position, length, acceleration_value).
    initial_velocity : float
        Starting velocity of the vehicle (m/s).
    dt : float
        Time step for integration (seconds).
    max_time : float
        Maximum simulation time (seconds).
    max_iterations : int
        Maximum loop iterations to prevent infinite simulation.

    Returns
    -------
    tuple :
        time_elapsed, energy_consumed, final_distance,
        velocity_profile, distance_profile, time_profile,
        resistive_force_profile, finished_flag,
        lateral_acceleration_profile, longitudinal_acceleration_profile
    """

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
            print("Vehicle Rolled Over")
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

        gravity_force = g * np.sin(elevation_gradient) * vehicle_mass * dt
        net_force = motor_force - resistive_force
        net_acc = net_force / vehicle_mass - gravity_force / vehicle_mass

        # --- Energy and motion update ---
        motor_work = motor_force * current_velocity * dt
        resistive_work = resistive_force * current_velocity * dt
        net_work = motor_work - resistive_work - gravity_force
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
                print("Lap Simulation Aborted: Vehicle Stalled")
                break
        else:
            stall_iterations = 0

        if (new_distance - current_distance) < progress_epsilon:
            small_progress_iterations += 1
            if small_progress_iterations >= max_small_progress_iterations:
                print("Lap Simulation Aborted: No Significant Progress")
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
    print("Telemetry saved to telemetry_output.csv")

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

def simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=0, n_candidates=1000, optimization=True, events=1):
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

    # Loop through two laps
    for lap in range(2):
        print(f"====== Lap {lap + 1} ======")
        candidate_results = []

        accel_events = 2  # Fixed number of acceleration events

        for candidate in range(n_candidates):
            print("Candidate: ", candidate)

            if optimization:
                # Generate strategy using your external optimizer
                lap1_strat, lap2_strat = optimize_strategies(filename_lap1, filename_lap2)

                if lap == 0:
                    strategy = lap1_strat
                else:
                    strategy = lap2_strat
            else:
                if lap == 0:
                    force_start = True
                
                strategy = generate_strategies(total_dist, 1, force_start, events, events, (10,150), (1.5,2.5), 20)

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
                print("Valid Strategy")
            else:
                efficiency = 0

            # Store simulation result
            candidate_results.append((
                strategy, efficiency, lap_time, energy_J, final_distance,
                velocity_profile, time_profile, resistive_force_profile, avg_speed_kph
            ))

        # Filter out failed candidates (efficiency = 0)
        valid_candidates = [c for c in candidate_results if c[1] > 0]

        # Sort by energy consumption (ascending) if valid candidates exist
        sorted_candidates = sorted(valid_candidates, key=lambda x: x[3]) if valid_candidates else []

        # Select top N candidates
        top_candidates = sorted_candidates[:strategy_selection_num]

        # From the top N, pick the one with highest energy usage (heuristic)
        best_candidate = max(top_candidates, key=lambda x: x[3]) if top_candidates else None

        if best_candidate:
            best_strategy = best_candidate[0]
            best_energy = best_candidate[3]
            best_vel_profile = best_candidate[5]
            best_time_profile = best_candidate[6]
            best_force_profile = best_candidate[7]

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

    print("Saved")

    for lap in sorted(df_analysis["Lap"].unique()):
        df_analysis[df_analysis["Lap"] == lap].to_csv(f"analysis_lap{lap}.csv", index=False)

    # Return summary of race performance and best strategies
    return race_time, race_energy, [], [], current_velocity, best_strategy_lap1, best_strategy_lap2, lap1energy, lap2energy, avg_length1, avg_length2, avg_start1, avg_start2

def flatten_strategy(strategy):
    return [value for event in strategy for value in event]

def optimize_strategies(filename_lap1, filename_lap2):
    """
    Optimizes two-lap acceleration strategies using statistical analysis of past candidates.

    Parameters
    ----------
    filename_lap1 : str
        CSV file path containing strategy data for lap 1.
    filename_lap2 : str
        CSV file path containing strategy data for lap 2.

    Returns
    -------
    tuple:
        lap1_strat : list of tuples (start, length, accel)
        lap2_strat : list of tuples (start, length, accel)
    """

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

# Running Functions
import_track_data("CircuitSilesiaRingData.csv")
# corner_calc("SilesiaCornerData.csv", True)
calculate_curvature(x_coordinates, y_coordinates, step=30)

strategy1 = [(0, 179.32584110834168, 2), (356.1923297628317, 67.05170726865427, 2)]
# time_elapsed, energy_consumed, current_distance, velocity_profile, distance_profile, time_profile, resistive_force_profile, motor_status, finished, lat_acc, lon_acc = simulate_lap_with_initial_velocity(strategy1, 0.5, dt=0.01, max_time=4000, max_iterations=10000)


# simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=4.5, n_candidates=1000, optimization=False, events=2)

all_results = []

l1 = []
s1 = []
l2 = []
s2 = []

# for optimization in range(10):
#     print("Iteration #: ", optimization)

#     if optimization == 0:
#         filename_lap1 = "top_strategies_lap1.csv"
#         filename_lap2 = "top_strategies_lap2.csv"
#     else:
#         filename_lap1 = "top_strategies_optimized_lap1.csv"
#         filename_lap2 = "top_strategies_optimized_lap2.csv"

#     race_time, race_energy, [], [], current_velocity, best_strategy_lap1, best_strategy_lap2, lap1energy, lap2energy, avg_length1, avg_length2, avg_start1, avg_start2 = simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=0.5, n_candidates=250, optimization=True, events=2)

#     iteration_result = {
#         "Optimization Iteration Number": optimization,
#         "Race Time" : race_time,
#         "Race Energy" : race_energy,
#         "Lap 1 Energy" : lap1energy,
#         "Lap 2 Energy" : lap2energy,
#     }

#     all_results.append(iteration_result)

# summary_rows = []
# for result in all_results:
#     summary_rows.append({
#         "Iteration": result["Optimization Iteration Number"],
#         "Race Time": result["Race Time"],
#         "Race Energy": result["Race Energy"],
#         "Lap 1 Energy" : result["Lap 1 Energy"],
#         "Lap 2 Energy" : result["Lap 2 Energy"]
#     })

# df_summary = pd.DataFrame(summary_rows)
# df_summary.to_csv("optimization_summary.csv", index=False)
# print("Optimization Saved to CSV")

# strategy2 = [(13.157193936688936, 122.9628765550031, 2), (361.7845467621548, 58.402130979868545, 2)]

strategy1 = [(0, 86.94, 2), (121.56, 99.87, 1.8051335162084188)]

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

strategy_lap1 = [(0.0, 60.45040260615454, 1.727925605506318), (218.64831636879333, 19.33211197811235, 2.37557712961573)]
strategy_lap2 = []

simulate_lap_with_initial_velocity(strategy_lap1, 4.5, dt=0.01, max_time=400000, max_iterations=100000)

plot_strategy_overlay(x_coordinates, y_coordinates, strategy_lap1, strategy_lap2, title="Optimized Strategy")

def main_menu():
    print("\n===== Green Bath Racing Simulation Tool =====")
    
    print("Track Selection")
    print("1. Silesia Ring")
    print("2. Nogaro")

    track_selection = input("Select a Track (1-2): ")

    track_files = {
        "1" : "CircuitSilesiaRingData.csv",
        "2" : "CircuitNogaroElevationData.txt"
    }
    
    print("\n Select a Function:")
    
    print("1. Simulate a Lap")
    print("2. Simulate a Race")
    print("3. Optimize Strategies")
    print("4. Exit")

    choice = input("Select an Option (1-4): ")
    
    if choice == "1":
        print("Simulating Lap")

        confirmed = False
        while not confirmed:
            print("\nInput the Strategy:")
            num_events = int(input("Number of Acceleration Events: "))
            strategy = []
            
            for event in range(num_events):
                print(f"Event #{event + 1}")
                start = float(input("  Select Starting Position (m): "))
                length = float(input("  Select Length (m): "))
                accel = 2
                strategy.append((start, length, accel))

            print("\nThe Strategy Selected Is:")
            for i, strat in enumerate(strategy):
                print(f"  Acceleration #{i + 1}: Start={strat[0]}, Length={strat[1]}, Accel={strat[2]}")

            confirmation = input("\nWould you like to proceed with this strategy? (Y/N): ").strip().upper()
            if confirmation == "Y":
                confirmed = True
            else:
                print("Let's try again.")
                
        time_elapsed, energy_consumed, current_distance, velocity_profile, distance_profile, time_profile, resistive_force_profile, power_profile, motor_status, finished, lat_acc, lon_acc = simulate_lap_with_initial_velocity(strategy, 0.5, dt=0.01, max_time=4000, max_iterations=10000)

        print("\nLap Summary:")
        print("")
        print("Lap Completion: ", finished)
        print("Distance Driven: ", current_distance)
        print("Lap Time: ", time_elapsed, " s - ", math.ceil(time_elapsed/60), " mins")
        print("Avg. Speed (km/h): ", (current_distance/time_elapsed)*3.6)
        print("Efficiency (km/kWh): ", (total_dist/1000) / (energy_consumed/3600000))

        acceleration_events = strategy

        fig, axs = plt.subplots(4, 1, figsize=(12,10), sharex=True)

        axs[0].plot(distance_profile, velocity_profile, label="Velocity (m/s)", color="blue")
        axs[0].set_ylabel("Velocity (m/s)")
        axs[0].set_title("Silesia Ring Simulated Lap Telemetry")

        axs[1].plot(distance_profile, power_profile, label="Power Output (W)", color="blue")
        axs[1].set_ylabel("Power Output (W)")

        axs[2].plot(distance_profile, motor_status, label = "Motor Status (On/Off)", color="blue")
        axs[2].set_ylabel("Motor Status")
        axs[2].set_yticks([0,1])

        axs[3].plot(distance_profile, energy_consumed, label="Energy (J)", color="blue")
        axs[3].set_ylabel("Energy Consumption (J)")

        plt.tight_layout()
        plt.show()
    
    elif choice == "2":
        num_events = int(input("Select Number of Acceleration Events: "))
        candidates = int(input("Select Number of Candidate Strategies: "))

        simulate_race(dt=0.1, max_time=4000, max_iterations=10000, initial_velocity=0.5, n_candidates=candidates, optimization=False, events=num_events)

        results_lap1 = pd.read_csv("top_strategies_lap1.csv")
        results_lap2 = pd.read_csv("top_strategies_lap2.csv")

        strategy_results = []
        strategy_results.append(results_lap1["Strategy"][0])
        strategy_results.append(results_lap2["Strategy"][0])

        # converted = [(round(start, 2), round(length, 2)) for start, length, _ in strategy_results]

        lap1_time = results_lap1["Lap Time"][0]
        lap2_time = results_lap2["Lap Time"][0]
        lap1_energy = results_lap1["Energy Consumption"][0]
        lap2_energy = results_lap2["Energy Consumption"][0]

        num_laps = math.floor(16000 / total_dist)
        efficiency = ((total_dist * num_laps)/1000) / (((lap1_energy + (num_laps - 1)*lap2_energy))/3600000)

        total_racetime = lap1_time + (num_laps-1)*lap2_time

        print("Race Efficiency (km/kWh): ", efficiency)
        print("Race Avg. Speed (km/h): ", ((num_laps*total_dist)/total_racetime))*3.6
        print("Total Time: ", round(total_racetime/60, 2))

        print("\nLap 1 Results:")
        print("Lap Time (s):")
        # print("Lap 1 Strategy: ", converted[0])

data = pd.read_csv("telemetry_output.csv")

velocity = data["Velocity_kph"]
time = data["Time"]
distance = data["Distance_m"]


plt.plot(distance, velocity)
plt.show()