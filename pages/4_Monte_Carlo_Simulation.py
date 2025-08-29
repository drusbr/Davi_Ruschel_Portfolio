import streamlit as st
import pandas as pd
import numpy as np 
import fastf1 as ff1
import matplotlib.pyplot as plt
import itertools
import altair as alt
import seaborn as sns
import random

OVERTAKE = {
    "attempt_window_s": 1.0,
    "follow_loss_s": 0.3,
    "pass_exec_bonus_s": 0.12,
    "alpha0": -2.0,
    "alpha1": 1.6,
    "alpha_drs": 0.6,
    "track_difficulty": -1.5
}

PIT_LOSS_GREEN = (19.5, 1.3)
SC_PIT_FACTOR = 0.55
EPS_PASS = 1e-3

SC_STABILISATION_MIN, SC_STABILISATION_MAX = 3, 8

def get_race_stats(race, driver):
    df = pd.read_csv("big_data1.csv")
    df_race = df[df["Race"]==race]
    df_driver = df_race[df_race["Driver"]==driver]
    if df_driver.empty:
        avg_pace, avg_std = 0, 0
        return avg_pace, avg_std

    avg_pace = float(df_driver["MeanLap_s"].iloc[0])
    avg_std = float(df_driver["StdLap_s"].iloc[0])

    return avg_pace, avg_std

def load_session(race):
    session = ff1.get_session(2025, race, "R")
    session.load()
    drivers_abbv = sorted(session.results["Abbreviation"].dropna().unique().tolist())
    race_laps = int(session.results["Laps"].dropna().max())

    return drivers_abbv, race_laps

def deg_scale_from_std_logistic(avg_pace, avg_std, mid=0.015, k=45.0, out=(0.9, 1.15)):
    # x = variability ratio; mid = “neutral” point; k controls steepness
    x = avg_std / max(1e-9, float(avg_pace))
    lo, hi = out
    s = 1.0 / (1.0 + np.exp(-k * (x - mid)))   # 0..1
    return lo + s * (hi - lo)

def precompute_stints(
    laps,
    avg_pace,
    avg_std,
    rng=None,
    tire_perf=None,
    base_targets=None,
    fuel_per_lap_s=0.035,           # crude fuel time/ lap (s), strategy-relevant
    deg_mid=0.015, deg_k=45.0, deg_bounds=(0.9, 1.15),
):
    """
    Returns: totals[compound][L] = cumulative time of a stint of length L (0..laps)
    Includes base pace, tyre degradation, and a simple fuel term.
    Jitters target wear parameters per driver to avoid identical argmins.
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    if tire_perf is None:
        # multiplicative pace factors (soft fastest)
        tire_perf = {"SOFT": 0.997, "MEDIUM": 1.000, "HARD": 1.006}

    if base_targets is None:
        # A: nominal usable laps; D: total deg over A laps (s);
        # S: initial slope (s/lap) at age=0; used to derive (r0, c)
        base_targets = {
            "SOFT":   {"A": 18, "D": 0.90, "S": 0.070},
            "MEDIUM": {"A": 22, "D": 0.60, "S": 0.045},
            "HARD":   {"A": 28, "D": 0.35, "S": 0.025},
        }

    def calibrate_quadratic(A, D, S):
        # c = 2(SA - D)/A^2 ; r0 = S - cA
        c = 2.0 * (S * A - D) / (A * A)
        r0 = S - c * A
        return r0, c

    L = laps
    ell = np.arange(L + 1, dtype=np.float64)        # stint length 0..L
    sum_age  = ell * (ell - 1) / 2.0                # Σ age
    sum_age2 = (ell - 1) * ell * (2*ell - 1) / 6.0  # Σ age^2

    # per-driver degradation *scale* from dispersion
    deg_scale = deg_scale_from_std_logistic(avg_pace, avg_std,
                                            mid=deg_mid, k=deg_k, out=deg_bounds)

    # per-driver jitter to A/D/S (changes curve shape → different optimum)
    targets = {}
    for comp, t in base_targets.items():
        A = max(10, int(round(t["A"] * rng.normal(1.0, 0.06))))          # ±6%
        D = max(0.1, float(t["D"] * rng.normal(1.0, 0.08)))              # ±8%
        S = max(0.005, float(t["S"] * rng.normal(1.0, 0.08)))            # ±8%
        targets[comp] = {"A": A, "D": D, "S": S}

    # Simple fuel term: more laps in a stint → more heavy-fuel time
    fuel_term = fuel_per_lap_s * ell

    totals = {}
    for comp, tf in tire_perf.items():
        r0, c = calibrate_quadratic(**targets[comp])
        degr_sum = deg_scale * (r0 * sum_age + 0.5 * c * sum_age2)
        base_mean = (avg_pace * tf) * ell
        total_time = base_mean + degr_sum + fuel_term
        total_time[0] = 0.0
        totals[comp] = total_time

    return totals


def optimize_strategy(
    laps,
    driver,
    race,
    pit_stop_penalty=20.0,
    min_stint=5,
    allow_lap1_pit=False,
    enforce_two_compounds=True,
    soft_max=None, med_max=None, hard_max=None,
    rng=None,
):
    """
    Returns:
      best_stops:   tuple (i,) or (i,j) nominal stop laps (1-indexed in race loop usage)
      best_strategy: tuple of compounds e.g. ('SOFT','MEDIUM') or ('SOFT','MEDIUM','HARD')
      best_time:    float
      windows:      {'stop1': (lo,hi) or None, 'stop2': (lo,hi) or None}
    """
    import itertools
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    avg_pace, avg_std = get_race_stats(race, driver)

    # driver variability:
    pit_pen_drv = float(pit_stop_penalty * rng.normal(1.0, 0.06))     # ±6% pit loss
    avg_std_eff = float(max(1e-6, avg_std * rng.normal(1.0, 0.07)))   # ±7% spreads

    totals = precompute_stints(laps, avg_pace, avg_std_eff, rng=rng)

    tires = ["SOFT", "MEDIUM", "HARD"]
    one_stop_strats = [(a, b) for a, b in itertools.permutations(tires, 2)]
    two_stop_strats = [
        s for s in itertools.product(tires, repeat=3)
        if (not enforce_two_compounds) or (len(set(s)) >= 2)
    ]

    def stint_ok(tire, length):
        if length < min_stint: return False
        if soft_max and tire == "SOFT" and length > soft_max:   return False
        if med_max  and tire == "MEDIUM" and length > med_max:  return False
        if hard_max and tire == "HARD" and length > hard_max:   return False
        return True

    i_start = 1 if allow_lap1_pit else max(2, min_stint)

    best_time = float("inf")
    best_strategy, best_stops = None, None

    # ----- 1 stop -----
    for (t1, t2) in one_stop_strats:
        for i in range(i_start, laps):
            len1 = i
            len2 = laps - i
            if not (stint_ok(t1, len1) and stint_ok(t2, len2)):
                continue
            race_time = totals[t1][len1] + totals[t2][len2] + pit_pen_drv
            if race_time < best_time:
                best_time = race_time
                best_strategy = (t1, t2)
                best_stops = (i,)

    # ----- 2 stop -----
    for (t1, t2, t3) in two_stop_strats:
        for i in range(i_start, laps - 1):
            for j in range(i + 1, laps):
                len1 = i
                len2 = j - i
                len3 = laps - j
                if not (stint_ok(t1, len1) and stint_ok(t2, len2) and stint_ok(t3, len3)):
                    continue
                race_time = totals[t1][len1] + totals[t2][len2] + totals[t3][len3] + 2 * pit_pen_drv
                if race_time < best_time:
                    best_time = race_time
                    best_strategy = (t1, t2, t3)
                    best_stops = (i, j)

    # Safety: if nothing found (shouldn't happen), fall back
    if best_stops is None:
        return (None, None, float("inf"), {"stop1": None, "stop2": None})

    # ----- driver-specific nominal jitter & windows -----
    def clamp(x, lo, hi): return int(max(lo, min(hi, int(x))))
    def make_window(center, half=5, lo=1, hi=None):
        if center is None: return None
        if hi is None: hi = laps - 1
        a = clamp(center - half, lo, hi)
        b = clamp(center + half, lo, hi)
        if b < a: a, b = b, a
        return (a, b)

    if len(best_stops) == 1:
        i_nom = clamp(best_stops[0] + rng.integers(-2, 3), 1, laps - 2)
        windows = {"stop1": make_window(i_nom, half=5, lo=1, hi=laps-1), "stop2": None}
        best_stops = (i_nom,)
    else:
        i_nom = clamp(best_stops[0] + rng.integers(-2, 3), 1, laps - 2)
        j_nom = clamp(best_stops[1] + rng.integers(-2, 3), i_nom + 1, laps - 1)
        windows = {
            "stop1": make_window(i_nom, half=5, lo=1, hi=j_nom - 1),
            "stop2": make_window(j_nom, half=5, lo=i_nom + 1, hi=laps - 1)
        }
        best_stops = (i_nom, j_nom)

    # debug print (optional)
    print(f"Results /  {driver}")
    print("Best Time: ", best_time)
    print("Best Strategy: ", best_strategy)
    print("Stop Laps: ", best_stops)

    return best_stops, best_strategy, best_time, windows

def overtake_attempt(driver, race_order, avg_pace, avg_std, lap, cumulative_laptimes, driver_position, rng):
    success = False
    driver_ahead = race_order[driver_position - 1]
    if lap > 0 and cumulative_laptimes[driver] and cumulative_laptimes[driver_ahead]:
        pace_ahead = avg_pace[driver_ahead]
        pace_driver = avg_pace[driver]
        relative_pace = pace_ahead - pace_driver
        gap_ahead = cumulative_laptimes[driver][-1] - cumulative_laptimes[driver_ahead][-1]
        if gap_ahead <= OVERTAKE["attempt_window_s"]:
            drs = 1
            x = OVERTAKE["alpha0"] + OVERTAKE["alpha1"] * relative_pace + OVERTAKE["alpha_drs"] * drs + OVERTAKE["track_difficulty"]
            prob_overtake = 1.0 / (1.0 + np.exp(-x))
            if rng.random() < prob_overtake:
                success = True
            return success
        else:
            success = False
            return success
    else:
        return success
    
def decision_making(driver, avg_pace, avg_std, race_order, cumulative_laptimes, lap, rng,
                    individual_laptimes, stop1_lap, stop2_lap, strategy, laps,
                    stint_laps_counter, current_tires, sim, racecontrol_log, decision_log):


    # --- early outs / indexes ---
    if driver not in race_order:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    pos = race_order.index(driver)
    if pos == 0:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    driver_ahead = race_order[pos - 1]
    if len(cumulative_laptimes[driver]) == 0 or len(cumulative_laptimes[driver_ahead]) == 0:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    gap_ahead = cumulative_laptimes[driver][-1] - cumulative_laptimes[driver_ahead][-1]

    # === 1) only act if you're actually close (fix sign with abs) ===
    if gap_ahead is None or abs(gap_ahead) > 2.5:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    if lap <= 5:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    # relative pace (negative means you're faster)
    pace_driver = np.mean(individual_laptimes[driver][-5:])
    pace_ahead  = np.mean(individual_laptimes[driver_ahead][-5:])
    relative_pace = pace_driver - pace_ahead
    if relative_pace >= -0.10:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    # windows
    w1 = range(max(1, (stop1_lap.get(driver) or 0) - 5))
    w2 = range(max(1, (stop2_lap.get(driver) or 0) - 5))
    in_w1, in_w2 = (lap in w1), (lap in w2)
    if not (in_w1 or in_w2):
        decision = {"action": "STAY", "new_tyre": None}
        return decision

    # === 2) tyre model: small additive compound deltas ===
    compound_add = {"SOFT": 0.00, "MEDIUM": 0.15, "HARD": 0.35}  # seconds/lap baseline
    targets = {
        "SOFT":   {"A": 18, "D": 0.90, "S": 0.070},
        "MEDIUM": {"A": 22, "D": 0.60, "S": 0.045},
        "HARD":   {"A": 28, "D": 0.35, "S": 0.025},
    }
    def calibrate_quadratic(A, D, S):
        c  = 2.0 * (S*A - D) / (A*A)
        r0 = S - c*A
        return r0, c
    TYRE_DEG = {c: calibrate_quadratic(**p) for c, p in targets.items()}

    def tyre_deg_seconds(comp, age, cliff_start_frac=None, gamma=0.0):
        age = max(0.0, float(age))
        r0, c = TYRE_DEG[comp]
        base = r0*age + 0.5*c*age*age
        if cliff_start_frac is not None and gamma > 0.0:
            A  = targets[comp]["A"]
            Ac = cliff_start_frac * A
            if age > Ac:
                base += gamma * (age - Ac)**3
        return base

    # pick current/new tyre according to which window we're in
    strat = strategy[driver]
    if in_w1:
        current_tire, new_tire = strat[0], strat[1]
        age_now = stint_laps_counter[driver]
        planned_stop = stop1_lap.get(driver)
        window_tag = "W1"
    else:
        current_tire, new_tire = strat[1], strat[2]
        age_now = stint_laps_counter[driver]
        planned_stop = stop2_lap.get(driver)
        window_tag = "W2"

    # === 3) horizon and delta definition (include pit loss on pit-now branch) ===
    pit_loss = 20.0  # seconds
    # horizon: at least 5, up to planned stop, capped at 10
    if planned_stop is None:
        H = 5
    else:
        H = max(5, min(10, planned_stop - lap))
    if H <= 0:
        H = 5

    # rejoin time for position check only (not in time_gain)
    lap_idx = len(cumulative_laptimes[driver]) - 1
    new_time = cumulative_laptimes[driver][lap_idx] + pit_loss

    others = [(cumulative_laptimes[d][lap_idx], d) for d in race_order if d != driver and len(cumulative_laptimes[d]) > lap_idx]
    others.sort(key=lambda x: x[0])
    rejoin_pos = 1 + sum(t <= new_time for t, _ in others)
    ahead_pairs = [(t, d) for (t, d) in others if t <= new_time]
    if ahead_pairs:
        time_ahead, driver_ahead_rejoin = ahead_pairs[-1]
        gap_ahead_rejoin = new_time - time_ahead
    else:
        driver_ahead_rejoin = None
        gap_ahead_rejoin = None

    # (optionally) keep your deg_scale, but default to 1.0 if it misbehaves
    try:
        deg_scale = deg_scale_from_std_logistic(avg_pace[driver], avg_std[driver], mid=0.015, k=45.0, out=(0.9, 1.15))
    except Exception:
        deg_scale = 1.0

    def model_lap(base_driver, comp, age, fuel_kg):
        fuel = 0.035 * fuel_kg
        return base_driver + compound_add[comp] + tyre_deg_seconds(comp, age, 0.8, 3e-4)*deg_scale + fuel

    # base pace (driver ability) – use your precomputed avg_pace
    base_me = float(avg_pace[driver])

    # === simulate H laps: stay vs pit-now ===
    sum_stay = 0.0
    sum_pit  = pit_loss  # pit now cost up-front

    for k in range(H):
        # simple fuel model – same for both branches; cancels, but keep for clarity
        fuel_kg = max(0.0, 110 - 100/laps*(lap + k))

        # stay: same compound, aging
        sum_stay += model_lap(base_me, current_tire, age_now + k, fuel_kg)

        # pit-now: new compound, aging from zero
        sum_pit  += model_lap(base_me, new_tire, k, fuel_kg)

    time_gain = sum_stay - sum_pit  # >0 means undercut is worth it

    # basic traffic effect for the next lap window after rejoin
    if driver_ahead_rejoin:
        base_a = float(avg_pace[driver_ahead_rejoin])
        # estimate ahead’s current tyre/age
        if current_tires[driver_ahead_rejoin]:
            tyre_ahead = current_tires[driver_ahead_rejoin][-1]
            age_ahead  = stint_laps_counter[driver_ahead_rejoin]
        else:
            tyre_ahead = strategy[driver_ahead_rejoin][0]
            age_ahead  = 0
        # crude “follow loss” model if you rejoin within 1.0 s and fail to pass
        ahead_lap = model_lap(base_a, tyre_ahead, age_ahead, fuel_kg)
        my_lap    = model_lap(base_me, new_tire, 0, fuel_kg)
        # recompute p(pass) if you like; for now, conservative penalty:
        if gap_ahead_rejoin is not None and gap_ahead_rejoin <= 1.0:
            follow_loss_s = OVERTAKE["follow_loss_s"]
            exec_bonus    = OVERTAKE["pass_exec_bonus_s"]
            # simple logit pass model (your original)
            drs = 1
            x = (OVERTAKE["alpha0"]
                 + OVERTAKE["alpha1"] * (ahead_lap - my_lap)
                 + OVERTAKE["alpha_drs"] * drs
                 + OVERTAKE["track_difficulty"])
            p_pass = 1.0 / (1.0 + np.exp(-x))
            if rng.random() < p_pass:
                time_gain += exec_bonus
            else:
                time_gain -= (my_lap - ahead_lap) + follow_loss_s

    # logging
    # if in_w1 or in_w2:
    #     print(f"[DM] Lap {lap} {driver}: gap={gap_ahead:.2f}s rel_pace={relative_pace:.3f}s, "
    #           f"time_gain={time_gain:.2f}s window={window_tag}")

    # === decision ===
    if time_gain < -1:  # threshold
        decision = {
            "action": "PIT",
            "new_tyre": new_tire,
            "projected_gain_s": time_gain,
            "rejoin_rank": rejoin_pos
        }
        racecontrol_log.append({
            "Sim": sim, "Lap": lap, "Driver": driver, "Event": "Decision",
            "Message": f"Early PitStop for {new_tire} / Gain {time_gain:.2f}s / "
                       f"Rejoin P{rejoin_pos}, Behind {driver_ahead_rejoin} by {gap_ahead_rejoin}"
        })
        decision_log.append({
            "Sim": sim, "Lap": lap, "Driver": driver, "Action": "PIT",
            "CurrentTire": current_tires[driver][-1] if current_tires[driver] else current_tire,
            "ExpectedTimeGain": round(time_gain, 3),
            "RejoinPosition": rejoin_pos,
            "DriverAheadRejoin": driver_ahead_rejoin,
            "GapAheadRejoin": gap_ahead_rejoin
        })
    else:
        decision = {"action": "STAY", "new_tyre": None}
        return decision

# def monte_carlo_safety_car(laps, drivers, race, sims=10000, seed=42):
#     load_session(race)
#     if isinstance(drivers, str):
#         drivers = [drivers]

#     rng = np.random.default_rng(seed)
#     p_SC = 0.30

#     # Precompute strategy + base pace for each driver (same for all sims)
#     avg_pace = {}
#     avg_std = {}
#     stop1_lap = {}
#     stop2_lap = {}
#     strategy = {}

#     for driver in drivers:
#         avg_pace[driver], avg_std[driver] = get_race_stats(race, driver)
#         best_stops, best_strategy, best_time = optimize_strategy(
#             laps, driver, race,
#             pit_stop_penalty=20.0, min_stint=5, allow_lap1_pit=False,
#             enforce_two_compounds=True, soft_max=None, med_max=None, hard_max=None
#         )
#         print(best_stops)
#         strategy[driver] = best_strategy
#         if len(best_stops) == 1:
#             stop1_lap[driver] = best_stops[0]
#             stop2_lap[driver] = None
#         else:
#             stop1_lap[driver] = best_stops[0]
#             stop2_lap[driver] = best_stops[1]

#     tire_perf = {"SOFT": 0.975, "MEDIUM": 1.0, "HARD": 1.04}
#     init_fuel = 110
#     fuel_cons = init_fuel / laps

#     # Starting Position Determine
#     qualifying_results = {}
#     for driver in drivers:
#         qualifying_results[driver] = np.random.normal(loc=avg_pace[driver], scale=avg_std[driver])
#     race_order = [driver for driver, _ in sorted(qualifying_results.items(), key=lambda x: x[1])]

#     all_lap_data = []      # per-lap data across sims
#     all_race_summaries = []  # per-race results
#     racecontrol_log = []
#     decision_log = []
#     driver_gaps = []

#     for sim in range(sims):
#         sc_happens = rng.random() < p_SC
#         sc_lap = rng.integers(1, laps + 1) if sc_happens else None
#         if sc_lap is not None:
#             sc_stabilisation_laps = random.randint(SC_STABILISATION_MIN, SC_STABILISATION_MAX)
#             stabilisation_window = list(range(sc_lap, sc_lap + sc_stabilisation_laps))
#         else:
#             sc_stabilisation_laps = 0
#             stabilisation_window = []

#         individual_laptimes = {d: [0.0] for d in drivers}
#         cumulative_laptimes = {d: [0.0] for d in drivers}
#         lap_gaps = {d: [] for d in drivers}
#         stint_laps_counter = {d: 0 for d in drivers}
#         current_tires = {d: [] for d in drivers}
#         order_prev = race_order[:]

#         def calibrate_quadratic(A, D, S):
#             # c = 2(SA - D)/A^2, r0 = S - cA
#             c = 2.0 * (S*A - D) / (A*A)
#             r0 = S - c*A
#             return r0, c

#         tire_perf = {
#             "SOFT": 0.975,
#             "MEDIUM": 1.0,
#             "HARD": 1.04
#         }

#         # Recommended targets
#         targets = {
#             "SOFT": {"A": 18, "D": 0.90, "S": 0.070},
#             "MEDIUM": {"A": 22, "D": 0.60, "S": 0.045},
#             "HARD": {"A": 28, "D": 0.35, "S": 0.025},
#         }

#         # Build coefficient dicts (r0, c) per compound
#         TYRE_DEG = {}
#         for comp, t in targets.items():
#             TYRE_DEG[comp] = calibrate_quadratic(t["A"], t["D"], t["S"])

#         def tyre_deg_seconds(compound: str, age_laps: float, cliff_start_frac: float = None, gamma: float = 0.0) -> float:
#             """
#             Returns additive laptime (s) from tyre wear at given tyre age.
#             - compound: "SOFT"|"MEDIUM"|"HARD"
#             - age_laps: laps on the current tyre (can be float if you interpolate)
#             - cliff_start_frac: if not None, fraction of the compound's A where a gentle cubic 'cliff' starts
#             - gamma: cubic intensity for the cliff (e.g., 1e-4 to 5e-4)
#             """
#             age = max(0.0, float(age_laps))
#             r0, c = TYRE_DEG[compound]
#             base = deg_scale * (r0 * age + 0.5 * c * age * age)
#             if cliff_start_frac is not None and gamma > 0.0:
#                 A = targets[compound]["A"]
#                 Ac = cliff_start_frac * A
#                 if age > Ac:
#                     base += gamma * (age - Ac)**3
#             return base

#         overtake_attemps = {d: 0 for d in drivers}
#         overtake_success = {d: 0 for d in drivers}

#         # ---- main lap loop ----
#         for lap in range(laps):
#             # snapshot order at start-of-lap for consistent in-lap decisions
#             order_snapshot = race_order[:]
#             pitted_this_lap = set()

#             for driver in drivers:
#                 # Early-only windows (exclude the scheduled stop lap itself)
#                 w1 = range(max(1, stop1_lap[driver]-5), stop1_lap[driver]) if stop1_lap[driver] else range(0)
#                 w2 = range(max(1, stop2_lap[driver]-5), stop2_lap[driver]) if stop2_lap[driver] else range(0)

#                 # Planned strategy tyres for this driver
#                 if len(strategy[driver]) == 2:
#                     first_tire, second_tire = strategy[driver][0], strategy[driver][1]
#                     third_tire = None
#                 else:
#                     first_tire, second_tire, third_tire = strategy[driver]

#                 # Determine the planned (intended) compound for this lap from scheduled stops
#                 if stop1_lap[driver] is None or lap < stop1_lap[driver]:
#                     intended = first_tire
#                 elif (stop2_lap[driver] is None) or (lap < stop2_lap[driver]):
#                     intended = second_tire
#                 else:
#                     intended = third_tire  # may be None for a 1-stop plan

#                 # --- Decision model (may choose to PIT earlier/later) ---
#                 # (Before tyre-state update so we know if we will pit now)
#                 decision = decision_making(
#                     driver, avg_pace, avg_std, race_order, cumulative_laptimes, lap, rng,
#                     individual_laptimes, stop1_lap, stop2_lap, strategy, laps,
#                     stint_laps_counter, current_tires, sim, racecontrol_log, decision_log
#                 )

#                 did_decision_pit = (decision["action"] == "PIT")

#                 # If early pit, consume the corresponding scheduled stop and mark as pitted
#                 if did_decision_pit:
#                     in_w1 = (stop1_lap[driver] is not None) and (lap in w1)
#                     in_w2 = (stop2_lap[driver] is not None) and (lap in w2)
#                     if in_w1 and lap < stop1_lap[driver]:
#                         stop1_lap[driver] = lap
#                     elif in_w2 and lap < stop2_lap[driver]:
#                         stop2_lap[driver] = lap
#                     pitted_this_lap.add(driver)

#                 is_sched_pit = ((stop1_lap[driver] is not None and lap == stop1_lap[driver]) or
#                                 (stop2_lap[driver] is not None and lap == stop2_lap[driver]))
#                 is_pitstop = did_decision_pit or is_sched_pit

#                 # The compound we will run AFTER any pit this lap
#                 new_compound = decision["new_tyre"] if did_decision_pit else intended

#                 # --- Update tyre state ONCE per lap (no duplicates) ---
#                 if lap == 0:
#                     current_tires[driver].append(new_compound if is_pitstop else intended)
#                     stint_laps_counter[driver] = 0
#                 elif is_pitstop:
#                     current_tires[driver].append(new_compound)
#                     stint_laps_counter[driver] = 0
#                 else:
#                     # stay on same tyre, age increases
#                     current_tires[driver].append(current_tires[driver][-1])
#                     stint_laps_counter[driver] += 1

#                 # Pace factors for current tyre
#                 tire_factors = {"SOFT": 0.997, "MEDIUM": 1.0, "HARD": 1.006}
#                 laptime_factor = tire_factors[current_tires[driver][-1]]

#                 # Fuel & degradation
#                 fuel_onboard = init_fuel - fuel_cons * lap
#                 fuel_effect = fuel_onboard * 0.035

#                 # Deg scale must be per-driver
#                 deg_scale = deg_scale_from_std_logistic(
#                     avg_pace[driver], avg_std[driver], mid=0.015, k=45.0, out=(0.9, 1.15)
#                 )
#                 tire_deg = tyre_deg_seconds(
#                     current_tires[driver][-1], stint_laps_counter[driver],
#                     cliff_start_frac=0.8, gamma=3e-4
#                 )

#                 # --- Base lap time for EVERYONE (leaders included) ---
#                 lap_time = np.random.normal(
#                     loc=avg_pace[driver] * laptime_factor,
#                     scale=avg_std[driver]
#                 ) + tire_deg + fuel_effect

#                 # --- Safety Car / green-flag pit loss handling ---
#                 if lap in stabilisation_window:
#                     if is_pitstop:
#                         # cheaper pit under SC
#                         pit_loss = np.random.normal(PIT_LOSS_GREEN[0], PIT_LOSS_GREEN[1]) * SC_PIT_FACTOR
#                         lap_time += pit_loss
#                     else:
#                         # running under SC
#                         lap_time = avg_pace[driver] * 0.4
#                 else:
#                     # green flag pit → add full pit loss
#                     if is_pitstop:
#                         pit_loss = np.random.normal(PIT_LOSS_GREEN[0], PIT_LOSS_GREEN[1])
#                         lap_time += pit_loss

#                 # --- Traffic / overtake logic (only if not leader and not pitting this lap) ---
#                 # Use snapshot for consistent in-lap decisions; don't mutate race_order here.
#                 driver_position_snapshot = order_snapshot.index(driver)
#                 if driver_position_snapshot > 0 and not is_pitstop:
#                     driver_ahead_snapshot = order_snapshot[driver_position_snapshot - 1]
#                     success = overtake_attempt(
#                         driver, order_snapshot, avg_pace, avg_std, lap,
#                         cumulative_laptimes, driver_position_snapshot, rng
#                     )
#                     overtake_attemps[driver] += 1
#                     if success:
#                         overtake_success[driver] += 1
#                         lap_time -= OVERTAKE["pass_exec_bonus_s"]
#                     else:
#                         lap_time += OVERTAKE["follow_loss_s"]

#                 # --- Log this lap ---
#                 individual_laptimes[driver].append(lap_time)
#                 cumulative_laptimes[driver].append(
#                     lap_time if lap == 0 else cumulative_laptimes[driver][-1] + lap_time
#                 )

#             # Re-sort race order every lap (reflects pit losses and pass bonuses)
#             race_order = sorted(drivers, key=lambda d: cumulative_laptimes[d][-1])
#             times_this_lap = {d: cumulative_laptimes[d][-1] for d in drivers}
#             order = sorted(times_this_lap, key=times_this_lap.get)
#             leader = order[0]
#             leader_time = times_this_lap[leader]

#             for idx, d in enumerate(order, start=1):
#                 gap_to_leader = times_this_lap[d] - leader_time
#                 interval = 0.0 if idx == 1 else times_this_lap[d] - times_this_lap[order[idx-2]]
#                 all_lap_data.append({
#                     "Sim": sim+1,
#                     "Lap": lap+1,
#                     "Position": idx,
#                     "Driver": d,
#                     "LapTime": individual_laptimes[d][-1],
#                     "CumulativeTime": times_this_lap[d],
#                     "GapToLeader": gap_to_leader,
#                     "IntervalAhead": interval,
#                     "Leader": leader,
#                     "Compound": current_tires[d][-1],
#                     "SC_Happens": sc_happens,
#                     "SC_Lap": sc_lap
#                 })

#         # Store race summary for this sim
#         final_times = {d: cumulative_laptimes[d][-1] for d in drivers}
#         order = sorted(final_times, key=final_times.get)
#         all_race_summaries.append({
#             "Sim": sim+1,
#             "Winner": order[0],
#             "SC_Happens": sc_happens,
#             "SC_Lap": sc_lap,
#             **final_times
#         })

#     # Convert to DataFrames
#     df_laps = pd.DataFrame(all_lap_data)
#     df_summary = pd.DataFrame(all_race_summaries)
#     df_racecontrol = pd.DataFrame(racecontrol_log)
#     df_decision = pd.DataFrame(decision_log)

#     # Save if needed
#     df_laps.to_csv("monte_carlo_laps.csv", index=False)
#     df_summary.to_csv("monte_carlo_summary.csv", index=False)
#     df_racecontrol.to_csv("race_control.csv", index=False)
#     df_decision.to_csv("decision_making.csv", index=False)

#     return df_laps, df_summary


def monte_carlo_safety_car(laps, drivers, race, sims=1000, seed=42):
    # Load the Session
    load_session(race)

    if isinstance(drivers, str):
        drivers = [drivers]
    
    # Safety Car Probability Setup
    rng = np.random.default_rng(seed)
    p_SC = 0.35

    # --------- Pre-Compute Per-Driver Base Stats & Nominal Strategies
    avg_pace, avg_std = {}, {}
    stop1_lap_nom, stop2_lap_nom = {}, {}
    strategy_nom = {}

    for driver in drivers:
        avg_pace[driver], avg_std[driver] = get_race_stats(race, driver)
        best_stops, best_strategy, _best_time, _windows = optimize_strategy(
            laps, driver, race, pit_stop_penalty=20.0, min_stint=5, allow_lap1_pit=True, 
            enforce_two_compounds=True, soft_max=None, med_max=None, hard_max=None, rng=rng
        )

        strategy_nom[driver] = list(best_strategy) if isinstance(best_strategy, (tuple, list)) else [best_strategy]
        if len(best_stops) == 1:
            stop1_lap_nom[driver] = int(best_stops[0])
            stop2_lap_nom[driver] = None
        else:
            stop1_lap_nom[driver] = int(best_stops[0])
            stop2_lap_nom[driver] = int(best_stops[1])
        
    def pit_window(center_1b, lo=1, hi=None, half_width=5):
        if center_1b is None:
            return range(0)
        if hi is None:
            hi = laps - 1
        a = max(lo, int(center_1b) - half_width)
        b = min(hi, int(center_1b) + half_width)
        return range(a, b)
    
    all_lap_rows = []
    all_summaries = []
    racecontrol_log = []
    decision_log = []
    driver_gaps = []

    for sim in range(1, sims + 1):
        # Per-Sim Deep Copies
        s1 = {d: (None if stop1_lap_nom[d] is None else int(stop1_lap_nom[d])) for d in drivers}
        s2 = {d: (None if stop2_lap_nom[d] is None else int(stop2_lap_nom[d])) for d in drivers}
        strat = {d: (strategy_nom[d][:]) for d in drivers}

        # Optional Per-Driver Planned Stop Jitter (+/- 1 Lap)
        for driver in drivers:
            if s1[driver] is not None:
                s1[driver] = int(np.clip(s1[driver] + rng.integers(-1, 2), 1, max(1, laps-2)))
            if s2[driver] is not None:
                s2[driver] = int(np.clip(s2[driver] + rng.integers(-1, 2), 2, laps-1))
            
        # Qualifying Sim & Starting Grid
        qual_draw = {driver: np.random.normal(loc=avg_pace[driver], scale=avg_std[driver]) for driver in drivers}
        race_order = [driver for driver, _ in sorted(qual_draw.items(), key=lambda x: x[1])]

        # Safety Car Probability
        sc_happens = rng.random() < p_SC
        sc_lap = int(rng.integers(1, laps+1)) if sc_happens else None
        if sc_lap is not None:
            sc_stabilisation_laps = random.randint(SC_STABILISATION_MIN, SC_STABILISATION_MAX)
            stabilisation_window = list(range(sc_lap - 1, min(laps, sc_lap + sc_stabilisation_laps)))
        else:
            stabilisation_window = []

        individual_times = {driver: [0.0] for driver in drivers}
        cumulative_laptimes = {driver: [0.0] for driver in drivers}
        stin_laps_counter = {driver: 0 for driver in drivers}
        current_tires = {driver: [] for driver in drivers}

        init_fuel = 110.0
        fuel_cons = init_fuel / laps
        tire_factors = {"SOFT": 0.997, "MEDIUM": 1.0, "HARD": 1.006}

        def calibrate_quadratic(A, D, S):
            c = 2.0 * (S * A - D) / (A * A)
            r0 = S - c * A
            return r0, c
        
        targets = {
            "SOFT": {"A": 18, "D": 0.90, "S": 0.070},
            "MEDIUM": {"A": 22, "D": 0.60, "S": 0.045},
            "HARD": {"A": 28, "D": 0.35, "S": 0.025}
        }
        TYRE_DEG = {c: calibrate_quadratic(t["A"], t["D"], t["S"]) for c, t in targets.items()}

        def tyre_deg_seconds(compound, age_laps, deg_scale, cliff_start_frac, gamma):
            age = max(0.0, float(age_laps))
            r0, c = TYRE_DEG[compound]
            base = deg_scale * (r0 * age + 0.5 * c * age * age)
            if cliff_start_frac is not None and gamma > 0.0:
                A = targets[compound]["A"]
                Ac = cliff_start_frac * A
                if age > Ac:
                    base += gamma * (age - Ac) ** 3
            return base
        
        overtake_attempts = {driver: 0 for driver in drivers}
        overtake_success = {driver: 0 for driver in drivers}

        # ------------- Lap Loop -------------
        for lap in range(laps):
            lap1b = lap + 1
            order_snapshot = race_order[:]
            pitted_this_lap = set()

            for driver in drivers:
                w1 = pit_window(s1[driver], lo=1, hi=laps-1, half_width=3)
                w2 = pit_window(s2[driver], lo=1, hi=laps-1, half_width=3)

                # Normalise Strategy to List of Strings
                seq = strat[driver] if isinstance(strat[driver], (list, tuple)) else [strat[driver]]
                seq = list(seq)
                if len(seq) == 3:
                    first_tire, second_tire, third_tire = seq
                elif len(seq) == 2:
                    first_tire, second_tire = seq
                    third_tire = None
                elif len(seq) == 1:
                    first_tire = seq[0]
                    second_tire, third_tire = None, None
                else:
                    return ValueError(f"Invalid Strategy for {driver}: {strat[driver]}")
                
                if (s1[driver] is None) or (lap1b < s1[driver]):
                    intended = first_tire
                elif (s2[driver] is None) or (lap1b < s2[driver]):
                    intended = second_tire
                else:
                    intended = third_tire

                decision = decision_making(driver, avg_pace, avg_std, race_order, cumulative_laptimes, lap, rng, individual_times,
                                           s1, s2, strat, laps, stin_laps_counter, current_tires, sim, racecontrol_log, decision_log)

                if decision is None:
                    print(f"[BUG] decision_making returned None at Lap {lap+1} for {driver}")
                else:
                    did_decision_pit = (decision.get("action") == "PIT")

                in_w1 = (lap1b in w1) if w1 else False
                in_w2 = (lap1b in w2) if w2 else False

                if did_decision_pit:
                    if in_w1 and (s1[driver] is None or lap1b != s1[driver]):
                        s1[driver] = lap1b
                    elif in_w2 and (s2[driver] is None or lap1b != s2[driver]):
                        s2[driver] = lap1b
                    pitted_this_lap.add(driver)

                is_sched_pit = ((s1[driver] is not None and lap1b == s1[driver]) or
                                (s2[driver] is not None and lap1b == s2[driver]))
                is_pitstop = did_decision_pit or is_sched_pit

                new_compound = decision.get("new_tyre") if did_decision_pit else intended
                if not isinstance(new_compound, str):
                    raise ValueError(f"new_tyre must be str for {driver}, got {new_compound}")
                
                # ----- Tyre State Update (once/driver/lap) -----
                if lap == 0:
                    current_tires[driver].append(new_compound if is_pitstop else intended)
                    stin_laps_counter[driver] = 0
                elif is_pitstop:
                    current_tires[driver].append(new_compound)
                    stin_laps_counter[driver] = 0
                else:
                    current_tires[driver].append(current_tires[driver][-1])
                    stin_laps_counter[driver] += 1

                compound_now = current_tires[driver][-1]
                if not isinstance(compound_now, str):
                    raise ValueError(f"Tyre not string for {driver}: {compound_now}")
                laptime_factor = tire_factors[compound_now]

                fuel_onboard = init_fuel - fuel_cons * lap
                fuel_effect = fuel_onboard * 0.035

                deg_scale = deg_scale_from_std_logistic(
                    avg_pace[driver], avg_std[driver], mid=0.015, k=45.0, out=(0.9, 1.5)
                )
                tire_deg = tyre_deg_seconds(
                    compound_now, stin_laps_counter[driver], deg_scale, cliff_start_frac=0.8, gamma=3e-4
                )

                base_loc = avg_pace[driver] * laptime_factor
                slowest_lap = min(avg_pace.values())
                lap_time = np.random.normal(loc=base_loc, scale=avg_std[driver]) + tire_deg + fuel_effect

                if lap in stabilisation_window:
                    if is_pitstop:
                        pit_loss = np.random.normal(PIT_LOSS_GREEN[0], PIT_LOSS_GREEN[1]) * SC_PIT_FACTOR
                        lap_time += pit_loss
                    else:
                        lap_time = slowest_lap * 1.4
                else:
                    if is_pitstop:
                        pit_loss = np.random.normal(PIT_LOSS_GREEN[0], PIT_LOSS_GREEN[1])
                        lap_time += pit_loss

                if (order_snapshot.index(driver) > 0) and (not is_pitstop) and (driver not in pitted_this_lap):
                    success = overtake_attempt(
                        driver, order_snapshot, avg_pace, avg_std, lap, cumulative_laptimes, order_snapshot.index(driver), rng
                    )
                    overtake_attempts[driver] += 1
                    if success:
                        overtake_success[driver] += 1
                        lap_time -= OVERTAKE["pass_exec_bonus_s"]
                    else:
                        lap_time += OVERTAKE["follow_loss_s"]
                
                individual_times[driver].append(lap_time)
                cumulative_laptimes[driver].append(lap_time if lap == 0 else cumulative_laptimes[driver][-1] + lap_time)
                    
            times_this_lap = {driver: cumulative_laptimes[driver][-1] for driver in drivers}
            race_order = sorted(drivers, key=lambda x: times_this_lap[x])
            leader = race_order[0]
            leader_time = times_this_lap[leader]

            prev = None
            for pos, drv in enumerate(race_order, start=1):
                gap_to_leader = times_this_lap[drv] - leader_time
                interval = 0.0 if prev is None else times_this_lap[drv] - times_this_lap[prev]
                all_lap_rows.append({
                    "Sim": sim,
                    "Lap": lap + 1,
                    "Position": pos,
                    "Driver": drv,
                    "LapTime": individual_times[driver][-1],
                    "CumulativeLapTime": times_this_lap[drv],
                    "GapToLeader": gap_to_leader,
                    "IntervalAhead": interval,
                    "Leader": leader,
                    "Compound": current_tires[drv][-1],
                    "SC_Happens": sc_happens,
                    "SC_Lap": sc_lap
                })
                prev = drv

        final_times = {driver: cumulative_laptimes[driver][-1] for driver in drivers}
        final_order = sorted(final_times, key=final_times.get)
        all_summaries.append({
            "Sim": sim,
            "Winner": final_order[0],
            "SC_Happens": sc_happens,
            "SC_Lap": sc_lap,
            **final_times
        })

    df_laps = pd.DataFrame(all_lap_rows)
    df_summary = pd.DataFrame(all_summaries)
    df_racecontrol = pd.DataFrame(racecontrol_log)
    df_decision = pd.DataFrame(decision_log)

    df_laps.to_csv("monte_carlo_laps.csv", index=False)
    df_summary.to_csv("monte_carlo_summary.csv", index=False)
    df_racecontrol.to_csv("race_control.csv", index=False)
    df_decision.to_csv("decision_making.csv", index=False)

    return df_laps, df_summary

st.set_page_config(layout="wide")

st.title("Monte Carlo Simulation Analysis")

def run_monte_carlo(num_runs, laps, drivers, race):
    df_laps, df_summary = monte_carlo_safety_car(laps, drivers, race, sims=num_runs, seed=42)
    
schedule = ff1.get_event_schedule(2025)
races = schedule["EventName"]

config1, config2 = st.columns(2)
with config1:
    race_selection = st.selectbox("Race / Circuit", races)
with config2:
    num_runs = st.number_input("Number of Simulations", min_value=100, max_value=500000, value=1000, step=100)

if st.button("Run Simulation"):
    with st.spinner(f"Running Simulation... Over {num_runs*1100} Data Points Being Generated"):
        drivers_abbv, race_laps = load_session(race_selection)
        st.session_state["mc_results"] = run_monte_carlo(num_runs, int(race_laps), drivers_abbv, race_selection)

if "mc_results" in st.session_state:
    results = st.session_state["mc_results"]

    st.subheader("Race Outcome Analysis")

    df_laps = pd.read_csv("monte_carlo_laps.csv")

    last_laps = df_laps.groupby("Sim")["Lap"].max().reset_index()
    last_laps.rename(columns={"Lap":"FinalLap"}, inplace=True)

    final_classification = df_laps.merge(last_laps, on=["Sim"])
    final_classification = final_classification[final_classification["Lap"]==final_classification["FinalLap"]]

    final_classification = final_classification.sort_values(["Sim", "Position"]).reset_index(drop=True)

    race_pos_chart = alt.Chart(final_classification).mark_line().encode(
        x="Sim:Q", y="Position:Q", color="Driver:N"
    )

    st.subheader("Finishing Positions / Simulation")
    st.altair_chart(race_pos_chart, use_container_width=True)

    driver_finish_pos = []
    driver_list = final_classification["Driver"].unique()
    for driver in driver_list:
        driver_df = final_classification[final_classification["Driver"]==driver]
        driver_finish_pos.append({
            "Driver": driver,
            "AvgPos": driver_df["Position"].mean()
        })

    driver_finish_pos_df = pd.DataFrame(driver_finish_pos)
    sorted_df = driver_finish_pos_df.sort_values("AvgPos")
    driver_avg_finish = alt.Chart(sorted_df).mark_bar().encode(
        x=alt.X("Driver:N", sort="y"), y="AvgPos:Q", color="Driver:N"
    )

    order = (df_laps.groupby("Driver", as_index=False)["Position"].median().sort_values("Position", ascending=True)["Driver"].tolist())

    box_plot_pos = alt.Chart(final_classification).mark_boxplot().encode(
        x=alt.X("Driver:N", title="Driver", sort=order), y=alt.Y("Position:Q", title="Finishing Position"), color="Driver:N"
    )

    graph2, graph3 = st.columns(2)
    with graph2:
        st.subheader("Average Finish Position")
        st.altair_chart(driver_avg_finish, use_container_width=True)
    with graph3:
        st.subheader("Box-Plot of Finish Position")
        st.altair_chart(box_plot_pos, use_container_width=True)

    df_summary = pd.read_csv("monte_carlo_summary.csv")
    winners = []

    win_num = df_summary["Winner"].value_counts()
    win_percent = (win_num / len(df_summary) * 100).reset_index()
    win_percent.columns = ["Driver", "WinPercent"]

    win_percent_chart = alt.Chart(win_percent).mark_bar().encode(
        x="Driver:N", y="WinPercent:Q", color="Driver:N"
    )

    graph4, graph5 = st.columns(2)
    with graph4:
        st.subheader("Win Percentage by Driver")
        st.altair_chart(win_percent_chart, use_container_width=True)

    num_runs = final_classification["Sim"].nunique()

    # Keep a full driver list so drivers with 0 podiums still appear
    all_drivers = final_classification["Driver"].unique()

    # Count podiums per driver (P1–P3 at final lap)
    podium_counts = (
        final_classification.query("Position <= 3")["Driver"]
        .value_counts()
        .reindex(all_drivers, fill_value=0)              # include zeros
        .rename_axis("Driver")
        .reset_index(name="PodiumCount")
    )

    # Convert to percentage
    podium_counts["PodiumPercent"] = podium_counts["PodiumCount"] / num_runs * 100

    # Plot percentage (sorted best→worst)
    podium_chart = (
        alt.Chart(podium_counts)
        .mark_bar()
        .encode(
            x=alt.X("Driver:N", sort="-y", title="Driver"),
            y=alt.Y("PodiumPercent:Q", title="Podium %"),
            color="Driver:N",
            tooltip=[
                "Driver:N",
                alt.Tooltip("PodiumCount:Q", title="Podium count"),
                alt.Tooltip("PodiumPercent:Q", title="Podium %", format=".1f")
            ],
        )
    )

    with graph5:
        st.subheader("Podium Percentage per Driver")
        st.altair_chart(podium_chart, use_container_width=True)

    if st.button("Clear Results"):
        st.session_state.pop("mc_results")