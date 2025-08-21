import streamlit as st

st.set_page_config(layout="wide")
st.title("Green Bath Racing - Competition Team")

st.info("""
        Green Bath Racing is the University of Bath's Shell Eco-Marathon team, a competition with the focus of developing the most energy-efficient vehicle.

        The team entered the competition in a strong position. In its first year in this category, it achieved 8th place in the Europe & North Africa Region, while also setting the UK's national record for efficiency.

        Fast forward to 2025, the team had an impressive improvement and went on to achieve 4th place, and once again, setting the UK national's record.
        """)

st.subheader("My Role in Green Bath Racing")
st.markdown("I joined GBR in my third year of studies as a Telemetry Engineer. The team wanted to focus on developing a brand-new telemetry system to capture live-data from the vehicle to be used during testing/competition.")

with st.expander("**Telemetry Engineer**"):
    st.markdown("""
    - **Goal**: Develop a brand-new telemetry system for data collection during testing and competition.
    - **Tools Used**: ESP32 Development Board, Sensors, Python and Google Sheets.
    - **What I Did**:
        - Assembled and programmed sensors to monitor temperature, current, and speed as well as captur motor-related metrics from the motor controller.
        - Built a live data dashboard using Streamlit.
        - Proposed a 4G cloud-based transmission method to avoid interference.
    - **Impact**: Enabled real-time driver feedback and post-run analysis, contributing to the team’s 4th-place finish in 2025.
    """)

with st.expander("**Race Strategist & Vehicle Performance Engineer**"):
    st.markdown("In the beginnning of 2025, as part of my university curriculum, I was part of a group that had the task of developing a new prototype vehicle to compete in the SEM.")
    st.markdown("My role in the group was to develop a simulation program that enabled the optimization of the race strategy.")

    st.markdown("""
    - **Goal**: Optimise race strategy to maximise energy efficiency during the Shell Eco-marathon using a simulation-based approach.
    - **Tools Used**: Python, Pandas, NumPy, Matplotlib, Streamlit.
    - **What I Did**:
        - Built a custom race strategy simulation tool that models vehicle dynamics, power usage, and acceleration strategy around a known track layout.
        - Developed a logic-based random strategy generator with constraints (e.g. cornering, acceleration zones) to simulate thousands of valid strategies.
        - Implemented a two-lap evaluation framework to determine strategy repeatability and efficiency under constraints.
        - Performed sensitivity analysis to assess the impact of parameters like motor efficiency and track elevation.
        - Presented findings to the team and iteratively adjusted strategy in real-time during competition.
    - **Impact**: The simulation informed our race strategy for the 2025 competition, contributing to a 4th-place finish and setting a new UK national record for efficiency.
    """)

# with st.expander("**Team Manager (from September 2025)**"):
#     st.markdown("""
#     - **Goal**: Lead Green Bath Racing into the Shell Eco-marathon 2026 season, overseeing technical development, team organisation, and competition delivery.  
#     - **Planned Focus Areas**:
#         - Coordinate a multidisciplinary team of engineers across powertrain, aerodynamics, telemetry, and strategy to deliver a new prototype vehicle.  
#         - Set and monitor project milestones, ensuring the build, testing, and strategy preparation are aligned with competition deadlines and academic requirements.  
#         - Manage the team’s finances and pursue sponsorship opportunities to fund development, testing, and logistics.  
#         - Act as the main point of contact with University staff and SEM organisers, representing the team externally.  
#         - Build team culture through regular workshops, clear communication, and structured delegation of responsibilities.  
#     - **Expected Impact**: Ensure the team delivers a competitive and reliable vehicle, while strengthening GBR’s long-term professionalism and legacy at the University of Bath.  
#     """)

with st.container():
    fig1_col1, fig1_col2, fig1_col3 = st.columns(3)
    with fig1_col1:
        st.image("IMG_5620.jpg", caption="Team Picture at the End of the Competition")
    with fig1_col2:
        st.image("GBR picture (162).JPG", caption = "Race Team Working During a Race")
    with fig1_col3:
        st.image("IMG_4801.jpg", caption = "Data Analysis Session Post Run")

st.info("“**The adrenaline hit me hard.** Watching the green lights come on, knowing the outcome of that run depended on the strategy I’d built — I felt something I hadn’t felt before: completely alive. I’ve been chasing that feeling ever since.”")

st.subheader("Shell Eco-marathon 2025 – Race Strategy Journey")

with st.expander("Read More About my Experience"):
    st.markdown("""
    Coming into the Shell Eco-marathon 2025, I had no idea what to expect. I had spent the past five months developing our race strategy simulation and telemetry tools — and I knew the track layout inside out. What I also knew was what kinds of strategies should work.

    Unfortunately, during Semester 2 we weren't able to conduct any on-track testing, which meant I had no telemetry data to validate the simulation. That was a major setback. But I improvised — I reverse-engineered the previous year’s strategy and found that my simulation predicted the efficiency within just **4.3% error**. That gave me the confidence to use it as our baseline going into competition.

    For our first run, I used a strategy generated by the simulation and made small adjustments based on my own intuition. I briefed the driver and her race engineer, and soon after, the car was rolling onto the track.

    **The adrenaline hit me hard.** Watching the green lights come on, knowing the outcome of that run depended on the strategy I’d built — I felt something I hadn’t felt before: completely alive. I’ve been chasing that feeling ever since.

    That first run returned a disappointing result: **600 km/kWh**, well below expectations. I immediately went back to the workstation and began analysing the telemetry. Looking into energy consumption, motor performance, and speed profiles, I revised the strategy. The next run improved slightly to **696 km/kWh**, but it still wasn’t enough.

    We had four practice runs before the official sessions began. Each one gave us more data to refine the strategy. Our efficiency steadily improved: **600 → 696 → 711 → 787 km/kWh**. By the end of the practice runs, we had already outperformed the previous year’s best by **91 km/kWh** — the first major goal was achieved.

    But I wasn’t satisfied yet. I knew there was more performance in the car. The simulation had shown how sensitive the outcome was to small changes — in fact, a **250-meter shift in an acceleration point** in the last practice run had boosted efficiency from **711 to 787 km/kWh**.

    That night at the hotel, I compared telemetry data from all runs, looking for patterns. I decided to return to the **original strategy suggested by the simulation**, but this time incorporating everything we had learned so far.

    The next morning, I presented the revised strategy to the driver and team manager. They were hesitant, but after I explained the data and reasoning, they trusted me — the responsibility for strategy was mine.

    On our first official run, we arrived **0.8 seconds late** to the start — disqualified. It was a devastating moment. The driver had done everything perfectly. The car was running flawlessly. And I felt the weight of that mistake deeply — it was my call.

    But when the result came in, the mood shifted: **835 km/kWh.** A new national record for GBR.

    We quickly lined up for another official run. I adjusted a few points in the strategy, briefed the driver again, and sent the car back on track. This time, the run was **valid** — and we scored **828 km/kWh**, just below the disqualified one but good enough to place GBR **3rd overall by the end of Day 1**.

    That moment — analysing data under pressure, making the right calls, watching them come to life — it confirmed what I already suspected: **this is what I want to do.**
    """)

strat_col1, strat_col2 = st.columns([2, 1])

st.subheader("What This Experience Taught Me")

st.markdown("**Skills & Lessons Learned**")
st.info("""
**Technical & Analytical Skills**  
- Built and validated a **race strategy simulation tool**, achieving a prediction error of just **4.3%** compared to real-world performance.  
- Developed the ability to **extract actionable insights from incomplete or imperfect data** (reverse-engineering past strategy due to lack of testing).  
- Applied **data-driven optimisation** to refine strategies, improving efficiency from 600 → 787 km/kWh across practice runs.  
- Understood **sensitivity of performance to small changes** (e.g. a 250m shift in an acceleration point delivering significant gains).  
- Conducted real-time analysis of telemetry, including **energy consumption, motor performance, and speed profiles**.  
- Strengthened knowledge of **vehicle dynamics, efficiency trade-offs, and simulation validation under competition conditions**.  

**Decision-Making & Problem-Solving**  
- Learned to make **high-stakes decisions under pressure** with limited time and data.  
- Balanced **simulation predictions with driver feedback** to create practical strategies.  
- Adapted strategies dynamically during competition, using each run as a feedback loop for optimisation.  
- Gained experience in **risk management**, balancing aggressive strategies with reliability.  

**Collaboration & Leadership**  
- Took ownership of **race strategy responsibility**, earning trust from the driver and team manager after presenting data-driven reasoning.  
- Communicated complex strategy adjustments clearly and concisely to the driver and race engineer.  
- Collaborated with a multidisciplinary team (drivers, engineers, strategists) under competition pressure.  

**Resilience & Growth**  
- Experienced setbacks (e.g. disqualification) and used them as **learning opportunities**, bouncing back to set a **new UK national record of 835 km/kWh**.  
- Built confidence in my ability to deliver results in uncertain, high-pressure environments.  
- Discovered a **deep passion for strategy work** — confirmed by the adrenaline and responsibility of competition.  
""")
