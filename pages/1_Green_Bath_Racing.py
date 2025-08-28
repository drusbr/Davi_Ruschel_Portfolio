import streamlit as st

st.set_page_config(layout="wide")
st.title("Green Bath Racing — Engineering Project Experience")

# ---- INTRO ----
st.info("""
Green Bath Racing is a student engineering team focused on developing ultra-efficient prototype vehicles.  
In 2025 the team achieved **4th place in Europe & North Africa** and set a new **UK efficiency record (835 km/kWh)**.  
This success was driven by advances in data systems, simulation, and strategy optimisation — areas I directly contributed to.
""")

# ---- ROLE ----
st.subheader("My Contribution")
st.markdown("""
I joined the team in my third year, initially as a **Data & Telemetry Engineer**, before expanding into **Simulation & Strategy Development**.  
My work centred on creating tools to **capture, process, and apply data in real time** to improve team performance.
""")

# ---- TELEMETRY ----
with st.expander("📡 Data & Telemetry Engineer"):
    st.markdown("""
    - **Objective**: Build a new telemetry system for testing and competition.  
    - **Tools**: ESP32 microcontroller, sensors, Python, Google Sheets, Streamlit.  
    - **Key Tasks**:  
        • Programmed and integrated sensors for temperature, current, and speed monitoring  
        • Developed a live data dashboard in Streamlit for driver and engineer feedback  
        • Designed a 4G cloud-based transmission concept to overcome interference issues  
    - **Impact**: Delivered reliable live-data capture and visualisation, enabling informed adjustments during runs.
    """)

# ---- SIMULATION ----
with st.expander("🧠 Simulation & Strategy Development"):
    st.markdown("""
    - **Objective**: Optimise vehicle efficiency using a simulation-driven approach.  
    - **Tools**: Python, NumPy, Pandas, Matplotlib, Streamlit.  
    - **Key Tasks**:  
        • Built a probabilistic simulation tool modelling vehicle dynamics, energy use, and acceleration strategy  
        • Created a logic-based random strategy generator with constraints (cornering, acceleration zones)  
        • Developed a two-lap evaluation framework to test strategy repeatability  
        • Conducted sensitivity analysis on factors such as motor efficiency and track gradient  
        • Presented findings to the team and adjusted strategies during live competition  
    - **Impact**: Improved efficiency progressively from **600 → 787 km/kWh** in practice runs, then to **835 km/kWh** in official runs, setting a new national record.
    """)

# ---- PHOTOS ----
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1: st.image("IMG_5620.jpg", caption="Team at Competition")
    with c2: st.image("GBR picture (162).JPG", caption="Engineering During Run")
    with c3: st.image("IMG_4801.jpg", caption="Post-Run Data Analysis")

# ---- KEY OUTCOMES ----
st.subheader("Key Outcomes")
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("National Record", "835 km/kWh")
with m2: st.metric("Regional Ranking", "4th place (Europe & North Africa)")
with m3: st.metric("Simulation Accuracy", "±4.3% vs real performance")
with m4: st.metric("Efficiency Improvement", "600 → 835 km/kWh")

# ---- LESSONS ----
st.subheader("Skills & Lessons Learned")
st.info("""
**Technical & Analytical**  
- Built and validated a simulation tool with real-world error of only 4.3%  
- Applied data-driven optimisation to achieve measurable gains across multiple runs  
- Strengthened knowledge of system dynamics, efficiency trade-offs, and sensitivity analysis  

**Problem-Solving & Decision-Making**  
- Made rapid adjustments under time pressure using imperfect data  
- Balanced simulation predictions with live feedback to refine strategies  
- Practiced risk management in selecting between conservative and aggressive approaches  

**Collaboration & Communication**  
- Took responsibility for strategy decisions, presenting evidence-based recommendations  
- Communicated complex adjustments clearly to drivers and engineers  
- Worked effectively in a multidisciplinary, high-pressure environment  

**Resilience**  
- Recovered from setbacks (e.g. disqualification) to deliver a record-breaking result  
- Built confidence in handling responsibility and accountability for outcomes  
""")

# ---- REFLECTION (optional) ----
with st.expander("Personal Reflection"):
    st.markdown("""
    Beyond the technical outcomes, this experience taught me what it feels like to make  
    **high-stakes engineering decisions under pressure**. Watching the results of my analysis  
    play out in real time confirmed my passion for applying data and engineering tools  
    to deliver performance improvements.
    """)

