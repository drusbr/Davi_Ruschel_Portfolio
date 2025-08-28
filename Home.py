import streamlit as st
from datetime import date

st.set_page_config(page_title="Davi Ruschel — Engineering Portfolio", layout="wide")

# ---- HERO ----
colA, colB = st.columns([1.1, 0.9], vertical_alignment="center")
with colA:
    st.title("Davi Ruschel")
    st.markdown(
        """
**MEng Mechanical Engineering (Final Year) — University of Bath**  
Simulation-driven performance engineering, motorsport strategy, and data products.
"""
    )
    st.markdown(
        """
- Built a **race strategy simulator** and **real-time telemetry system** used in competition  
- Contributed to **4th place at Shell Eco-Marathon (Europe & North Africa, 2025)**  
- Part of setting a **new UK efficiency record** with Green Bath Racing
"""
    )
    # c1, c2, c3, c4 = st.columns(4)
    # with c1: st.link_button("📄 View CV", "https://your-cv-link.pdf")
    # with c2: st.link_button("🧠 Strategy Simulator", "/SEM_Simulation")
    # with c3: st.link_button("📊 F1 Dashboard", "/F1_Dashboard")
    # with c4: st.link_button("🛠 Projects", "/Projects")

with colB:
    st.image("GBR picture (48).jpeg", caption="Silesia Ring 2025 — Shell Eco-Marathon", use_container_width=True)

st.divider()

# ---- QUICK METRICS ----
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("SEM Result (2025)", "4th", help="Europe & North Africa")
with m2: st.metric("National Record", "UK Efficiency", help="Set with GBR 2025")
with m3: st.metric("Exam Avg (Y3 S1)", "79%")
with m4: st.metric("Tech Report (Y3)", "76%", help="Simulation & telemetry")

# ---- ABOUT (tight) ----
st.subheader("About Me")
st.markdown(
    """
I’m a final-year MEng at Bath focused on **vehicle performance, simulation, and race strategy**.  
I build tools that **turn data into decisions** — from Monte Carlo race simulators to real-time telemetry GUIs.  
Goal: **motorsport performance/strategy**; open to **aerospace/consulting** roles where modelling and decisions matter.
"""
)

# ---- CORE SKILLS (collapsed by default) ----
sk1, sk2 = st.columns(2)
with sk1:
    with st.expander("Technical"):
        st.markdown(
            """
- **Programming & Data**: Python (pandas, NumPy, matplotlib, scikit-learn), MATLAB  
- **Simulation/Modelling**: vehicle dynamics, lap simulation, Monte Carlo, control  
- **CAD/Design**: Autodesk Inventor, AutoCAD  
- **Dashboards/Apps**: Streamlit, Tkinter; real-time telemetry GUIs
"""
        )
with sk2:
    with st.expander("Soft & Leadership"):
        st.markdown(
            """
- Team leadership & project delivery (GBR Team Management/Strategy)  
- Decision-making under pressure (race strategy), stakeholder comms  
- Entrepreneurship (Baianá events), multilingual (EN/PT, ES professional)
"""
        )

# ---- FEATURED PROJECTS (card-like rows) ----
st.subheader("Featured Work")
p1, p2 = st.columns(2)
with p1:
    st.markdown("### 🧠 Race Strategy Simulator")
    st.markdown(
        """
Probabilistic **Monte Carlo** simulator with tyre degradation, SC events, and pit windows.  
Used to evaluate strategies pre-race and generate driver guidance.
"""
    )
    st.markdown("[Open project →](/SEM_Simulation)")
with p2:
    st.markdown("### 📡 Real-Time Telemetry System")
    st.markdown(
        """
ESP32-based data pipeline to live dashboards; **driver feedback** and post-run analysis.  
Deployed at SEM; informed decisions contributing to record performance.
"""
    )
    st.markdown("[See build notes →](/Telemetry)")

p3, p4 = st.columns(2)
with p3:
    st.markdown("### 🏁 F1 Data & Strategy Dashboard")
    st.markdown(
        """
Interactive Streamlit app for pace deltas, stint models, and pit windows across seasons.  
Focus on **race insight and decision support**.
"""
    )
    st.markdown("[Explore dashboard →](/F1_Dashboard)")
with p4:
    st.markdown("### ✈️ Structural Analysis — Landing Gear Fork")
    st.markdown(
        """
FEA of C152 fork for landing/ground loads; stress concentrations & safety factors.  
Demonstrates **aerospace-relevant analysis** capability.
"""
    )
    st.markdown("[Read summary →](/Projects#landing-gear)")

st.divider()

# ---- STUDIES (concise + scannable) ----
st.subheader("Recent Academic Highlights")
st.markdown(
    """
- **Year 3**: 79% exams (S1), **76% Technical Report** (simulation & telemetry)  
- **Final-Year Modules**: System Modelling & Simulation, CFD, Composites, Electric Propulsion, Aerodynamics
"""
)

st.info("I’m currently targeting graduate roles in **vehicle performance & race strategy** (motorsport), and **simulation/data-driven engineering** across aerospace & consulting.")

# ---- FOOTER / CONTACT ----
st.divider()
cA, cB = st.columns([0.6, 0.4])
with cA:
    st.markdown("**Contact**: [LinkedIn](https://www.linkedin.com/in/davi-ruschel-aa6a45244/) • [Email](mailto:daviruschel9@gmail.com)")
with cB:
    st.caption(f"Last updated: {date.today().isoformat()}")



