import streamlit as st
from datetime import date

st.set_page_config(page_title="Davi Ruschel — Engineering Portfolio", layout="wide")

# ---- HERO ----
col1, col2 = st.columns([1.1, 0.9], vertical_alignment="center")

with col1:
    st.title("Davi Ruschel")
    st.markdown(
        """
**MEng Mechanical Engineering (Final Year) — University of Bath**  
Engineer specialising in **simulation, optimisation, and data-driven performance** across  
motorsport, aerospace, and advanced engineering domains.
"""
    )
    st.markdown(
        """
- Designed **probabilistic simulation tools** for complex decision-making  
- Built **real-time telemetry and data visualisation platforms**  
- Applied engineering solutions in competition and industry: from **Shell Eco-marathon** to **Zikeli (Brazil)**  
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.link_button("📄 View CV", "https://your-cv-link.pdf")
    with c2: st.link_button("🧠 Strategy Simulator", "/Strategy_Simulation")
    with c3: st.link_button("📊 F1 Data Dashboard", "/F1_Strategy_Project")
    with c4: st.link_button("🚗 Green Bath Racing", "/Green_Bath_Racing")

with col2:
    st.image("GBR picture (48).jpeg", caption="Applied at Shell Eco-Marathon — Silesia Ring 2025", use_container_width=True)

st.divider()

# ---- METRICS ----
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Competition Result", "4th", help="Shell Eco-Marathon Europe & North Africa, 2025")
with m2: st.metric("UK Record", "835 km/kWh", help="New efficiency record with GBR 2025")
with m3: st.metric("Exam Avg (Year 3 S1)", "79%")
with m4: st.metric("Tech Report", "76%", help="Simulation & telemetry project")

# ---- ABOUT ----
st.subheader("About Me")
st.markdown(
    """
I am a **final-year Mechanical Engineering student at Bath**, set to graduate with **First Class Honours**.  
My focus is building engineering tools that **turn data into decisions**: from Monte Carlo race strategy simulators  
to telemetry GUIs and structural analysis.

While I am deeply passionate about **motorsport and performance engineering**, my skillset is **domain-flexible** —  
equally applicable to **aerospace, automotive, and data-driven engineering challenges** where optimisation and  
decision support are critical.
"""
)

# ---- SKILLS ----
colA, colB = st.columns(2)
with colA:
    with st.expander("Technical Skills"):
        st.markdown(
            """
- **Programming & Data**: Python (pandas, NumPy, scikit-learn, matplotlib), MATLAB  
- **Simulation/Modelling**: vehicle dynamics, lap simulation, Monte Carlo, control systems  
- **CAD/Design**: Autodesk Inventor, AutoCAD  
- **Dashboards/Apps**: Streamlit, Tkinter (real-time GUIs & analytics tools)
"""
        )
with colB:
    with st.expander("Leadership & Transferable Skills"):
        st.markdown(
            """
- Team leadership & project management (Team Manager, Green Bath Racing)  
- Decision-making under pressure (competition strategy)  
- Communication: reporting, presenting, stakeholder engagement  
- Entrepreneurship: founder of **Baianá** (profitable student events brand)  
- Languages: Portuguese (native), English (fluent), Spanish (professional)
"""
        )

# ---- PROJECT TEASERS ----
st.subheader("Featured Projects")
p1, p2 = st.columns(2)
with p1:
    st.markdown("### 🧠 Probabilistic Simulation Tool")
    st.markdown(
        """
Built a **Monte Carlo decision-support simulator** to optimise performance under uncertainty.  
**Application:** Shell Eco-Marathon race strategy, contributing to UK efficiency record.  
"""
    )
    st.markdown("[Explore →](/Strategy_Simulation)")
with p2:
    st.markdown("### 📡 Real-Time Telemetry Platform")
    st.markdown(
        """
Developed a data acquisition & visualisation pipeline for live driver feedback.  
**Application:** Implemented in competition, enabled data-driven adjustments.  
"""
    )
    st.markdown("[See more →](/Green_Bath_Racing)")

p3, p4 = st.columns(2)
with p3:
    st.markdown("### 📊 F1 Data & Strategy Dashboard")
    st.markdown(
        """
Streamlit app analysing millions of lap/telemetry data points.  
Focus: **pit windows, stint models, SC scenarios**.  
"""
    )
    st.markdown("[View →](/F1_Strategy_Project)")
with p4:
    st.markdown("### 🏭 Industrial Internship — Zikeli (Brazil)")
    st.markdown(
        """
Hands-on mechanical design and CAD, including shaft/bearing design  
and pass-by-pass tooling weight calculators.  
"""
    )
    st.markdown("[Read summary →](/Zikeli_Internship)")

st.divider()

# ---- STUDIES ----
st.subheader("Academic Highlights")
st.markdown(
    """
- **Year 3:** 79% (exams, S1) · 76% (individual Technical Report — simulation & telemetry)  
- **Final Year Modules:** System Modelling & Simulation, CFD, Composites, Electric Propulsion, Aerodynamics
"""
)

st.info(
    "Targeting graduate roles in **vehicle performance & race strategy (motorsport)** and in **simulation/data-driven engineering** across aerospace, automotive, and consulting."
)

# ---- FOOTER ----
st.divider()
colF1, colF2 = st.columns([0.7, 0.3])
with colF1:
    st.markdown("**Contact**: [LinkedIn](https://linkedin.com/in/your-handle) • [Email](mailto:you@email.com)")
with colF2:
    st.caption(f"Last updated: {date.today().isoformat()}")
