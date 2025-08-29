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
Focused on **simulation, optimisation, and data-driven engineering solutions**.  
"""
    )
    st.markdown(
        """
- Designed **simulation tools** for complex systems and performance evaluation  
- Built **real-time data acquisition and visualisation platforms**  
- Delivered projects across academic, team, and industry settings  
"""
    )
        
    st.link_button("📄 View CV", "Davi Ruschel - CV.pdf", use_container_width = True)

with col2:
    st.image("GBR picture (48).jpeg", caption="Engineering project in competition environment", use_container_width=True)

st.divider()

# ---- METRICS ----
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Competition Result", "4th", help="Delivered with custom tools & strategy")
with m2: st.metric("Record", "835 km/kWh", help="National efficiency record")
with m3: st.metric("Exam Avg (Year 3 S1)", "79%")
with m4: st.metric("Technical Report", "76%", help="Individual project on simulation & data systems")

# ---- ABOUT ----
st.subheader("About Me")
st.markdown(
    """
I am a **final-year Mechanical Engineering student**, set to graduate with **First Class Honours**.  
My focus is developing tools that combine **engineering analysis with data-driven decision making**.  

This portfolio highlights the projects, skills, and experiences that shaped my journey —  
from building simulation models and interactive dashboards to delivering engineering solutions  
through academic research, group projects, and industry internships.
"""
)

# ---- SKILLS ----
colA, colB = st.columns(2)
with colA:
    with st.expander("Technical Skills"):
        st.markdown(
            """
- **Programming & Data**: Python (pandas, NumPy, scikit-learn, matplotlib), MATLAB  
- **Simulation & Modelling**: dynamic systems, probabilistic analysis, optimisation methods  
- **CAD & Design**: Autodesk Inventor, AutoCAD  
- **Dashboards & Apps**: Streamlit, Tkinter (real-time GUIs & analytics tools)  
"""
        )
with colB:
    with st.expander("Transferable Skills"):
        st.markdown(
            """
- Team leadership & project management  
- Decision-making under pressure  
- Clear technical communication (reports, presentations, stakeholder engagement)  
- Entrepreneurship: founded and manage a profitable event brand  
- Multilingual: Portuguese (native), English (fluent), Spanish (professional)  
"""
        )

# ---- FEATURED PROJECTS ----
st.subheader("Featured Projects")
p1, p2 = st.columns(2)
with p1:
    st.markdown("### 🧠 Lap Simulation & Telemetry System")
    st.markdown(
        """
Developed a **probabilistic simulator** to evaluate system performance under uncertainty.  
Applied in a real competition setting with measurable results.  
"""
    )
    
with p2:
    st.markdown("### 📡 FEA of a Cessna-152 Landing Gear")
    st.markdown(
        """
Simulated the structural performance of the landing gear fork of a Cessna-152 under landing conditions, and exploring whether the component could withstand hard-landing scenarios. 
Conducted a material selection procedure for the landing gear fork, using results from the FEA as basis for different values.
"""
    )
    

p3, p4 = st.columns(2)
with p3:
    st.markdown("### 📊 F1 Monte-Carlo & Probabilistic Simulation")
    st.markdown(
        """
Developed a simulation to model expected outcomes on different races using over 15 million datapoints from previous seasons. 
Modelling tire degradation, overtakes, crashes, and other stochastic events.
"""
    )
    
with p4:
    st.markdown("### 🏎️ Vehicle Performance Test Rig")
    st.markdown(
        """
Designing a modular, portable dyno capable of replicating track-specific loads (cornering, elevation, resistive forces), enabling off-track vehicle testing.
"""
    )
    

st.divider()

# ---- STUDIES ----
st.subheader("Academic Highlights")
st.markdown(
    """
- **Year 3:** 79% (exams, S1) · 76% (individual technical report)  
- **Final Year Modules:** System Modelling & Simulation, CFD, Composites, Electric Propulsion, Aerodynamics
"""
)

st.info(
    "Currently seeking graduate opportunities where I can apply **simulation, optimisation, and data-driven engineering** to deliver impactful results."
)

# ---- FOOTER ----
st.divider()
colF1, colF2 = st.columns([0.7, 0.3])
with colF1:
    st.markdown("**Contact**: [LinkedIn](https://linkedin.com/in/your-handle) • [Email](mailto:you@email.com)")
with colF2:
    st.caption(f"Last updated: {date.today().isoformat()}")







