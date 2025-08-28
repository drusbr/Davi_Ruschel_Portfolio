import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("Engineering Portfolio")

with st.expander("**Key Highlights of this Portfolio**"):
    st.write("- Read about me in this page")
    st.write("- Learn more about my experience in Green Bath Racing as a Race Strategist & Performance Engineer")
    st.write("- Have a play with the simulation I developed, it allows you to understand what it does more in depth!")
    st.write("- Read about my internship at Zikeli")
    st.write("- Explore the F1 Dashboard I developed for personal use on data analysis and race insights.")
    st.write("- See some other projects I have done over the years")

fig1_col1, fig1_col2 = st.columns(2)
with fig1_col1:
    st.image("GBR picture (48).jpeg", caption="Silesia Ring 2025 - Shell Eco Marathon", use_container_width=True)

with fig1_col2:
    st.subheader("About Me")
    st.markdown("I am a **final-year (MEng) Mechanical Engineering student at the University of Bath**, set to graduate with a **First Class Honours degree**.")

    st.markdown("I carry a deep passion for motorsport, performance engineering, and simulation-driven strategy.")
    st.markdown("Over the past few years, I’ve developed hands-on experience through roles in Green Bath Racing, where I built a custom race strategy simulation tool and a real-time telemetry system used in competition. My work contributed to the team’s 4th-place finish at the 2025 Shell Eco-marathon and a new UK national efficiency record.")
    st.markdown("Beyond technical projects, I thrive in high-pressure environments where engineering decisions matter — especially when they intersect with racecraft, data, and optimisation. My goal is to work in motorsport, particularly in roles focused on vehicle performance, strategy, or data engineering.")
    st.markdown("This portfolio highlights some of the projects, tools, and experiences that have shaped my journey so far. Thanks for stopping by!")

seccol1, seccol2 = st.columns(2)

st.subheader("Core Skills & Tools")

with st.expander("Technical Skills"):
    st.markdown("""
    **Programming & Data Analysis**  
    Python (pandas, NumPy, matplotlib, seaborn, scikit-learn), MATLAB; experience building dashboards, simulators, and ML models.  

    **Simulation & Modelling**  
    Vehicle dynamics, lap simulation, probabilistic strategy modelling (Monte Carlo, game theory), control systems, and FEA.  

    **CAD & Design**  
    Autodesk Inventor, AutoCAD; experience with design-for-assembly and subsystem/component design.  

    **Data Visualisation & Dashboards**  
    Streamlit, Tkinter; real-time telemetry GUIs and interactive strategy simulators.  

    **Office & Documentation Tools**  
    Advanced Microsoft Excel, Word, PowerPoint; experience producing high-level technical documentation and analysis.  

    **Engineering Tools**  
    Granta EduPack (material selection), stress calculations, mechanical testing, root locus and time-domain system analysis.
    """)

with st.expander("Soft Skills"):
    st.markdown("""
    **Soft & Transferable Skills**  
    - Team leadership & project management (Team Manager – Green Bath Racing)  
    - Decision-making under pressure (Shell Eco-marathon race strategy)  
    - Communication: technical reporting, presentations, stakeholder engagement  
    - Entrepreneurship: founded and manage a profitable event brand in Bath  
    - Multilingual: fluent Portuguese & English, professional Spanish  
    """)

st.subheader("Studies")

st.markdown("""
Throughout my degree, I’ve focused on continuous improvement. While my first two years were below my own expectations, 
this challenge drove me to realign my priorities and push for excellence.  

By Year 3, I achieved an **average of 79% in Semester 1 exams** and **76% in my individual Technical Report** 
(for the Green Bath Racing simulation & telemetry project in Semester 2), finishing the year with a **76% overall average**.  
These results reflect both resilience and my ability to excel when applying engineering concepts to real-world projects.  
""")

st.info("""
Coming into my final year, I have chosen advanced modules that align with goal of pushing the limits and taking on challenges:

- **System Modelling & Simulation**  
- **Computational Fluid Dynamics**  
- **Composite Materials**  
- **Electric Propulsion Systems**  
- **Aerodynamics**
""")

st.markdown("""
This academic foundation equips me with the tools to analyse, optimise, and innovate in data-driven engineering environments.
""")
