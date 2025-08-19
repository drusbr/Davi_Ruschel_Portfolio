import streamlit as st

st.title("Industry Experience - Zikeli Engineering Internship")
st.info("""
In Summer 2025, I completed an engineering internship at Zikeli, a leading manufacturing company specializing in roll-forming machinery and automation systems.

My work covered both mechanical assembly and engineering design, giving me insight into how complex systems go from concept to real-world operation.
""")

st.header("Mechanical Assembly, Jul 2025")

fig_col1, fig_col2, fig_col3 = st.columns(3)
with fig_col1:
    st.image("IMG_4815.jpg")
with fig_col2:
    st.image("IMG_4853.jpg")
with fig_col3:
    st.image("IMG_4865.jpg")

mechAss_col1, mechAss_col2 = st.columns([2, 1])

with mechAss_col1:
    st.write("During the first phase of my internship, I worked on the shop floor assembling machines, including a roll-forming machine, gearboxes and hydraulic systems. This hands-on exposure gave me a deep appreciation for how engineering designs are translated into functional machinery.")
    with st.expander("**Experience Output**"):
        st.markdown("""
                    - **Assembled complex machinery**, including roll-forming systems, gearboxes, and hydraulic subsystems.
                    - **Collaborated with technicians** on the shop floor, gaining direct exposure to manufacturing processes and tolerances.
                    - **Interpreted technical drawings** and BOMs to guide accurate mechnical assembly and system integration.
                    - **Identified design-for-assembly limitations** and proposed improvements to reduce time and complexity.
                    - **Developed practical engineering intuition**, understanding how design choices impact real-world assembly and maintenance.
                    """)
    
    st.markdown("""                
                Being directly involved in the mechnical assembly process - rather than just observing - allowed me to understand the practical constraints of manufacturing. I worked alongside technicians and machinists, and had the opportunity to interact with machining stations, quality control, and testing.

                What stood out most was the value of *design for assembly*. Encountering designs that were difficult to implement pushed me to think about how I would have approached them differently as an engineer. This exposure grounded my design thinking in practicality, and made me a more well-rounded engineer. 
                """)
    
with mechAss_col2:
    st.image("IMG_4888.jpg")

st.header("Design Engineer & Optimization, Aug 2025")

st.markdown("""
After completing the mechanical assembly phase, I moved into the role of **Design Engineer**. 
This stage of the internship exposed me to the **full cycle of engineering design** — from diagnosing failures on existing equipment 
to contributing to the creation of entirely new roll-forming machinery.  

One of my first tasks was to investigate recurring saw blade failures on a tube-cutting machine. By carrying out 
data analysis, I discovered a misalignment between two key components that was causing the saw to experience lateral forces 
and eventually break. This diagnosis allowed the team to implement corrective measures, preventing further downtime and costs.  

Beyond problem-solving, I was also trusted to contribute directly to new product development. 
I supported the design of a client’s roll-forming machine, where I ran stress calculations, conducted material testing 
to assess the suitability of the chosen material, and created CAD models for several critical components. 
This gave me valuable experience in bridging theoretical calculations with practical machine design.  
""")

with st.expander("**Experience Output**"):
    st.markdown("""
    - **Failure Diagnosis & Data Analysis**: Investigated recurring saw blade failures on a tube-cutting machine. By analysing performance data, I identified a critical misalignment between two components that was causing lateral forces and premature blade breakage, enabling the team to implement corrective measures.  

    - **Roll-Forming Machine Development**: Supported the design of a new roll-forming machine for a client by performing **stress calculations**, conducting **material testing** to validate the suitability of the chosen material, and creating **CAD models** for key components.  

    - **Technical Documentation**: Managed and digitised the company’s archive of technical drawings and maintenance guides for roll-forming machines, improving accessibility and traceability of engineering data.  

    - **System Design Contribution**: Produced a new **technical drawing for the lubrication system** of a tube-forming machine, supporting the company’s design and maintenance documentation.  
    """)

st.markdown("""
This stage of my internship strengthened my ability to connect **theory with practice** — from diagnosing mechanical failures 
to contributing to new machine design. It broadened my perspective as an engineer, giving me a more complete understanding of 
how design choices affect manufacturing, reliability, and maintenance.
""")