import streamlit as st

st.set_page_config(layout="wide")
st.title("Industry Experience — Engineering Internship at Zikeli")

st.info("""
In Summer 2025, I completed an internship at Zikeli Engineering, a manufacturing company specialising in roll-forming machinery and automation systems.  
My work spanned **mechanical assembly** and **engineering design**, giving me end-to-end exposure to how machines are built, tested, and optimised in an industrial setting.
""")

# ---- MECHANICAL ASSEMBLY ----
st.subheader("Mechanical Assembly (Jul 2025)")
st.markdown("""
Worked hands-on in the assembly of complex machinery, gaining direct experience of how engineering designs are translated into functional equipment.
""")

col1, col2, col3 = st.columns(3)
with col1: st.image("IMG_4815.jpg", caption="Roll-forming machine assembly")
with col2: st.image("IMG_4853.jpg", caption="Gearbox integration")
with col3: st.image("IMG_4865.jpg", caption="Hydraulic system assembly")

with st.expander("**Key Outputs**"):
    st.markdown("""
    - **Assembled** roll-forming machines, gearboxes, and hydraulic subsystems on the shop floor  
    - **Collaborated with technicians** to understand real-world tolerances and manufacturing processes  
    - **Interpreted technical drawings** and BOMs to ensure accurate assembly  
    - **Proposed design-for-assembly improvements** to reduce complexity and time  
    - Gained practical understanding of how **design decisions affect manufacturability and maintenance**
    """)

# ---- DESIGN & OPTIMISATION ----
st.subheader("Design Engineering & Optimisation (Aug 2025)")
st.markdown("""
Transitioned into the design office, contributing to both **failure diagnosis** on existing equipment and **new product development**.
""")

with st.expander("**Key Outputs**"):
    st.markdown("""
    - **Failure Diagnosis**: Investigated recurring saw-blade failures; identified misalignment causing lateral forces → corrective redesign prevented downtime and cost  
    - **Machine Development**: Supported design of a new roll-forming machine by running **stress calculations**, conducting **material testing**, and producing **CAD models** of key components  
    - **Documentation & Traceability**: Digitised technical drawings and maintenance guides, improving data accessibility  
    - **System Design**: Produced a new technical drawing for the lubrication system of a tube-forming machine
    """)

st.image("IMG_4888.jpg", caption="Design contribution — lubrication system schematic")

# ---- REFLECTION / VALUE ----
st.subheader("Experience Value")
st.info("""
This experience gave me:  
- First-hand exposure to **how machines are built, tested, and improved** in industry  
- Confidence in bridging **theoretical calculations with practical design and assembly**  
- A deeper appreciation of **design-for-manufacture, reliability, and maintainability** principles  
- Stronger collaboration skills working across technicians, engineers, and designers
""")
