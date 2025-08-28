import streamlit as st
from datetime import date

st.set_page_config(page_title="Davi Ruschel — Portfolio", layout="wide")

# ---------- small helpers ----------
def chip(text):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:4px 10px;
            border:1px solid #e5e7eb;
            border-radius:999px;
            background:#f8fafc;
            font-size:0.85rem;">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def card(title, blurb, link_text, link_url):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(blurb)
        st.link_button(link_text, link_url)

# =======================================================================
# HERO
# =======================================================================
col1, col2 = st.columns([1.25, 1], vertical_alignment="center")

with col1:
    st.title("Davi Ruschel")
    st.markdown(
        "Engineer focused on **simulation, optimisation, and data-driven tooling**. "
        "I build models and apps that turn raw data into decisions."
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1: st.link_button("📄 CV", "https://your-cv-link.pdf")
    with b2: st.link_button("💼 LinkedIn", "https://linkedin.com/in/your-handle")
    with b3: st.link_button("🧪 Projects", "#projects")
    with b4: st.link_button("✉️ Email", "mailto:you@email.com")

with col2:
    st.image(
        "GBR picture (48).jpeg",
        caption="Engineering project in competition environment",
        use_container_width=True,
    )
    st.write("**Focus areas**")
    r1 = st.columns(3)
    with r1[0]: chip("Simulation")
    with r1[1]: chip("Optimisation")
    with r1[2]: chip("Data Apps")
    r2 = st.columns(3)
    with r2[0]: chip("Python")
    with r2[1]: chip("MATLAB")
    with r2[2]: chip("CAD")

st.divider()

# =======================================================================
# METRICS
# =======================================================================
m1, m2, m3 = st.columns(3)
with m1: st.metric("Year 3 Exam Avg", "79%")
with m2: st.metric("Technical Report", "76%")
with m3: st.metric("Interactive Tools Shipped", "4+")

# =======================================================================
# ABOUT
# =======================================================================
st.subheader("About")
st.markdown(
    "Final-year **MEng Mechanical Engineering** student (University of Bath), on track for **First Class**. "
    "My work blends analytical modelling with practical implementation: simulators, telemetry/data apps, "
    "and clear technical documentation."
)

# =======================================================================
# SKILLS (compact)
# =======================================================================
s1, s2 = st.columns(2)
with s1:
    with st.expander("Technical"):
        st.markdown(
            "- **Programming & Data**: Python (pandas, NumPy, matplotlib, scikit-learn), MATLAB  \n"
            "- **Modelling & Optimisation**: dynamic systems, Monte Carlo/sensitivity, control basics  \n"
            "- **Dashboards & Apps**: Streamlit (interactive tools, telemetry-style UIs)  \n"
            "- **Design**: Autodesk Inventor, AutoCAD; documentation & BOM literacy"
        )
with s2:
    with st.expander("Transferable"):
        st.markdown(
            "- Project leadership & delivery  \n"
            "- Decisions under time constraints  \n"
            "- Clear communication (reports, presentations)  \n"
            "- Entrepreneurship (founded a profitable events brand)  \n"
            "- Languages: Portuguese, English, Spanish"
        )

st.divider()

# =======================================================================
# PROJECTS (cards)
# =======================================================================
st.subheader("Projects")
st.markdown("<a name='projects'></a>", unsafe_allow_html=True)

row1 = st.columns(3)
with row1[0]:
    card(
        "Interactive Lap Simulation",
        "Monte Carlo–driven simulator to evaluate strategy trade-offs under constraints. "
        "Includes an interactive demo with parameter inputs and telemetry plots.",
        "Open",
        "/Strategy_Simulation",
    )
with row1[1]:
    card(
        "Real-Time Data Platform",
        "Data acquisition + visualisation app for live monitoring and post-run analysis. "
        "Improves decision speed and quality during testing.",
        "Open",
        "/Green_Bath_Racing",
    )
with row1[2]:
    card(
        "Industrial Internship",
        "Hands-on assembly and design work: failure diagnosis, stress calcs, material testing, "
        "and CAD for production tooling.",
        "Open",
        "/Zikeli_Internship",
    )

row2 = st.columns(3)
with row2[0]:
    card(
        "Data Dashboard",
        "Exploratory analytics with custom visuals and scenario views. Built for fast insight and what-if checks.",
        "Open",
        "/F1_Strategy_Project",
    )
with row2[1]:
    card(
        "Additional Projects",
        "A collection of smaller tools and experiments (controls, visualisation, scripting).",
        "Browse",
        "/Additional_Projects",
    )
with row2[2]:
    st.empty()

st.divider()

# =======================================================================
# STUDIES
# =======================================================================
st.subheader("Studies")
st.markdown(
    "- **Final-Year Modules**: System Modelling & Simulation, CFD, Composites, Electric Propulsion, Aerodynamics  \n"
    "- Academic work emphasises model validation, sensitivity, and translating results into clear recommendations."
)

# =======================================================================
# FOOTER
# =======================================================================
st.divider()
f1, f2 = st.columns([0.7, 0.3])
with f1:
    st.markdown("**Contact** · [LinkedIn](https://linkedin.com/in/your-handle) · [Email](mailto:you@email.com)")
with f2:
    st.caption(f"Last updated: {date.today().isoformat()}")
