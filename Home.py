import streamlit as st
from datetime import date

st.set_page_config(page_title="Davi Ruschel — Portfolio", layout="wide")

# ---------- small helpers ----------
def chip(text):  # simple inline tag
    st.markdown(f"<span style='padding:4px 8px;border:1px solid #ddd;border-radius:999px;font-size:0.85rem;'>{text}</span>",
                unsafe_allow_html=True)

def card(title, blurb, link_text, link_url):
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(blurb)
        st.link_button(link_text, link_url)

# ---------- HERO ----------
left, right = st.columns([1.2, 0.8])
with left:
    st.title("Davi Ruschel")
    st.markdown(
        "Engineer focused on **simulation, optimisation, and data-driven tooling**. "
        "I build models and apps that turn raw data into decisions."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.link_button("📄 CV", "https://your-cv-link.pdf")
    with c2: st.link_button("💼 LinkedIn", "https://linkedin.com/in/your-handle")
    with c3: st.link_button("🧪 Projects", "#projects")
    with c4: st.link_button("✉️ Email", "mailto:you@email.com")

with right:
    st.write("**Focus areas**")
    colA, colB, colC = st.columns(3)
    with colA: chip("Simulation")
    with colB: chip("Optimisation")
    with colC: chip("Data Apps")
    st.write("")
    colA, colB, colC = st.columns(3)
    with colA: chip("Python")
    with colB: chip("MATLAB")
    with colC: chip("CAD")

st.divider()

# ---------- METRICS (neutral + concise) ----------
m1, m2, m3 = st.columns(3)
with m1: st.metric("Year 3 Exam Avg", "79%")
with m2: st.metric("Technical Report", "76%")
with m3: st.metric("Interactive Tools Shipped", "4+")

# ---------- ABOUT ----------
st.subheader("About")
st.markdown(
    "Final-year **MEng Mechanical Engineering** student (University of Bath), on track for **First Class**. "
    "My work blends analytical modelling with practical implementation: simulators, telemetry/data apps, and clean technical documentation."
)

# ---------- SKILLS (two compact expanders) ----------
cL, cR = st.columns(2)
with cL:
    with st.expander("Technical"):
        st.markdown(
            "- **Programming & Data**: Python (pandas, NumPy, matplotlib, scikit-learn), MATLAB  \n"
            "- **Modelling & Optimisation**: dynamic systems, Monte Carlo / sensitivity, control basics  \n"
            "- **Dashboards & Apps**: Streamlit (interactive tools, telemetry-style UIs)  \n"
            "- **Design**: Autodesk Inventor, AutoCAD; documentation & BOM literacy"
        )
with cR:
    with st.expander("Transferable"):
        st.markdown(
            "- Project leadership & delivery  \n"
            "- Decisions under time constraints  \n"
            "- Clear communication (reports, presentations)  \n"
            "- Entrepreneurship (founded a profitable events brand)  \n"
            "- Languages: Portuguese, English, Spanish"
        )

st.divider()

# ---------- PROJECTS (cards) ----------
st.subheader("Projects")
st.markdown("<a name='projects'></a>", unsafe_allow_html=True)

row1 = st.columns(3)
with row1[0]:
    card(
        "Interactive Lap Simulation",
        "Monte Carlo–driven simulator to evaluate strategy trade-offs under constraints. "
        "Includes a fully interactive demo with parameter inputs and telemetry plots.",
        "Open",
        "/Strategy_Simulation"
    )
with row1[1]:
    card(
        "Real-Time Data Platform",
        "Data acquisition + visualisation app for live monitoring and post-run analysis. "
        "Improves decision speed and quality during testing.",
        "Open",
        "/Green_Bath_Racing"
    )
with row1[2]:
    card(
        "Industrial Internship",
        "Hands-on assembly and design work: failure diagnosis, stress calcs, material testing, "
        "and CAD for production tooling.",
        "Open",
        "/Zikeli_Internship"
    )

row2 = st.columns(3)
with row2[0]:
    card(
        "Data Dashboard",
        "Exploratory analytics with custom visuals and scenario views. Built for fast insight and what-if checks.",
        "Open",
        "/F1_Strategy_Project"
    )
with row2[1]:
    card(
        "Additional Projects",
        "A small collection of smaller tools and experiments (controls, visualisation, scripting).",
        "Browse",
        "/Additional_Projects"
    )
with row2[2]:
    st.empty()

st.divider()

# ---------- STUDIES ----------
st.subheader("Studies")
st.markdown(
    "- **Final-Year Modules**: System Modelling & Simulation, CFD, Composites, Electric Propulsion, Aerodynamics  \n"
    "- Academic work emphasises model validation, sensitivity, and translating results into clear recommendations."
)

# ---------- FOOTER ----------
st.divider()
f1, f2 = st.columns([0.7, 0.3])
with f1:
    st.markdown("**Contact** · [LinkedIn](https://linkedin.com/in/your-handle) · [Email](mailto:you@email.com)")
with f2:
    st.caption(f"Last updated: {date.today().isoformat()}")
