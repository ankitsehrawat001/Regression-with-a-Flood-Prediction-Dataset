from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AquaSignal | Flood Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_regression_model.joblib"
SCALER_PATH = BASE_DIR / "minmax_scaler.pkl"

FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors",
]

FEATURE_GROUPS = {
    "Climate & terrain": ["MonsoonIntensity", "TopographyDrainage", "ClimateChange", "Landslides", "Watersheds"],
    "Infrastructure": ["RiverManagement", "DamsQuality", "DrainageSystems", "DeterioratingInfrastructure"],
    "Human pressure": ["Deforestation", "Urbanization", "AgriculturalPractices", "Encroachments", "PopulationScore"],
    "Preparedness": ["Siltation", "IneffectiveDisasterPreparedness", "CoastalVulnerability", "WetlandLoss", "InadequatePlanning", "PoliticalFactors"],
}

# Fallback used when the original notebook scaler was not exported.
FALLBACK_MAX_VALUES = [16, 18, 16, 17, 17, 17, 16, 16, 16, 16, 16, 17, 16, 16, 16, 17, 18, 19, 16, 16]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


def scale_inputs(inputs: pd.DataFrame, scaler) -> pd.DataFrame:
    if scaler is not None:
        return pd.DataFrame(scaler.transform(inputs), columns=FEATURES)
    return inputs.div(FALLBACK_MAX_VALUES)


def risk_label(probability: float) -> tuple[str, str]:
    if probability < 0.45:
        return "Lower projected risk", "low"
    if probability < 0.55:
        return "Moderate projected risk", "medium"
    return "Higher projected risk", "high"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#eef6f4; --muted:#8da7aa; --line:rgba(163,210,208,.16); --aqua:#67e0d1; --blue:#78a9ff; --bg:#071113; }
    html, body, [class*="css"] { font-family:'Manrope', sans-serif; }
    .stApp { color:var(--ink); background:var(--bg); background-image:radial-gradient(circle at 82% 0%, rgba(34,120,113,.23), transparent 28%), radial-gradient(circle at 4% 44%, rgba(40,78,145,.14), transparent 27%), linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px); background-size:auto, auto, 72px 72px, 72px 72px; }
    .stApp:before { content:''; position:fixed; inset:0; pointer-events:none; opacity:.22; background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.16'/%3E%3C/svg%3E"); z-index:0; }
    .block-container { max-width:1280px; padding:1.25rem 3.5rem 4rem; position:relative; z-index:1; }
    h1,h2,h3 { font-family:'Manrope', sans-serif !important; color:var(--ink) !important; letter-spacing:-.045em; }
    h2 { font-size:1.45rem !important; margin-top:2rem !important; }
    .topbar { display:flex; justify-content:space-between; align-items:center; padding:0 0 2.4rem; border-bottom:1px solid var(--line); }
    .brand { display:flex; gap:.75rem; align-items:center; font-weight:800; letter-spacing:-.03em; font-size:1.05rem; }
    .brand-mark { display:grid; place-items:center; width:30px; height:30px; border:1px solid rgba(103,224,209,.6); border-radius:9px; color:var(--aqua); font-family:'DM Mono', monospace; box-shadow:0 0 24px rgba(103,224,209,.16); }
    .top-meta { display:flex; align-items:center; gap:1rem; color:var(--muted); font:500 .68rem 'DM Mono', monospace; letter-spacing:.09em; text-transform:uppercase; }
    .live-dot { width:7px; height:7px; border-radius:50%; background:var(--aqua); box-shadow:0 0 0 5px rgba(103,224,209,.12), 0 0 18px var(--aqua); animation:pulse 2s ease-in-out infinite; }
    .hero { padding:5.2rem 0 4rem; max-width:900px; animation:rise .9s cubic-bezier(.16,1,.3,1) both; }
    .eyebrow { color:var(--aqua); font:500 .7rem 'DM Mono', monospace; letter-spacing:.16em; text-transform:uppercase; }
    .hero h1 { font-size:clamp(3.2rem, 8vw, 7.4rem) !important; line-height:.94; margin:.9rem 0 1.45rem !important; max-width:900px; }
    .hero h1 span { color:transparent; background:linear-gradient(105deg, #effaf7 20%, #84e3d6 58%, #78a9ff 100%); background-clip:text; -webkit-background-clip:text; }
    .hero-copy { color:var(--muted); font-size:1.04rem; line-height:1.75; max-width:580px; }
    .hero-index { display:flex; gap:2.4rem; margin-top:2.4rem; padding-top:1.2rem; border-top:1px solid var(--line); width:min(100%, 620px); }
    .index-item b { display:block; color:var(--ink); font:500 .9rem 'DM Mono', monospace; } .index-item small { color:var(--muted); font-size:.7rem; }
    .section-kicker { color:var(--aqua); font:500 .65rem 'DM Mono', monospace; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.4rem; }
    .panel { background:linear-gradient(145deg, rgba(18,39,41,.78), rgba(8,21,23,.72)); border:1px solid var(--line); border-radius:18px; padding:1.5rem; box-shadow:0 22px 70px rgba(0,0,0,.2); }
    .panel-title { font-weight:700; font-size:1.1rem; color:var(--ink); margin-bottom:.35rem; } .panel-copy { color:var(--muted); font-size:.8rem; line-height:1.55; }
    .metric { background:rgba(13,31,33,.72); border:1px solid var(--line); border-radius:14px; padding:1.15rem; min-height:100px; transition:transform .3s ease, border-color .3s ease; } .metric:hover { transform:translateY(-4px); border-color:rgba(103,224,209,.5); }
    .metric-label { color:var(--muted); font:500 .63rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.08em; } .metric-value { color:var(--ink); font-size:1.7rem; font-weight:800; margin-top:.35rem; letter-spacing:-.05em; } .metric-value.aqua { color:var(--aqua); }
    .result-panel { border:1px solid rgba(103,224,209,.28); background:radial-gradient(circle at 92% 5%, rgba(103,224,209,.18), transparent 32%), linear-gradient(145deg, rgba(14,52,52,.92), rgba(8,24,27,.94)); border-radius:18px; padding:1.6rem; }
    .result-label { color:var(--aqua); font:500 .65rem 'DM Mono', monospace; text-transform:uppercase; letter-spacing:.13em; } .result-value { font-size:clamp(3.2rem, 7vw, 5.5rem); line-height:1; font-weight:800; letter-spacing:-.08em; margin:.4rem 0 .55rem; } .result-note { color:#a9c1c1; font-size:.78rem; }
    .signal { display:inline-flex; align-items:center; gap:.45rem; border:1px solid rgba(103,224,209,.26); border-radius:999px; padding:.4rem .65rem; color:var(--aqua); font:500 .67rem 'DM Mono', monospace; margin-top:1.2rem; } .signal:before { content:''; width:6px; height:6px; border-radius:50%; background:var(--aqua); }
    .footer-line { margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--line); color:#587174; font:500 .65rem 'DM Mono', monospace; display:flex; justify-content:space-between; }
    [data-testid="stSlider"] label { color:#bdd0d0 !important; font-size:.78rem !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--aqua); border-color:var(--aqua); }
    .stButton > button, button[kind="primary"] { background:linear-gradient(105deg, #67e0d1, #78a9ff) !important; color:#071113 !important; border:0 !important; border-radius:9px !important; font-weight:800 !important; min-height:3rem; transition:transform .25s ease, box-shadow .25s ease !important; } .stButton > button:hover { transform:translateY(-2px); box-shadow:0 10px 30px rgba(103,224,209,.2) !important; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .stProgress > div > div > div > div { background:linear-gradient(90deg, #67e0d1, #78a9ff); }
    @keyframes rise { from { opacity:0; transform:translateY(26px); } to { opacity:1; transform:none; } } @keyframes pulse { 50% { opacity:.45; transform:scale(.75); } }
    @media (prefers-reduced-motion:reduce) { *, *:before, *:after { animation:none !important; transition:none !important; } }
    @media (max-width:760px) { .block-container { padding:1rem 1.15rem 3rem; } .topbar { padding-bottom:1.5rem; } .top-meta { font-size:.56rem; } .hero { padding:3.6rem 0 2.7rem; } .hero h1 { font-size:clamp(3rem, 16vw, 5rem) !important; } .hero-index { gap:1.1rem; flex-wrap:wrap; } .footer-line { display:block; line-height:2; } }
    </style>
    <div class="topbar"><div class="brand"><span class="brand-mark">A</span> AQUASIGNAL</div><div class="top-meta"><span class="live-dot"></span> MODEL ONLINE <span>v1.0 / REGRESSION</span></div></div>
    <div class="hero"><div class="eyebrow">Flood intelligence / scenario engine</div><h1>Read the <span>water</span> before it rises.</h1><div class="hero-copy">A focused prediction workspace for exploring how climate, terrain, infrastructure and human pressure combine to shape flood probability.</div><div class="hero-index"><div class="index-item"><b>20</b><small>risk indicators</small></div><div class="index-item"><b>1.1M</b><small>training rows</small></div><div class="index-item"><b>LIVE</b><small>local inference</small></div></div></div>
    """,
    unsafe_allow_html=True,
)

try:
    model, scaler = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model artifacts: {error}")
    st.stop()

st.markdown('<div class="section-kicker">01 / Build a scenario</div><div class="panel-title">Tune the signals</div><div class="panel-copy">Set each indicator from 0 to 10. A higher score represents greater exposure or vulnerability.</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    values = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        st.markdown(f'<div class="section-kicker" style="margin-top:1.35rem">{group_name}</div>', unsafe_allow_html=True)
        group_columns = st.columns(3 if len(group_features) >= 5 else 2)
        for index, feature in enumerate(group_features):
            with group_columns[index % len(group_columns)]:
                values[feature] = st.slider(feature, 0, 10, 5, key=feature)
    submitted = st.form_submit_button("Run flood projection  ->", width="stretch")

if submitted:
    raw_inputs = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    scaled_inputs = scale_inputs(raw_inputs, scaler)
    probability = float(model.predict(scaled_inputs)[0])
    probability = max(0.0, min(1.0, probability))
    label, _ = risk_label(probability)
    summary = pd.DataFrame({"Factor": FEATURES, "Score": [values[feature] for feature in FEATURES]})
    average_score = summary["Score"].mean()
    highest_factor = summary.loc[summary["Score"].idxmax(), "Factor"]

    st.markdown('<div class="section-kicker" style="margin-top:3.5rem">02 / Projection output</div>', unsafe_allow_html=True)
    result_columns = st.columns([1.35, 1, 1, 1])
    with result_columns[0]:
        st.markdown(f'<div class="result-panel"><div class="result-label">Flood probability</div><div class="result-value">{probability:.1%}</div><div class="result-note">{label}. Statistical estimate based on the configured scenario.</div><div class="signal">INFERENCE COMPLETE</div></div>', unsafe_allow_html=True)
    with result_columns[1]:
        st.markdown(f'<div class="metric"><div class="metric-label">Scenario average</div><div class="metric-value">{average_score:.1f}<span style="font-size:.8rem;color:#8da7aa"> / 10</span></div><div class="panel-copy">across all signals</div></div>', unsafe_allow_html=True)
    with result_columns[2]:
        st.markdown(f'<div class="metric"><div class="metric-label">Peak signal</div><div class="metric-value aqua">{values[highest_factor]}</div><div class="panel-copy">{highest_factor}</div></div>', unsafe_allow_html=True)
    with result_columns[3]:
        st.markdown(f'<div class="metric"><div class="metric-label">Model inputs</div><div class="metric-value">{len(FEATURES)}</div><div class="panel-copy">active indicators</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker" style="margin-top:2.5rem">03 / Explore the signals</div>', unsafe_allow_html=True)
    chart_columns = st.columns([1, 1.8])
    with chart_columns[0]:
        st.markdown('<div class="panel"><div class="panel-title">Probability scale</div><div class="panel-copy">The model output mapped from 0 to 100 percent.</div></div>', unsafe_allow_html=True)
        st.progress(probability, text=f"{probability:.1%} projected probability")
        st.markdown('<div class="panel-copy">Lower estimate &nbsp; 0% &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 100% &nbsp; Higher estimate</div>', unsafe_allow_html=True)
    with chart_columns[1]:
        st.markdown('<div class="panel-title">Signal intensity map</div><div class="panel-copy">Compare the relative weight of each configured scenario factor.</div>', unsafe_allow_html=True)
        st.bar_chart(summary.set_index("Factor"), y="Score", horizontal=True, height=480, color="#67e0d1")

    st.markdown('<div class="section-kicker" style="margin-top:2.5rem">04 / Scenario record</div>', unsafe_allow_html=True)
    st.dataframe(summary, hide_index=True, width="stretch")
else:
    st.markdown('<div class="panel" style="margin-top:2rem"><div class="panel-title">Your projection is waiting</div><div class="panel-copy">Configure the signals above and run the model to reveal a probability estimate, intensity map and scenario record.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer-line"><span>AQUASIGNAL / FLOOD INTELLIGENCE</span><span>LOCAL MODEL · FOR EXPLORATION ONLY</span></div>', unsafe_allow_html=True)
