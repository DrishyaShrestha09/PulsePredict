import streamlit as st
import numpy as np
import pickle as pkl

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulsePredict — Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0f1a;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 760px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e2d45;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #e63946 0%, #c1121f 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.app-title-block {}
.app-title {
    font-size: 22px;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.3px;
    margin: 0;
}
.app-subtitle {
    font-size: 13px;
    color: #64748b;
    margin: 2px 0 0;
}

/* ── Section headers ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e63946;
    margin: 1.8rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2d45;
}

/* ── Input fields ── */
.stNumberInput > label,
.stSelectbox > label,
.stRadio > label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin-bottom: 4px !important;
}

.stNumberInput input {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
    transition: border-color 0.2s;
}
.stNumberInput input:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.12) !important;
}

.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
.stSelectbox > div > div:hover {
    border-color: #334155 !important;
}

/* ── Radio (model selector) ── */
.stRadio > div {
    display: flex;
    gap: 10px;
    margin-top: 6px;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    color: #94a3b8 !important;
}

/* ── Predict button ── */
.stButton > button {
    background: #e63946 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #c1121f !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(230, 57, 70, 0.3) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Clear button (second button) ── */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid #1e2d45 !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #111827 !important;
    color: #94a3b8 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Result boxes ── */
.stAlert {
    border-radius: 10px !important;
    border-width: 1px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: #1e2d45 !important;
    margin: 1.5rem 0 !important;
}

/* ── Help tooltip ── */
.stTooltipIcon { color: #334155 !important; }

/* ── Mono numbers in inputs ── */
input[type="number"] {
    font-family: 'DM Mono', monospace !important;
}
</style>

<div class="app-header">
    <div class="app-logo">🫀</div>
    <div class="app-title-block">
        <p class="app-title">PulsePredict — Heart Disease Predictor</p>
        <p class="app-subtitle">Enter patient health data to assess the risk of heart disease using predictive analytics.</p>
    </div>
</div>
""", unsafe_allow_html=True)

def clear_all():
    for k in ["age","sex","cp","trestbps","chol","fbs","restecg",
              "thalach","exang","oldpeak","slope","ca","thal","model_choice"]:
        if k in st.session_state:
            del st.session_state[k]

# loading models

@st.cache_resource
def load_models():
    with open("logistic_regression_model.pkl", "rb") as f:
        lr = pkl.load(f)
    with open("random_forest_model.pkl", "rb") as f:
        rfc = pkl.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm = pkl.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pkl.load(f)
    return lr, rfc, svm, scaler

lr, rfc, svm, scaler = load_models()

# patient info
st.markdown('<div class="section-label">Patient info</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=20, max_value=80, value=50, key="age",
                          help="Patient age in years (20–80)")
with col2:
    sex = st.selectbox("Sex", [0, 1],
                       format_func=lambda x: "Female" if x == 0 else "Male",
                       key="sex")

# clinical measurements
st.markdown('<div class="section-label">Clinical measurements</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    trestbps = st.number_input("BP (mmHg)", min_value=80, max_value=200, value=120, key="trestbps",
                               help="Resting blood pressure")
with col2:
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200, key="chol",
                           help="Serum cholesterol in mg/dL")
with col3:
    thalach = st.number_input("Max HR", min_value=60, max_value=220, value=150, key="thalach",
                              help="Maximum heart rate achieved")
with col4:
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0,
                              step=0.1, key="oldpeak",
                              help="ST depression induced by exercise relative to rest")

# diagnostic indicators
st.markdown('<div class="section-label">Diagnostic indicators</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "Typical angina",
                          1: "Atypical angina",
                          2: "Non-anginal pain",
                          3: "Asymptomatic"
                      }[x], key="cp")
    fbs = st.selectbox("Fasting blood sugar >120?", [0, 1],
                       format_func=lambda x: "No — normal" if x == 0 else "Yes — elevated",
                       key="fbs")
with col2:
    restecg = st.selectbox("Resting ECG", [0, 1, 2],
                           format_func=lambda x: {
                               0: "Normal",
                               1: "ST-T wave abnormality",
                               2: "LV hypertrophy"
                           }[x], key="restecg")
    exang = st.selectbox("Exercise angina", [0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes — present",
                         key="exang")
with col3:
    slope = st.selectbox("ST slope", [0, 1, 2],
                         format_func=lambda x: {
                             0: "Upsloping",
                             1: "Flat",
                             2: "Downsloping"
                         }[x], key="slope")
    ca = st.selectbox("Major vessels", [0, 1, 2, 3],
                      format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''} colored",
                      key="ca")

thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "Fixed defect",
                        2: "Reversible defect",
                        3: "Unknown"
                    }[x], key="thal")

# model selection
st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "Choose prediction model",
    ["Logistic Regression", "Random Forest", "SVM"],
    horizontal=True,
    key="model_choice"
)

st.markdown("<br>", unsafe_allow_html=True)

# buttons
btn_col1, btn_col2 = st.columns([5, 1])
with btn_col1:
    predict = st.button("Run prediction", use_container_width=True)
with btn_col2:
    st.button("Clear", use_container_width=True, on_click=clear_all)

# result display
if predict:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    model = {"Logistic Regression": lr, "Random Forest": rfc, "SVM": svm}[model_choice]
    pred = model.predict(input_scaled)[0]

    proba_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1] * 100
        proba_text = f" · Confidence: **{proba:.1f}%**"

    st.markdown("<br>", unsafe_allow_html=True)
    if pred == 1:
        st.error(
            f"**⚠ High risk detected{proba_text}**\n\n"
            "Patient likely has heart disease. "
            "Recommend further clinical evaluation and specialist referral."
        )
    else:
        st.success(
            f"**✓ Low risk detected{proba_text}**\n\n"
            "No heart disease detected based on current input values. "
            "Continue routine monitoring."
        )

    st.markdown(
        f"<p style='font-size:11px; color:#334155; margin-top:8px; font-family:DM Mono,monospace;'>"
        f"Model: {model_choice} · Features: 13 · Dataset: Cleveland Heart Disease</p>",
        unsafe_allow_html=True
    )