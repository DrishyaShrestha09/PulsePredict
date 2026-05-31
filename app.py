import streamlit as st
import numpy as np
import pickle as pkl
import plotly.graph_objects as go

#  Page config 
st.set_page_config(
    page_title="PulsePredict — Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

#  defigning custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 780px; }

.app-header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e2d45;
}
.app-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #e63946 0%, #c1121f 100%);
    border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 24px; flex-shrink: 0;
}
.app-title { font-size: 22px; font-weight: 600; color: #f1f5f9; letter-spacing: -0.3px; margin: 0; }
.app-subtitle { font-size: 13px; color: #64748b; margin: 2px 0 0; }

.section-label {
    font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase; color: #e63946;
    margin: 1.8rem 0 0.8rem; display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: #1e2d45; }

.stNumberInput > label, .stSelectbox > label, .stRadio > label {
    font-size: 12px !important; font-weight: 500 !important;
    color: #94a3b8 !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; margin-bottom: 4px !important;
}
.stNumberInput input {
    background: #111827 !important; border: 1px solid #1e2d45 !important;
    border-radius: 8px !important; color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important; font-size: 15px !important;
}
.stNumberInput input:focus {
    border-color: #e63946 !important;
    box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.12) !important;
}
.stSelectbox > div > div {
    background: #111827 !important; border: 1px solid #1e2d45 !important;
    border-radius: 8px !important; color: #f1f5f9 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important;
    border: 1px solid #1e2d45 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px !important;
    color: #64748b !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e2d45 !important; color: #f1f5f9 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

/* Metric cards */
.metric-card {
    background: #111827; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { font-family: 'DM Mono', monospace; font-size: 22px; font-weight: 500; color: #f1f5f9; }
.metric-sub { font-size: 11px; color: #334155; margin-top: 3px; }

.stButton > button {
    background: #e63946 !important; color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.65rem 1.5rem !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #c1121f !important; transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(230, 57, 70, 0.3) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: transparent !important; color: #64748b !important;
    border: 1px solid #1e2d45 !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: #111827 !important; color: #94a3b8 !important;
    transform: none !important; box-shadow: none !important;
}
.stAlert { border-radius: 10px !important; border-width: 1px !important; }
input[type="number"] { font-family: 'DM Mono', monospace !important; }
</style>

<div class="app-header">
    <div class="app-logo">🫀</div>
    <div class="app-title-block">
        <p class="app-title">PulsePredict — Heart Disease Predictor</p>
        <p class="app-subtitle">Enter patient health data to assess the risk of heart disease using predictive analytics.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# helper function to clear session state 
def clear_all():
    for k in ["age","sex","cp","trestbps","chol","fbs","restecg",
              "thalach","exang","oldpeak","slope","ca","thal","model_choice"]:
        if k in st.session_state:
            del st.session_state[k]

PLOTLY_TRANSPARENT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Resting BP",
    "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
    "Max Heart Rate", "Exercise Angina", "ST Depression",
    "ST Slope", "Major Vessels", "Thalassemia"
]

# Load models 
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

# tabs
tab_predict, tab_compare = st.tabs(["🔍  Predict", "📊  Model comparison"])

# prediction tab
with tab_predict:

    # Patient info
    st.markdown('<div class="section-label">Patient info</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=80, value=50, key="age",
                              help="Patient age in years (20–80)")
    with col2:
        sex = st.selectbox("Sex", [0, 1],
                           format_func=lambda x: "Female" if x == 0 else "Male", key="sex")

    # Clinical measurements
    st.markdown('<div class="section-label">Clinical measurements</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trestbps = st.number_input("BP (mmHg)", min_value=80, max_value=200, value=120,
                                   key="trestbps", help="Resting blood pressure")
    with col2:
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200,
                               key="chol", help="Serum cholesterol in mg/dL")
    with col3:
        thalach = st.number_input("Max HR", min_value=60, max_value=220, value=150,
                                  key="thalach", help="Maximum heart rate achieved")
    with col4:
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0,
                                  step=0.1, key="oldpeak",
                                  help="ST depression induced by exercise relative to rest")

    # Diagnostic indicators
    st.markdown('<div class="section-label">Diagnostic indicators</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        cp = st.selectbox("Chest pain type", [0, 1, 2, 3],
                          format_func=lambda x: {
                              0: "Typical angina", 1: "Atypical angina",
                              2: "Non-anginal pain", 3: "Asymptomatic"
                          }[x], key="cp")
        fbs = st.selectbox("Fasting blood sugar >120?", [0, 1],
                           format_func=lambda x: "No — normal" if x == 0 else "Yes — elevated",
                           key="fbs")
    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                               format_func=lambda x: {
                                   0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"
                               }[x], key="restecg")
        exang = st.selectbox("Exercise angina", [0, 1],
                             format_func=lambda x: "No" if x == 0 else "Yes — present",
                             key="exang")
    with col3:
        slope = st.selectbox("ST slope", [0, 1, 2],
                             format_func=lambda x: {
                                 0: "Upsloping", 1: "Flat", 2: "Downsloping"
                             }[x], key="slope")
        ca = st.selectbox("Major vessels", [0, 1, 2, 3],
                          format_func=lambda x: f"{x} vessel{'s' if x != 1 else ''} colored",
                          key="ca")

    thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                        format_func=lambda x: {
                            0: "Normal", 1: "Fixed defect",
                            2: "Reversible defect", 3: "Unknown"
                        }[x], key="thal")

    # Model
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Choose prediction model",
        ["Logistic Regression", "Random Forest", "SVM"],
        horizontal=True, key="model_choice"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Buttons
    btn_col1, btn_col2 = st.columns([5, 1])
    with btn_col1:
        predict = st.button("Run prediction", use_container_width=True)
    with btn_col2:
        st.button("Clear", use_container_width=True, on_click=clear_all)

    # Result 
    if predict:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)

        model = {"Logistic Regression": lr, "Random Forest": rfc, "SVM": svm}[model_choice]
        pred  = model.predict(input_scaled)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            risk_pct = model.predict_proba(input_scaled)[0][1] * 100
        else:
            risk_pct = 85.0 if pred == 1 else 15.0

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        needle_color = "#e63946" if pred == 1 else "#22c55e"
        gauge_color  = "#e63946" if pred == 1 else "#22c55e"

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#f1f5f9", "family": "DM Mono"}},
            title={"text": "Risk level", "font": {"size": 13, "color": "#64748b", "family": "DM Sans"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#1e2d45",
                    "tickfont": {"color": "#334155", "size": 10},
                    "nticks": 6,
                },
                "bar": {"color": gauge_color, "thickness": 0.25},
                "bgcolor": "#111827",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  33], "color": "#0d1f14"},
                    {"range": [33, 66], "color": "#1a1a0d"},
                    {"range": [66,100], "color": "#1f0d0d"},
                ],
                "threshold": {
                    "line": {"color": needle_color, "width": 3},
                    "thickness": 0.85,
                    "value": risk_pct,
                },
            }
        ))
        gauge_fig.update_layout(
            **PLOTLY_TRANSPARENT,
            height=220,
            margin=dict(t=40, b=10, l=30, r=30),
            font={"family": "DM Sans"},
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # textual result
        if pred == 1:
            st.error(
                f"**⚠ High risk detected · {risk_pct:.1f}% confidence**\n\n"
                "Patient likely has heart disease. "
                "Recommend further clinical evaluation and specialist referral."
            )
        else:
            st.success(
                f"**✓ Low risk detected · {risk_pct:.1f}% confidence**\n\n"
                "No heart disease detected based on current input values. "
                "Continue routine monitoring."
            )

        # Feature importance of Random Forest model
        if model_choice == "Random Forest":
            st.markdown('<div class="section-label">Feature importance</div>',
                        unsafe_allow_html=True)

            importances = rfc.feature_importances_
            sorted_idx  = np.argsort(importances)
            sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
            sorted_vals  = importances[sorted_idx]

            bar_colors = [
                "#e63946" if v >= sorted_vals[-3] else
                "#f4a261" if v >= sorted_vals[-6] else
                "#334155"
                for v in sorted_vals
            ]

            fi_fig = go.Figure(go.Bar(
                x=sorted_vals,
                y=sorted_names,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            fi_fig.update_layout(
                **PLOTLY_TRANSPARENT,
                height=380,
                margin=dict(t=10, b=10, l=10, r=20),
                xaxis=dict(
                    showgrid=True, gridcolor="#1e2d45",
                    tickfont=dict(color="#64748b", size=10, family="DM Mono"),
                    zeroline=False,
                ),
                yaxis=dict(
                    tickfont=dict(color="#94a3b8", size=12, family="DM Sans"),
                    showgrid=False,
                ),
                font=dict(family="DM Sans"),
            )
            st.plotly_chart(fi_fig, use_container_width=True)
            st.markdown(
                "<p style='font-size:11px;color:#334155;font-family:DM Mono,monospace'>"
                "Red = top 3 most influential · Orange = mid-tier · Gray = lower impact</p>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<p style='font-size:11px;color:#334155;margin-top:8px;font-family:DM Mono,monospace'>"
            f"Model: {model_choice} · Features: 13 · Dataset: Cleveland Heart Disease</p>",
            unsafe_allow_html=True
        )



# Model comparison tab

with tab_compare:
    st.markdown('<div class="section-label">Performance overview</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:13px;color:#64748b;margin-bottom:1.2rem'>"
        "Metrics evaluated on the Cleveland Heart Disease test split (20%). "
        "Higher is better for all metrics.</p>",
        unsafe_allow_html=True
    )

    models_meta = [
        {
            "name": "Logistic Regression",
            "accuracy": 79.5, "f1": 81.1, "precision": 75.6, "recall": 87.4,
            "color": "#60a5fa",
        },
        {
            "name": "Random Forest",
            "accuracy": 98.5, "f1": 98.5, "precision": 100.0, "recall": 97.1,
            "color": "#e63946",
        },
        {
            "name": "SVM",
            "accuracy": 88.8, "f1": 89.4, "precision": 85.1, "recall": 94.2,
            "color": "#a78bfa",
        },
    ]

    #  Metric cards
    for m in models_meta:
        badge = " 🏆" if m["name"] == "Random Forest" else ""
        st.markdown(
            f"<p style='font-size:12px;font-weight:500;color:#94a3b8;"
            f"text-transform:uppercase;letter-spacing:0.08em;"
            f"margin:1rem 0 0.5rem'>{m['name']}{badge}</p>",
            unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in zip(
            [c1, c2, c3, c4],
            ["Accuracy", "F1 Score", "Precision", "Recall"],
            [m["accuracy"], m["f1"], m["precision"], m["recall"]]
        ):
            with col:
                color = m["color"]
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value' style='color:{color}'>{val:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # bar chart comparison between models
    st.markdown('<div class="section-label">Side-by-side comparison</div>', unsafe_allow_html=True)

    metrics     = ["Accuracy", "F1 Score", "Precision", "Recall"]
    bar_fig     = go.Figure()

    for m in models_meta:
        bar_fig.add_trace(go.Bar(
            name=m["name"],
            x=metrics,
            y=[m["accuracy"], m["f1"], m["precision"], m["recall"]],
            marker_color=m["color"],
            marker_line_width=0,
            hovertemplate="%{x}: %{y:.1f}%<extra>" + m["name"] + "</extra>",
        ))

    bar_fig.update_layout(
        **PLOTLY_TRANSPARENT,
        barmode="group",
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            font=dict(color="#94a3b8", size=12, family="DM Sans"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis=dict(
            tickfont=dict(color="#94a3b8", size=12, family="DM Sans"),
            showgrid=False,
        ),
        yaxis=dict(
            range=[70, 102],
            tickfont=dict(color="#64748b", size=10, family="DM Mono"),
            gridcolor="#1e2d45", showgrid=True, zeroline=False,
            ticksuffix="%",
        ),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown(
        "<p style='font-size:11px;color:#334155;font-family:DM Mono,monospace'>"
        "Evaluated on Cleveland Heart Disease test split (20% holdout).</p>",
        unsafe_allow_html=True
    )