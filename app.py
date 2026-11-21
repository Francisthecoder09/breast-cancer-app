import re
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

MODEL_DIR = Path(__file__).parent / "models"
BRAND_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="Support Vector Machine mark">
  <rect x="1" y="1" width="62" height="62" rx="17" fill="#0f766e"/>
  <path d="M37 11 26 53" fill="none" stroke="#d7f5ef" stroke-width="3.2" stroke-linecap="round"/>
  <path d="M10 20 26 29M13 45 26 35M39 20 53 14M38 41 53 48" fill="none" stroke="#d7f5ef" stroke-width="2.2" stroke-linecap="round" opacity=".9"/>
  <circle cx="11" cy="19" r="4.2" fill="#fff"/><circle cx="13" cy="46" r="4.2" fill="#fff"/>
  <circle cx="53" cy="14" r="4.2" fill="#fff"/><circle cx="53" cy="49" r="4.2" fill="#fff"/>
  <circle cx="27" cy="32" r="5.1" fill="#f3c969" stroke="#fff" stroke-width="1.5"/>
</svg>
"""
st.set_page_config(page_title="Breast Cancer SVM Explorer", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: var(--background-color); color: var(--text-color);}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.22);}
[data-testid="stSidebar"] h2 {letter-spacing: -.03em;}
.sidebar-brand {display: flex; align-items: center; gap: .8rem; margin: .45rem 0 .25rem;}
.brand-mark {display: grid; place-items: center; width: 46px; height: 46px; flex: 0 0 46px;}
.brand-mark svg {display: block; width: 46px; height: 46px;}
.brand-name {font-size: 1.04rem; font-weight: 750; line-height: 1.2;}
.brand-product {font-size: .84rem; line-height: 1.35; opacity: .72; margin-top: .15rem;}
.sidebar-caption {font-size: .88rem; line-height: 1.45; opacity: .76; margin: .65rem 0 1.35rem;}
.sidebar-section-label {font-size: .72rem; font-weight: 700; letter-spacing: .12em;
                        text-transform: uppercase; opacity: .62; margin: 0 0 .55rem .15rem;}
.sidebar-note {background: var(--secondary-background-color); color: var(--text-color);
               border: 1px solid rgba(128,128,128,.24); border-left: 3px solid #0f766e;
               border-radius: 10px; padding: .8rem .85rem; margin-top: 1rem;}
.sidebar-note-title {font-size: .88rem; font-weight: 700; margin-bottom: .3rem;}
.sidebar-note p {font-size: .82rem; line-height: 1.45; opacity: .78; margin: 0;}
[data-testid="stSidebar"] .stButton > button {justify-content: flex-start; text-align: left;
    border-radius: 9px; min-height: 2.55rem; padding: .55rem .75rem; font-size: .94rem;
    transition: background-color .16s ease, border-color .16s ease, box-shadow .16s ease;}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: transparent; border-color: transparent; color: var(--text-color);}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(15,118,110,.09); border-color: rgba(15,118,110,.18);}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #0f766e; border-color: #0f766e; color: #fff;}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #115e59; border-color: #115e59;}
h1, h2, h3 {letter-spacing: -.025em;}
.hero {display: flex; align-items: center; justify-content: space-between; gap: 2rem;
       background: linear-gradient(120deg,#12313b 0%,#145b66 100%); border-radius: 18px;
       padding: 2.15rem 2.25rem; min-height: 250px; color: #fff; margin-bottom: 1.25rem;
       box-shadow: 0 14px 35px rgba(15,45,55,.14);}
.hero-copy {flex: 1; min-width: 0;}
.hero-art {display: grid; place-items: center; width: 142px; height: 142px; flex: 0 0 142px;
           border: 1px solid rgba(255,255,255,.2); border-radius: 28px;
           background: rgba(255,255,255,.08);}
.hero-art svg {display: block; width: 94px; height: 94px;}
.eyebrow {font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
          opacity: .86; margin-bottom: .9rem;}
.hero-title {font-size: clamp(1.9rem,3.3vw,2.65rem); font-weight: 750; line-height: 1.16;
             max-width: 760px; margin: 0 0 .75rem 0;}
.hero-sub {font-size: 1rem; line-height: 1.6; max-width: 700px; opacity: .92; margin: 0;}
.card {height: 100%; border: 1px solid rgba(128,128,128,.25);
       background: var(--secondary-background-color); color: var(--text-color);
       border-radius: 14px; padding: 1.2rem 1.25rem; min-height: 142px;
       box-shadow: 0 4px 14px rgba(15,23,42,.035);}
.card-mark {height: 3px; width: 36px; border-radius: 5px; background: #0f766e; margin-bottom: .85rem;}
.card h3 {margin: 0 0 .45rem 0; font-size: 1.04rem; font-weight: 700;}
.card p {margin: 0; color: var(--text-color); opacity: .82; font-size: .95rem; line-height: 1.55;}
.stat {height: 100%; border: 1px solid rgba(128,128,128,.25);
       background: var(--secondary-background-color); color: var(--text-color);
       border-radius: 14px; padding: 1rem .8rem; text-align: center;
       box-shadow: 0 4px 14px rgba(15,23,42,.035);}
.stat .v {font-size: clamp(1.45rem,2.8vw,2rem); font-weight: 750; color: #0f766e;
          font-variant-numeric: tabular-nums;}
.stat .l {font-size: .88rem; line-height: 1.4; opacity: .82;}
.step {border-left: 3px solid #0f766e; background: var(--secondary-background-color);
       border-radius: 0 10px 10px 0; padding: .9rem 1rem; margin-bottom: .65rem;
       line-height: 1.55;}
.step b {color: #0f766e;}
.result {border-radius: 16px; padding: 1.45rem 1.6rem; color: #fff; margin: 1rem 0;
         box-shadow: 0 10px 24px rgba(15,23,42,.12);}
.result.pos {background: linear-gradient(120deg,#7f1d1d,#b91c1c);}
.result.neg {background: linear-gradient(120deg,#115e59,#0f766e);}
.result .small {opacity: .92; font-size: .92rem; line-height: 1.5;}
.result .big {font-size: 2rem; font-weight: 750; margin: .22rem 0;}
.gauge {position: relative; height: 12px; border-radius: 99px; margin: 1.45rem 0 .45rem;
        background: linear-gradient(90deg,#67a99d,#e5e7eb 50%,#cf7777);}
.marker {position: absolute; top: -6px; width: 8px; height: 24px; background: #0f172a;
         border: 2px solid #fff; border-radius: 5px; transform: translateX(-50%);}
.gauge-labels {display: flex; justify-content: space-between; gap: .5rem;
               font-size: .82rem; opacity: .9;}
.footer {text-align: center; opacity: .68; font-size: .88rem; margin-top: 2.5rem;
         padding-top: 1rem; border-top: 1px solid rgba(128,128,128,.2);}
.stButton > button {border-radius: 10px; min-height: 2.7rem; font-weight: 600;
                    transition: transform .16s ease, box-shadow .16s ease;}
.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 5px 14px rgba(15,23,42,.12);}
button:focus-visible, input:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid #0f766e !important; outline-offset: 2px !important;
}
@media (prefers-color-scheme: dark) {
  .stat .v, .step b {color: #5eead4;}
  .card-mark {background: #5eead4;}
}
@media (max-width: 700px) {
  .block-container {padding-top: 1.25rem; padding-left: 1rem; padding-right: 1rem;}
  .hero {flex-direction: column; align-items: flex-start; gap: 1rem; padding: 1.5rem 1.25rem; border-radius: 16px;}
  .hero-art {order: -1; width: 68px; height: 68px; flex-basis: 68px; border-radius: 18px;}
  .hero-art svg {width: 48px; height: 48px;}
  .hero-sub {font-size: 1rem;}
  .card {min-height: 0;}
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {scroll-behavior: auto !important; transition-duration: .01ms !important;}
}
</style>
""", unsafe_allow_html=True)

NAV = ["Home", "Prediction", "Model evaluation", "Methodology", "Datasets"]
DATASETS = {"Wisconsin Diagnostic": "wis", "Coimbra": "coi"}


# ---------- model loading ----------
@st.cache_resource
def load(name):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


if not (MODEL_DIR / "wisconsin.joblib").exists() or not (MODEL_DIR / "coimbra.joblib").exists():
    with st.spinner("First run: downloading data and training models (about a minute)..."):
        import train_models
        train_models.main()
    st.cache_resource.clear()

WIS, COI = load("wisconsin"), load("coimbra")
BUNDLES = {"wis": WIS, "coi": COI}


# ---------- helpers ----------
def goto(page):
    st.session_state.page = page


def card(title, text):
    return f'<div class="card"><div class="card-mark" aria-hidden="true"></div><h3>{title}</h3><p>{text}</p></div>'


def stat(value, label):
    return f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>'


def predict_df(b, df):
    scores = b["model"].decision_function(df[b["features"]])
    neg, pos = b["labels"]
    out = df.copy()
    out["Prediction"] = np.where(scores > 0, pos, neg)
    out["SVM score"] = scores.round(3)
    return out


def result_card(b, score):
    neg, pos = b["labels"]
    is_pos = score > 0
    label = pos if is_pos else neg
    pct = (np.clip(score, -3, 3) + 3) / 6 * 100
    st.markdown(f"""
    <div class="result {'pos' if is_pos else 'neg'}">
      <div class="small">Model prediction</div>
      <div class="big">{label}</div>
      <div class="small">SVM decision score: {score:.3f} &nbsp;|&nbsp; {b['best_config']}</div>
      <div class="gauge"><div class="marker" style="left:{pct:.1f}%"></div></div>
      <div class="gauge-labels"><span>{neg}</span><span>decision boundary</span><span>{pos}</span></div>
    </div>""", unsafe_allow_html=True)
    st.caption("The further the marker is from the centre, the further the case is from the decision boundary. "
               "This is a raw SVM score, not a calibrated probability.")


def report_html(title, b, values, score):
    neg, pos = b["labels"]
    label = pos if score > 0 else neg
    rows = "".join(f"<tr><td>{k}</td><td>{v:.5g}</td></tr>" for k, v in values.items())
    return f"""<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:Arial;max-width:720px;margin:2rem auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;text-align:left}}
.box{{padding:1rem;border-radius:10px;background:#f3f4f6}}</style></head><body>
<h1>{title}</h1><p>Generated {date.today().isoformat()}</p>
<div class="box"><h2>Prediction: {label}</h2>
<p>SVM decision score: {score:.3f}<br>Model: {b['best_config']}</p></div>
<h3>Input values</h3><table><tr><th>Feature</th><th>Value</th></tr>{rows}</table>
<p><i>Educational project only. Not a medical device and not for clinical use.</i></p>
</body></html>"""


def feature_inputs(b, key):
    feats, stats = b["features"], b["stats"]
    values = {}

    def field(f, col, label):
        kw = {"key": f"{key}_{f}"}
        if kw["key"] not in st.session_state:
            kw["value"] = float(stats.loc[f, "median"])
        values[f] = col.number_input(label, format="%.5f", **kw)

    if all(re.fullmatch(r"[a-z_]+[123]", f) for f in feats):
        names = {"1": "Mean values", "2": "Standard error", "3": "Worst values"}
        for tab, suffix in zip(st.tabs(list(names.values())), names):
            cols = tab.columns(3)
            group = [f for f in feats if f.endswith(suffix)]
            for i, f in enumerate(group):
                field(f, cols[i % 3], f[:-1].replace("_", " ").title())
    else:
        cols = st.columns(3)
        for i, f in enumerate(feats):
            field(f, cols[i % 3], f)
    return values


def results_view(b):
    m = b["metrics"]
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.markdown(stat(f"{v:.3f}", k), unsafe_allow_html=True)
    st.caption(f"Held-out test set of {len(b['y_test'])} cases. "
               f"Selected model: **{b['best_config']}** (chosen by cross-validated {b['scoring'].replace('_', ' ')}).")

    neg, pos = b["labels"]
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(b["y_test"], b["y_pred"], display_labels=[neg, pos],
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(b["y_test"], b["scores"], ax=ax)
        ax.set_title("ROC curve")
        fig.tight_layout()
        st.pyplot(fig)
    with c3:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        PrecisionRecallDisplay.from_predictions(b["y_test"], b["scores"], ax=ax)
        ax.set_title("Precision-recall curve")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("All six models compared")
    table = b["comparison"].copy()
    chosen = table["Feature method"] + " + " + table["SVM kernel"] + " SVM" == b["best_config"]
    table.insert(0, "Selected", np.where(chosen, "✓", ""))
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if table[c].dtype == float}),
                 width="stretch", hide_index=True)


# ---------- sidebar ----------
if "page" not in st.session_state:
    st.session_state.page = NAV[0]

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
      <div class="brand-mark" aria-hidden="true">{BRAND_SVG}</div>
      <div><div class="brand-name">Breast Cancer</div><div class="brand-product">SVM Explorer</div></div>
    </div>
    <p class="sidebar-caption">A student project exploring machine-learning methods across two datasets.</p>
    <div class="sidebar-section-label">Workspace</div>
    """, unsafe_allow_html=True)
    for destination in NAV:
        st.button(
            destination,
            key=f"nav_{destination.lower()}",
            type="primary" if st.session_state.page == destination else "secondary",
            width="stretch",
            on_click=goto,
            args=(destination,),
        )
    st.markdown("""
    <div class="sidebar-note">
      <div class="sidebar-note-title">Educational use only</div>
      <p>Model outputs are experimental and are not medical advice or for clinical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ---------- pages ----------
if page == NAV[0]:
    st.markdown(f"""
    <div class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Student research project · Machine learning</div>
        <div class="hero-title">Breast cancer data, explored with SVMs</div>
        <p class="hero-sub">Explore two independent classification studies, compare feature-selection
        methods, and review the model outputs.</p>
      </div>
      <div class="hero-art" aria-hidden="true">{BRAND_SVG}</div>
    </div>""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Try a prediction", type="primary", on_click=goto, args=(NAV[1],))
    b2.button("Model evaluation", on_click=goto, args=(NAV[2],))

    st.write("")
    s = st.columns(4)
    s[0].markdown(stat(f"{WIS['metrics']['Accuracy']:.1%}", "Wisconsin test accuracy"), unsafe_allow_html=True)
    s[1].markdown(stat(f"{WIS['metrics']['Recall']:.1%}", "Wisconsin malignant recall"), unsafe_allow_html=True)
    s[2].markdown(stat(f"{COI['metrics']['Accuracy']:.1%}", "Coimbra test accuracy"), unsafe_allow_html=True)
    s[3].markdown(stat("12", "models trained and compared"), unsafe_allow_html=True)

    st.subheader("What you can do")
    f = st.columns(3)
    f[0].markdown(card("Model predictions",
                       "Enter measurements manually or upload a CSV to explore model outputs."), unsafe_allow_html=True)
    f[1].markdown(card("Evaluation results",
                       "Review confusion matrices, ROC and precision-recall curves, and model comparisons."), unsafe_allow_html=True)
    f[2].markdown(card("Feature selection",
                       "Compare no selection, SelectKBest, and RFE with linear and RBF kernels."), unsafe_allow_html=True)
    st.write("")
    f = st.columns(3)
    f[0].markdown(card("Two independent datasets",
                       "Wisconsin uses cell-nucleus measurements; Coimbra uses routine blood and body markers."), unsafe_allow_html=True)
    f[1].markdown(card("Leakage-aware workflow",
                       "Scaling and feature selection are fitted within the training pipeline during cross-validation."), unsafe_allow_html=True)
    f[2].markdown(card("Exportable outputs",
                       "Download an individual prediction summary or a CSV of batch predictions."), unsafe_allow_html=True)

elif page == NAV[1]:
    st.title("Prediction")
    choice = st.radio("Dataset", list(DATASETS), horizontal=True)
    key = DATASETS[choice]
    b = BUNDLES[key]
    neg, pos = b["labels"]

    if key == "wis":
        st.caption("Benign vs malignant, from 30 cell-nucleus measurements.")
    else:
        st.caption("Healthy control vs patient, from 9 routine blood and body measurements.")
        st.info("This dataset is small (116 rows) and the model misses a share of patients. Treat results with caution.")

    mode = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)

    if mode == "Manual entry":
        c1, c2, c3, _ = st.columns([1.3, 1.3, 1.1, 2])
        if c1.button(f"Load example: {neg}"):
            st.session_state.update({f"{key}_{f}": v for f, v in b["examples"][neg].items()})
            st.session_state.pop(f"last_{key}", None)
        if c2.button(f"Load example: {pos}"):
            st.session_state.update({f"{key}_{f}": v for f, v in b["examples"][pos].items()})
            st.session_state.pop(f"last_{key}", None)
        if c3.button("Reset"):
            st.session_state.update({f"{key}_{f}": float(b["stats"].loc[f, "median"]) for f in b["features"]})
            st.session_state.pop(f"last_{key}", None)

        values = feature_inputs(b, key)
        if st.button("Predict", type="primary"):
            score = float(predict_df(b, pd.DataFrame([values]))["SVM score"].iloc[0])
            st.session_state[f"last_{key}"] = (values, score)

        if f"last_{key}" in st.session_state:
            vals, score = st.session_state[f"last_{key}"]
            st.markdown("#### Prediction output")
            result_card(b, score)
            st.download_button("Download report", report_html(f"{choice} prediction report", b, vals, score),
                               f"{key}_prediction_report.html", "text/html")
    else:
        st.write("Your CSV needs these columns:")
        st.code(", ".join(b["features"]))
        up = st.file_uploader("CSV file", type="csv")
        if up:
            df = pd.read_csv(up)
            missing = [c for c in b["features"] if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                res = predict_df(b, df)
                flagged = int((res["Prediction"] == pos).sum())
                m = st.columns(3)
                m[0].markdown(stat(len(res), "rows"), unsafe_allow_html=True)
                m[1].markdown(stat(flagged, f"predicted {pos.lower()}"), unsafe_allow_html=True)
                m[2].markdown(stat(len(res) - flagged, f"predicted {neg.lower()}"), unsafe_allow_html=True)
                st.dataframe(res[["Prediction", "SVM score"] + b["features"]], hide_index=True)
                st.download_button("Download predictions (CSV)", res.to_csv(index=False),
                                   "predictions.csv", "text/csv")

elif page == NAV[2]:
    st.title("Model evaluation")
    st.caption("These charts summarize held-out test performance. New case predictions appear directly below the Predict button on the Prediction page.")
    choice = st.radio("Dataset", list(DATASETS), horizontal=True)
    if DATASETS[choice] == "coi":
        st.info("Only 24 test cases, so these numbers are noisy. Missing patients (false negatives) is the main weakness.")
    results_view(BUNDLES[DATASETS[choice]])

elif page == NAV[3]:
    st.title("Methodology")
    st.write("The experiments use a similar workflow, but the datasets predict different outcomes. Their metrics should be interpreted separately.")
    steps = [
        ("1. Split", "80% training, 20% held-out test, stratified so both sets keep the class balance (random state 42)."),
        ("2. Scale", "StandardScaler, fitted on training data only."),
        ("3. Select features", "Three options: none, SelectKBest (ANOVA F-test), or RFE with a linear SVM."),
        ("4. Train an SVM", "Linear and RBF kernels, tuned over C, class weights and gamma."),
        ("5. Cross-validate", "GridSearchCV with stratified 5-fold CV on the training set only. "
                              "Wisconsin is scored on malignant recall; Coimbra on balanced accuracy."),
        ("6. Test once", "The best of the six models is evaluated once on the untouched test set."),
    ]
    for t, d in steps:
        st.markdown(f'<div class="step"><b>{t}</b> &nbsp; {d}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.markdown(card("Why recall matters",
                     "A false negative is a cancer case predicted as benign, which could delay further assessment. "
                     "That is why Wisconsin model selection optimises malignant recall."), unsafe_allow_html=True)
    c2.markdown(card("Limitations",
                     "Small datasets, a single train/test split, and no external validation. "
                     "The Coimbra model in particular should not be relied on."), unsafe_allow_html=True)

elif page == NAV[4]:
    st.title("Datasets")
    c1, c2 = st.columns(2)
    c1.markdown(card("Breast Cancer Wisconsin (Diagnostic)",
                     "569 observations, 30 numeric features computed from digitised images of fine-needle aspirates. "
                     "Target: benign or malignant. UCI repository, dataset ID 17."), unsafe_allow_html=True)
    c2.markdown(card("Breast Cancer Coimbra",
                     "116 observations, 9 routine blood and body measurements. "
                     "Target: healthy control or breast cancer patient. UCI repository, dataset ID 451."), unsafe_allow_html=True)
    st.write("")
    with st.expander("Wisconsin features"):
        st.write("Ten measurements of each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, "
                 "concavity, concave points, symmetry and fractal dimension. Each one is reported as a mean, a "
                 "standard error and a worst (largest) value, giving 30 features.")
    with st.expander("Coimbra features"):
        st.write("Age, BMI, glucose, insulin, HOMA, leptin, adiponectin, resistin and MCP-1.")
    st.caption("The two experiments are run separately. Metrics are never combined across datasets.")

st.markdown('<div class="footer">Educational research demonstration · Not a medical device and not for clinical use</div>', unsafe_allow_html=True)
