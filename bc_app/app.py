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
st.set_page_config(page_title="BreastDx SVM", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#be185d 0%,#7c3aed 100%); border-radius: 22px;
       padding: 3rem 2.6rem; color: #fff; margin-bottom: 1.5rem;}
.hero-title {font-size: 2.7rem; font-weight: 800; line-height: 1.15; margin: 0 0 .8rem 0;}
.hero-sub {font-size: 1.15rem; max-width: 720px; opacity: .95; margin: 0;}
.badge {display:inline-block; background: rgba(255,255,255,.2); border-radius: 99px;
        padding: .25rem .9rem; font-size: .85rem; margin-bottom: 1rem;}
.card {border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.08);
       border-radius: 16px; padding: 1.2rem 1.3rem; min-height: 150px;}
.card h4 {margin: 0 0 .4rem 0; font-size: 1.05rem;}
.card p {margin: 0; opacity: .85; font-size: .95rem; line-height: 1.45;}
.card .icon {font-size: 1.8rem; margin-bottom: .4rem;}
.stat {border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.08);
       border-radius: 16px; padding: 1.1rem; text-align: center;}
.stat .v {font-size: 2rem; font-weight: 800; color: #db2777;}
.stat .l {font-size: .85rem; opacity: .8;}
.step {border-left: 4px solid #db2777; background: rgba(128,128,128,.08);
       border-radius: 10px; padding: .8rem 1rem; margin-bottom: .6rem;}
.step b {color: #db2777;}
.result {border-radius: 18px; padding: 1.6rem 1.8rem; color: #fff; margin: 1rem 0;}
.result.pos {background: linear-gradient(135deg,#b91c1c,#ef4444);}
.result.neg {background: linear-gradient(135deg,#047857,#10b981);}
.result .small {opacity: .9; font-size: .9rem;}
.result .big {font-size: 2.1rem; font-weight: 800; margin: .2rem 0;}
.gauge {position: relative; height: 14px; border-radius: 99px; margin: 1.6rem 0 .4rem 0;
        background: linear-gradient(90deg,#10b981,#e5e7eb 50%,#ef4444);}
.marker {position: absolute; top: -6px; width: 8px; height: 26px; background: #111;
         border: 2px solid #fff; border-radius: 5px; transform: translateX(-50%);}
.gauge-labels {display: flex; justify-content: space-between; font-size: .8rem; opacity: .8;}
.footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2.5rem;}
</style>
""", unsafe_allow_html=True)

NAV = ["🏠 Home", "🔬 Predict", "📊 Results", "🧠 Methodology", "📚 Datasets"]
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


def card(icon, title, text):
    return f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'


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
    st.markdown("## 🩺 BreastDx SVM")
    st.caption("Feature selection + Support Vector Machines for breast cancer diagnosis")
    st.radio("Navigate", NAV, key="page", label_visibility="collapsed")
    st.divider()
    st.warning("Educational project only. Not a medical device and not for clinical use.")

page = st.session_state.page

# ---------- pages ----------
if page == NAV[0]:
    st.markdown("""
    <div class="hero">
      <div class="badge">AI course project · Machine learning for healthcare</div>
      <div class="hero-title">Breast cancer diagnosis with feature selection and SVMs</div>
      <p class="hero-sub">Enter measurements, get an instant prediction, and explore how six
      different SVM models compare on two real clinical datasets.</p>
    </div>""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Try a prediction", type="primary", on_click=goto, args=(NAV[1],))
    b2.button("See results", on_click=goto, args=(NAV[2],))

    st.write("")
    s = st.columns(4)
    s[0].markdown(stat(f"{WIS['metrics']['Accuracy']:.1%}", "Wisconsin test accuracy"), unsafe_allow_html=True)
    s[1].markdown(stat(f"{WIS['metrics']['Recall']:.1%}", "Wisconsin malignant recall"), unsafe_allow_html=True)
    s[2].markdown(stat(f"{COI['metrics']['Accuracy']:.1%}", "Coimbra test accuracy"), unsafe_allow_html=True)
    s[3].markdown(stat("12", "models trained and compared"), unsafe_allow_html=True)

    st.subheader("What you can do")
    f = st.columns(3)
    f[0].markdown(card("🔬", "Instant predictions",
                       "Enter values by hand or upload a CSV for batch predictions, with a clear score gauge."), unsafe_allow_html=True)
    f[1].markdown(card("📊", "Transparent results",
                       "Confusion matrix, ROC and precision-recall curves, and the full model comparison."), unsafe_allow_html=True)
    f[2].markdown(card("🧠", "Feature selection",
                       "No selection, SelectKBest and RFE are each tested with linear and RBF kernels."), unsafe_allow_html=True)
    st.write("")
    f = st.columns(3)
    f[0].markdown(card("🧪", "Two datasets",
                       "Wisconsin Diagnostic (cell-nucleus images) and Coimbra (routine blood markers)."), unsafe_allow_html=True)
    f[1].markdown(card("🛡️", "No data leakage",
                       "Scaling and feature selection live inside the training pipeline and are tuned with cross-validation."), unsafe_allow_html=True)
    f[2].markdown(card("📥", "Downloadable reports",
                       "Save any single prediction as a report, or download batch predictions as CSV."), unsafe_allow_html=True)

elif page == NAV[1]:
    st.title("🔬 Predict")
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
    st.title("📊 Results")
    choice = st.radio("Dataset", list(DATASETS), horizontal=True)
    if DATASETS[choice] == "coi":
        st.info("Only 24 test cases, so these numbers are noisy. Missing patients (false negatives) is the main weakness.")
    results_view(BUNDLES[DATASETS[choice]])

elif page == NAV[3]:
    st.title("🧠 Methodology")
    st.write("Both datasets go through the same experiment, so the results are directly comparable.")
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
    c1.markdown(card("🎯", "Why recall matters",
                     "A false negative is a cancer case predicted as benign, which could delay further assessment. "
                     "That is why Wisconsin model selection optimises malignant recall."), unsafe_allow_html=True)
    c2.markdown(card("⚠️", "Limitations",
                     "Small datasets, a single train/test split, and no external validation. "
                     "The Coimbra model in particular should not be relied on."), unsafe_allow_html=True)

elif page == NAV[4]:
    st.title("📚 Datasets")
    c1, c2 = st.columns(2)
    c1.markdown(card("🔬", "Breast Cancer Wisconsin (Diagnostic)",
                     "569 observations, 30 numeric features computed from digitised images of fine-needle aspirates. "
                     "Target: benign or malignant. UCI repository, dataset ID 17."), unsafe_allow_html=True)
    c2.markdown(card("🩸", "Breast Cancer Coimbra",
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

st.markdown('<div class="footer">BreastDx SVM · educational project, not for clinical use</div>', unsafe_allow_html=True)
