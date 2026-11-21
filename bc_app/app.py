from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

MODEL_DIR = Path(__file__).parent / "models"
st.set_page_config(page_title="Breast Cancer SVM Diagnosis", page_icon="🩺", layout="wide")


@st.cache_resource
def load(name):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def predict_df(b, df):
    X = df[b["features"]]
    scores = b["model"].decision_function(X)
    neg, pos = b["labels"]
    out = df.copy()
    out["Prediction"] = np.where(scores > 0, pos, neg)
    out["SVM score"] = scores.round(3)
    return out


def predict_tab(b, key):
    neg, pos = b["labels"]
    stats = b["stats"]
    st.caption(f"Model: **{b['best_config']}** (chosen by cross-validation). "
               f"Features used by the model: {len(b['kept_features'])} of {len(b['features'])}.")

    mode = st.radio("Input", ["Manual entry", "Upload CSV"], horizontal=True, key=f"mode_{key}")

    if mode == "Manual entry":
        c1, c2, c3 = st.columns(3)
        if c1.button(f"Load example: {neg}", key=f"ex0_{key}"):
            st.session_state.update({f"{key}_{f}": v for f, v in b["examples"][neg].items()})
        if c2.button(f"Load example: {pos}", key=f"ex1_{key}"):
            st.session_state.update({f"{key}_{f}": v for f, v in b["examples"][pos].items()})
        if c3.button("Reset to medians", key=f"rs_{key}"):
            st.session_state.update({f"{key}_{f}": float(stats.loc[f, "median"]) for f in b["features"]})

        values, cols = {}, st.columns(3)
        for i, f in enumerate(b["features"]):
            default = float(stats.loc[f, "median"])
            kw = {"key": f"{key}_{f}"}
            if kw["key"] not in st.session_state:
                kw["value"] = default
            values[f] = cols[i % 3].number_input(f, format="%.5f", **kw)

        if st.button("Predict", type="primary", key=f"go_{key}"):
            res = predict_df(b, pd.DataFrame([values])).iloc[0]
            if res["Prediction"] == pos:
                st.error(f"Prediction: **{pos}** (SVM score {res['SVM score']})")
            else:
                st.success(f"Prediction: **{neg}** (SVM score {res['SVM score']})")
            st.caption("Positive score = " + pos + ", negative = " + neg +
                       ". Larger magnitude means further from the decision boundary.")
    else:
        st.write("CSV columns must be:", ", ".join(b["features"]))
        up = st.file_uploader("CSV file", type="csv", key=f"up_{key}")
        if up:
            df = pd.read_csv(up)
            missing = [f for f in b["features"] if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                res = predict_df(b, df)
                st.dataframe(res[["Prediction", "SVM score"] + b["features"]])
                st.download_button("Download results", res.to_csv(index=False),
                                   "predictions.csv", "text/csv")


def results_tab(b):
    m = b["metrics"]
    st.subheader("Held-out test performance")
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, f"{v:.3f}")
    st.caption(f"Test set size: {len(b['y_test'])}. Model selected by CV {b['scoring']}.")

    neg, pos = b["labels"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(b["y_test"], b["y_pred"],
                                            display_labels=[neg, pos], ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(b["y_test"], b["scores"], ax=axes[1])
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(b["y_test"], b["scores"], ax=axes[2])
    axes[2].set_title("Precision-recall curve")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Model comparison")
    st.dataframe(b["comparison"].style.format(
        {c: "{:.3f}" for c in b["comparison"].columns if b["comparison"][c].dtype == float}),
        width="stretch")


st.title("🩺 Breast Cancer Diagnosis: Feature Selection + SVM")
st.warning("Educational project only. Not a medical device and not for clinical use.")

if not (MODEL_DIR / "wisconsin.joblib").exists() or not (MODEL_DIR / "coimbra.joblib").exists():
    with st.spinner("First run: downloading data and training models (about a minute)..."):
        import train_models
        train_models.main()
    st.cache_resource.clear()

wis, coi = load("wisconsin"), load("coimbra")

t1, t2, t3, t4 = st.tabs(["Wisconsin: predict", "Coimbra: predict", "Wisconsin: results", "Coimbra: results"])
with t1:
    st.caption("UCI Breast Cancer Wisconsin (Diagnostic): benign vs malignant from cell-nucleus measurements.")
    predict_tab(wis, "wis")
with t2:
    st.caption("UCI Breast Cancer Coimbra: healthy control vs patient from routine blood and anthropometric data.")
    st.info("Small dataset (116 rows) and a weaker model, with missed patients on the test set. Treat results with caution.")
    predict_tab(coi, "coi")
with t3:
    results_tab(wis)
with t4:
    results_tab(coi)
