"""Train both SVM pipelines (same method as the notebooks) and save them for the app.
Run once:  python train_models.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

MODEL_DIR = Path(__file__).parent / "models"


def run_experiment(X, y, scoring, k_values, labels):
    """labels = (negative_name, positive_name). y must be encoded 0/1."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    selectors = [
        ("No selection", "passthrough"),
        ("SelectKBest", SelectKBest(score_func=f_classif)),
        ("RFE", RFE(LinearSVC(dual=False, max_iter=10000, random_state=42))),
    ]
    rows, searches = [], []
    for method, selector in selectors:
        for kernel in ["linear", "rbf"]:
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("selector", selector),
                             ("svm", SVC(kernel=kernel))])
            grid = {"svm__C": [0.1, 1, 10],
                    "svm__class_weight": [None, "balanced"]}
            if kernel == "rbf":
                grid["svm__gamma"] = ["scale", "auto"]
            if method == "SelectKBest":
                grid["selector__k"] = k_values
            elif method == "RFE":
                grid["selector__n_features_to_select"] = k_values

            search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            pred = best.predict(X_test)
            kept = X_train.shape[1] if method == "No selection" else int(
                best.named_steps["selector"].get_support().sum())
            rows.append({
                "Feature method": method, "SVM kernel": kernel,
                "Features kept": kept, f"CV {scoring}": search.best_score_,
                "Test recall": recall_score(y_test, pred, pos_label=1),
                "Test precision": precision_score(y_test, pred, pos_label=1, zero_division=0),
                "Test false negatives": int(((y_test.to_numpy() == 1) & (pred == 0)).sum()),
            })
            searches.append(search)

    table = pd.DataFrame(rows)
    best_idx = int(table[f"CV {scoring}"].to_numpy().argmax())  # first max wins ties
    model = searches[best_idx].best_estimator_
    pred = model.predict(X_test)
    scores = model.decision_function(X_test)

    sel = model.named_steps["selector"]
    kept_features = (list(X_train.columns[sel.get_support()])
                     if sel != "passthrough" else list(X_train.columns))

    return {
        "model": model,
        "features": list(X.columns),
        "kept_features": kept_features,
        "labels": labels,
        "scoring": scoring,
        "best_config": f"{table.loc[best_idx, 'Feature method']} + {table.loc[best_idx, 'SVM kernel']} SVM",
        "comparison": table,
        "stats": X_train.describe().loc[["min", "max", "50%"]].T.rename(columns={"50%": "median"}),
        "y_test": y_test.to_numpy(), "y_pred": pred, "scores": scores,
        "metrics": {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, scores),
        },
        "examples": {
            labels[0]: X_test[y_test == 0].iloc[0].to_dict(),
            labels[1]: X_test[y_test == 1].iloc[0].to_dict(),
        },
    }


def main():
    from ucimlrepo import fetch_ucirepo
    MODEL_DIR.mkdir(exist_ok=True)

    d = fetch_ucirepo(id=17)  # Wisconsin Diagnostic
    X = d.data.features.copy()
    y = d.data.targets.squeeze("columns").map({"B": 0, "M": 1}).astype("int64")
    joblib.dump(run_experiment(X, y, "recall", [5, 10, 20], ("Benign", "Malignant")),
                MODEL_DIR / "wisconsin.joblib")
    print("Saved Wisconsin model")

    d = fetch_ucirepo(id=451)  # Coimbra
    X = d.data.features.copy()
    y = d.data.targets.squeeze().map({1: 0, 2: 1}).astype("int64")
    joblib.dump(run_experiment(X, y, "balanced_accuracy", [3, 5, 7], ("Healthy control", "Patient")),
                MODEL_DIR / "coimbra.joblib")
    print("Saved Coimbra model")


if __name__ == "__main__":
    main()
