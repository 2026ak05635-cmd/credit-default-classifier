from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score
)

st.set_page_config(
    page_title="Credit Default Model Explorer",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

FEATURES = [f"X{i}" for i in range(1, 24)]
TARGET = "default"

@st.cache_resource
def load_model(filename):
    return joblib.load(BASE_DIR / "model" / filename)

st.title("Credit Default Model Explorer")
st.write(
    "Compare five classifiers trained on the UCI Default of Credit Card "
    "Clients dataset."
)
st.caption(
    "Target: 0 = no default; 1 = default. "
    "An educational comparison using a held-out test set."
)

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox(
        "Choose a classifier",
        list(MODEL_FILES)
    )

    sample_path = BASE_DIR / "test_data.csv"

    if sample_path.exists():
        st.download_button(
            "Download held-out test CSV",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv"
        )

    st.write(
        "Upload test data with columns X1 through X23 and default. "
        "Saved preprocessing is applied automatically."
    )

comparison_path = BASE_DIR / "model_comparison.csv"

if comparison_path.exists():
    with st.expander("Compare all models on the original held-out test set"):
        comparison = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comparison.round(4), use_container_width=True)
        st.caption(
            "These are saved training-run results. "
            "The evaluation below uses your uploaded CSV."
        )

uploaded_file = st.file_uploader(
    "Upload labeled test data",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Download the held-out test CSV from the sidebar, "
        "then upload it here to evaluate a model."
    )
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

try:
    inputs = data[FEATURES].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(data[TARGET], errors="raise")
except (ValueError, TypeError):
    st.error("All input features and the default column must be numeric.")
    st.stop()

if not np.isfinite(inputs.to_numpy(dtype=float)).all():
    st.error("Input features must not contain missing or infinite values.")
    st.stop()

if not target.isin([0, 1]).all():
    st.error("The default column must contain only 0 and 1.")
    st.stop()

target = target.astype(int)

st.subheader("Uploaded test data")
left, right = st.columns(2)
left.metric("Test records", f"{len(data):,}")
right.metric("Actual default rate", f"{target.mean():.2%}")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), use_container_width=True)

if not st.button("Evaluate selected model", type="primary"):
    st.stop()

try:
    model = load_model(MODEL_FILES[selected_model])

    with st.spinner(f"Evaluating {selected_model}..."):
        predictions = model.predict(inputs)

        if hasattr(model, "predict_proba"):
            positive_index = list(model.classes_).index(1)
            scores = model.predict_proba(inputs)[:, positive_index]
        else:
            scores = model.decision_function(inputs)
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()

auc = (
    roc_auc_score(target, scores)
    if target.nunique() == 2
    else None
)

metrics = {
    "Accuracy": accuracy_score(target, predictions),
    "AUC": auc,
    "Precision": precision_score(target, predictions, zero_division=0),
    "Recall": recall_score(target, predictions, zero_division=0),
    "F1": f1_score(target, predictions, zero_division=0),
    "MCC": matthews_corrcoef(target, predictions)
}

st.subheader(f"{selected_model}: evaluation results")

for row in [
    ["Accuracy", "AUC", "Precision"],
    ["Recall", "F1", "MCC"]
]:
    columns = st.columns(3)

    for column, name in zip(columns, row):
        value = metrics[name]
        column.metric(
            name,
            "N/A" if value is None else f"{value:.4f}"
        )

if auc is None:
    st.warning(
        "AUC requires both target classes. "
        "This CSV contains only one class."
    )

st.caption(
    "Precision, recall, and F1 use default (1) as the positive class. "
    "SVM AUC uses decision scores."
)

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")

    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No default", "Default"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")

    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["No default", "Default"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True
    )

prediction_data = data[required_columns].copy()
prediction_data["predicted_default"] = predictions

st.download_button(
    "Download predictions",
    data=prediction_data.to_csv(index=False).encode("utf-8"),
    file_name="credit_default_predictions.csv",
    mime="text/csv"
)
