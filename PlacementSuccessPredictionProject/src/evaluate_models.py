from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_ingestion import ingest_data
from preprocessing import split_features_target


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"


def evaluate_model(model, X_test, y_test, class_names: list[str] | None = None) -> dict[str, object]:
    predictions = model.predict(X_test)
    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return results


def print_evaluation(results: dict[str, object]) -> None:
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(pd.DataFrame(results["classification_report"]).transpose().to_string())
    print("Confusion matrix:")
    print(pd.DataFrame(results["confusion_matrix"]).to_string(index=False, header=False))


def save_confusion_matrix(y_true, y_pred, output_path: Path | str, class_names: list[str] | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_roc_curve(y_true, y_score, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_trained_model(path: Path | str = MODEL_PATH) -> dict[str, object]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")
    return artifact


def main(model=None, class_names=None) -> dict[str, object]:
    artifact = None
    if model is None:
        artifact = load_trained_model()
        model = artifact["pipeline"]
        class_names = artifact.get("class_names")
    elif class_names is None:
        class_names = ["Below median salary", "At or above median salary"]

    df = ingest_data()
    X, y, _ = split_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = evaluate_model(model, X_test, y_test, class_names=class_names)
    y_pred = model.predict(X_test)
    save_confusion_matrix(
        y_test,
        y_pred,
        REPORTS_DIR / "confusion_matrix.png",
        class_names=class_names,
    )
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        save_roc_curve(y_test, y_score, REPORTS_DIR / "roc_curve.png")
    print_evaluation(results)
    return results


if __name__ == "__main__":
    loaded_artifact = load_trained_model()
    main(loaded_artifact["pipeline"], loaded_artifact.get("class_names"))
