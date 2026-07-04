from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import FEATURE_COLUMNS


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _resolve_classifier(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("classifier", model)
    return model


def _resolve_feature_names(model, feature_names: list[str] | None = None) -> list[str]:
    if feature_names is not None:
        return feature_names
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            return list(preprocessor.get_feature_names_out())
    return []


def get_feature_importance(model, top_n: int = 10, feature_names: list[str] | None = None) -> pd.DataFrame:
    classifier = _resolve_classifier(model)
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    resolved_names = _resolve_feature_names(model, feature_names)
    importances = classifier.feature_importances_
    if len(resolved_names) != len(importances):
        resolved_names = [f"feature_{index}" for index in range(len(importances))]

    frame = pd.DataFrame({"feature": resolved_names, "importance": importances})
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def explain_prediction(model, row: pd.DataFrame, top_n: int = 5) -> dict[str, object]:
    prediction = model.predict(row)[0]
    confidence = float(model.predict_proba(row)[0].max()) if hasattr(model, "predict_proba") else None
    importance = get_feature_importance(model, top_n=top_n)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_features": importance.to_dict(orient="records"),
    }


def save_feature_importance(model, output_path: Path | str, feature_names: list[str] | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    get_feature_importance(model, feature_names=feature_names).to_csv(output_path, index=False)
    return output_path


def build_transformed_frame(df: pd.DataFrame, pipeline, sample_size: int | None = None) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    if sample_size is not None and len(X) > sample_size:
      X = X.sample(n=sample_size, random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"] if hasattr(pipeline, "named_steps") else None
    if preprocessor is None:
        raise ValueError("Expected a pipeline with a preprocessor step.")

    transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def generate_shap_summary(model, transformed_X: pd.DataFrame, output_path: Path | str, sample_size: int = 1000) -> Path:
    import shap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = _resolve_classifier(model)
    sample = transformed_X
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[-1], sample, show=False)
    else:
        shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def generate_lime_explanation(model, transformed_X: pd.DataFrame, output_path: Path | str, sample_index: int = 0) -> Path:
    from lime.lime_tabular import LimeTabularExplainer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = _resolve_classifier(model)
    sample = transformed_X.iloc[sample_index].to_numpy()
    training_data = transformed_X.sample(n=min(len(transformed_X), 2000), random_state=42).to_numpy()

    class_names = None
    if hasattr(model, "classes_"):
        class_names = list(model.classes_)
    elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
        class_names = list(model.named_steps["classifier"].classes_)

    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=list(transformed_X.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(
        sample,
        classifier.predict_proba,
        num_features=min(10, transformed_X.shape[1]),
    )
    output_path.write_text(exp.as_html(), encoding="utf-8")
    return output_path


def ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR
