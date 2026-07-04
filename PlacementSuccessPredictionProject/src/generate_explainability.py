from pathlib import Path
import sys

import joblib
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_ingestion import ingest_data
from explainability import (
    build_transformed_frame,
    ensure_reports_dir,
    generate_lime_explanation,
    generate_shap_summary,
    save_feature_importance,
)
from evaluate_models import save_roc_curve
from preprocessing import split_features_target


def main():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "placement_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    reports_dir = ensure_reports_dir()
    df = ingest_data()
    _, _, target_info = split_features_target(df)

    transformed = build_transformed_frame(df, pipeline, sample_size=2000)
    X, y, _ = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    y_score = pipeline.predict_proba(X_test)[:, 1]

    feature_importance_path = save_feature_importance(pipeline, reports_dir / "feature_importance.csv")
    roc_path = save_roc_curve(y_test, y_score, reports_dir / "roc_curve.png")
    shap_path = generate_shap_summary(pipeline, transformed, reports_dir / "shap_summary.png")
    lime_path = generate_lime_explanation(pipeline, transformed, reports_dir / "lime_explanation.html")

    print(f"Saved feature importance to {feature_importance_path}")
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved SHAP summary to {shap_path}")
    print(f"Saved LIME explanation to {lime_path}")
    print(f"Target source: {target_info.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
