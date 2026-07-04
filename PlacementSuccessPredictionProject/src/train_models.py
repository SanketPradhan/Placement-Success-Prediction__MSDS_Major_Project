from pathlib import Path
import sys

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_ingestion import ingest_data
from evaluate_models import evaluate_model, print_evaluation, save_confusion_matrix, save_roc_curve
from explainability import get_feature_importance, save_feature_importance, ensure_reports_dir
from preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, split_features_target


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "placement_model.pkl"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    classifier = XGBClassifier(
        learning_rate=0.08,
        max_depth=5,
        n_estimators=250,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        objective="binary:logistic",
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main():
    df = ingest_data()
    X, y, target_info = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    class_names = [target_info["negative_label"], target_info["positive_label"]]
    evaluation = evaluate_model(pipeline, X_test, y_test, class_names=class_names)
    y_pred = pipeline.predict(X_test)
    save_confusion_matrix(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png", class_names=class_names)
    save_roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1], REPORTS_DIR / "roc_curve.png")
    print_evaluation(evaluation)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ensure_reports_dir()
    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_names": class_names,
        "target_info": target_info,
    }
    joblib.dump(artifact, MODEL_PATH)

    feature_importance = get_feature_importance(pipeline, top_n=15)
    feature_importance_path = save_feature_importance(
        pipeline,
        REPORTS_DIR / "feature_importance.csv",
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature importance to {feature_importance_path}")
    print(feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()
