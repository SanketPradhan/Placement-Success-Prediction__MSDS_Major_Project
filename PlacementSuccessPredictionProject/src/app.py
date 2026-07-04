from pathlib import Path
import sys

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import CATEGORICAL_FEATURES, DATA_PATH, FEATURE_COLUMNS, NUMERIC_FEATURES


app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent.parent / "templates"))

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"

FIELD_SPECS = [
    {"name": "Age", "label": "Age", "type": "number", "step": "1", "min": 16, "max": 40},
    {"name": "Gender", "label": "Gender", "type": "select", "options": ["Female", "Male", "Other"]},
    {"name": "College_Tier", "label": "College Tier", "type": "select", "options": ["Tier 1", "Tier 2", "Tier 3"]},
    {
        "name": "Specialization",
        "label": "Specialization",
        "type": "select",
        "options": [
            "AI/ML",
            "Computer Science",
            "Data Science",
            "Electronics",
            "Information Technology",
            "Mechanical",
            "Civil",
        ],
    },
    {"name": "CGPA", "label": "CGPA", "type": "number", "step": "0.01", "min": 0, "max": 10},
    {"name": "DSA_Problems_Solved", "label": "DSA Problems Solved", "type": "number", "step": "1", "min": 0, "max": 500},
    {"name": "Internships", "label": "Internships", "type": "number", "step": "1", "min": 0, "max": 10},
    {"name": "Certifications", "label": "Certifications", "type": "number", "step": "1", "min": 0, "max": 20},
    {"name": "Projects_Count", "label": "Projects Count", "type": "number", "step": "1", "min": 0, "max": 15},
    {"name": "Communication_Skills", "label": "Communication Skills", "type": "number", "step": "1", "min": 0, "max": 100},
    {"name": "Aptitude_Test_Score", "label": "Aptitude Test Score", "type": "number", "step": "1", "min": 0, "max": 100},
    {"name": "LeetCode_Rating", "label": "LeetCode Rating", "type": "number", "step": "1", "min": 0, "max": 5000},
    {"name": "GitHub_Contributions", "label": "GitHub Contributions", "type": "number", "step": "1", "min": 0, "max": 1000},
    {"name": "Hackathons_Participated", "label": "Hackathons Participated", "type": "number", "step": "1", "min": 0, "max": 20},
    {"name": "AI_ML_Skill_Level", "label": "AI/ML Skill Level", "type": "number", "step": "1", "min": 0, "max": 10},
    {"name": "System_Design_Knowledge", "label": "System Design Knowledge", "type": "number", "step": "1", "min": 0, "max": 10},
    {"name": "Resume_Score", "label": "Resume Score", "type": "number", "step": "1", "min": 0, "max": 100},
    {"name": "Mock_Interview_Score", "label": "Mock Interview Score", "type": "number", "step": "1", "min": 0, "max": 100},
]

artifact = None
MODEL_LOAD_ERROR = None


def load_model():
    if not MODEL_PATH.exists():
        return None, f"Missing model artifact: {MODEL_PATH}"
    try:
        loaded = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, f"Failed to load model artifact: {exc}"
    if not isinstance(loaded, dict):
        return None, "Model artifact is invalid: expected a dictionary."
    if "pipeline" not in loaded:
        return None, "Model artifact is invalid: missing 'pipeline' entry."
    return loaded, None


def normalize_payload(payload):
    normalized = {}
    for spec in FIELD_SPECS:
        name = spec["name"]
        value = payload.get(name, "")
        if spec["type"] == "number":
            if value in ("", None):
                normalized[name] = None
            else:
                try:
                    normalized[name] = float(value)
                except (TypeError, ValueError):
                    normalized[name] = value
        else:
            normalized[name] = str(value).strip()
    return normalized


def build_prediction(normalized):
    if artifact is None or not isinstance(artifact, dict) or "pipeline" not in artifact:
        return None, {"error": MODEL_LOAD_ERROR or "Model is not available. Please try again later."}, 503

    missing = [spec["name"] for spec in FIELD_SPECS if normalized.get(spec["name"]) in (None, "")]
    if missing:
        return None, {"error": "Missing required fields", "missing": missing}, 400

    invalid_numeric = [name for name in NUMERIC_FEATURES if not isinstance(normalized.get(name), (int, float))]
    if invalid_numeric:
        return None, {"error": "Numeric fields must contain numbers", "invalid": invalid_numeric}, 400

    row = pd.DataFrame([{field: normalized[field] for field in FEATURE_COLUMNS}])
    pipeline = artifact["pipeline"]
    probability = float(pipeline.predict_proba(row)[0][1])
    prediction = int(pipeline.predict(row)[0])
    class_names = artifact.get("class_names", ["Below median salary", "At or above median salary"])
    target_info = artifact.get("target_info", {})

    result = {
        "Predicted Outcome": class_names[prediction] if prediction < len(class_names) else str(prediction),
        "Success Probability": round(probability, 2),
        "Failure Probability": round(1.0 - probability, 2),
        "Decision Rule": target_info.get("description", "Placement model output"),
        "Target Source": target_info.get("source", "unknown"),
    }
    threshold = target_info.get("threshold")
    if threshold is not None:
        result["Salary Threshold (LPA)"] = round(float(threshold), 2)
    return result, None, 200


artifact, MODEL_LOAD_ERROR = load_model()


@app.route("/", methods=["GET"])
def index():
    target_info = artifact.get("target_info", {}) if artifact else {}
    return render_template(
        "index.html",
        result=None,
        error=MODEL_LOAD_ERROR,
        form_values={},
        service_status="ok" if MODEL_LOAD_ERROR is None else "degraded",
        field_specs=FIELD_SPECS,
        target_info=target_info,
    )


@app.route("/", methods=["POST"])
def index_submit():
    form_values = request.form.to_dict()
    normalized = normalize_payload(form_values)
    result, error, status_code = build_prediction(normalized)

    target_info = artifact.get("target_info", {}) if artifact else {}
    if error:
        return render_template(
            "index.html",
            result=None,
            error=error.get("error"),
            missing=error.get("missing", []),
            form_values=form_values,
            service_status="ok" if MODEL_LOAD_ERROR is None else "degraded",
            field_specs=FIELD_SPECS,
            target_info=target_info,
        ), status_code

    return render_template(
        "index.html",
        result=result,
        error=None,
        form_values=form_values,
        service_status="ok" if MODEL_LOAD_ERROR is None else "degraded",
        field_specs=FIELD_SPECS,
        target_info=target_info,
    )


@app.route("/health", methods=["GET"])
def health():
    if MODEL_LOAD_ERROR:
        return jsonify({"status": "error", "error": MODEL_LOAD_ERROR}), 503
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_LOAD_ERROR:
        return jsonify({"error": MODEL_LOAD_ERROR}), 503

    normalized = normalize_payload(request.get_json(silent=True) or {})
    result, error, status_code = build_prediction(normalized)
    if error:
        return jsonify(error), status_code
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
