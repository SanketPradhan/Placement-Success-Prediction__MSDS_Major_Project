from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_placement_career_success_dataset_2026.csv"

TARGET_COLUMN = "Placement_Status"
SALARY_COLUMN = "Salary_LPA"

NUMERIC_FEATURES = [
    "Age",
    "CGPA",
    "DSA_Problems_Solved",
    "Internships",
    "Certifications",
    "Projects_Count",
    "Communication_Skills",
    "Aptitude_Test_Score",
    "LeetCode_Rating",
    "GitHub_Contributions",
    "Hackathons_Participated",
    "AI_ML_Skill_Level",
    "System_Design_Knowledge",
    "Resume_Score",
    "Mock_Interview_Score",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "College_Tier",
    "Specialization",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    required_columns = required_columns or FEATURE_COLUMNS + [TARGET_COLUMN, SALARY_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")


def coerce_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN, SALARY_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype(str).str.strip()
    return cleaned


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy().drop_duplicates()
    validate_columns(cleaned)
    cleaned = coerce_feature_types(cleaned)
    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, SALARY_COLUMN])
    return cleaned


def derive_target(df: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    if df[TARGET_COLUMN].nunique(dropna=True) > 1:
        target = df[TARGET_COLUMN].astype(int)
        target_info = {
            "source": TARGET_COLUMN,
            "kind": "placement_status",
            "positive_label": "Placed",
            "negative_label": "Not placed",
            "threshold": None,
            "description": "Predicts the dataset's placement status column.",
        }
        return target, target_info

    threshold = float(df[SALARY_COLUMN].median())
    target = (df[SALARY_COLUMN] >= threshold).astype(int)
    target_info = {
        "source": SALARY_COLUMN,
        "kind": "salary_threshold",
        "positive_label": "Strong placement outcome",
        "negative_label": "Below median outcome",
        "threshold": threshold,
        "description": f"Uses {SALARY_COLUMN} >= {threshold:.2f} LPA as the success proxy.",
    }
    return target, target_info


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    cleaned = clean_dataset(df)
    X = cleaned[FEATURE_COLUMNS].copy()
    y, target_info = derive_target(cleaned)
    return X, y, target_info
