from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, coerce_feature_types


def prepare_student_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = coerce_feature_types(df)
    return cleaned[FEATURE_COLUMNS].copy()


def summarize_student_profile(row: pd.Series) -> dict[str, float]:
    numeric_row = pd.to_numeric(row[NUMERIC_FEATURES], errors="coerce")
    return {
        "academic_strength": float(numeric_row[["CGPA", "Aptitude_Test_Score", "Resume_Score"]].mean()),
        "technical_depth": float(
            numeric_row[["DSA_Problems_Solved", "LeetCode_Rating", "GitHub_Contributions", "System_Design_Knowledge"]].mean()
        ),
        "interview_readiness": float(
            numeric_row[["Communication_Skills", "Mock_Interview_Score", "Resume_Score"]].mean()
        ),
    }
