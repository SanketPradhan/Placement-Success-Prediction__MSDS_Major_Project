from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import DATA_PATH, load_dataset, validate_columns


def ingest_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = load_dataset(path)
    validate_columns(df)
    return df


def get_basic_profile(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    dataset = ingest_data()
    print(get_basic_profile(dataset))
