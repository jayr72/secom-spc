"""Load the UCI SECOM files into one time-ordered DataFrame."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_secom(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, meta) sorted by timestamp.

    X    : 590 sensor columns S000..S589 (float, NaN = missing)
    meta : columns ['timestamp', 'fail'] where fail = 1 for a line-test fail, 0 for pass
    """
    X = pd.read_csv(raw_dir / "secom.data", sep=r"\s+", header=None)
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        raw_dir / "secom_labels.data", sep=" ", header=None, names=["label", "timestamp"], quotechar='"'
    )
    meta = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S"),
            "fail": (labels["label"] == 1).astype(int),  # -1 = pass, 1 = fail
        }
    )

    order = meta["timestamp"].sort_values(kind="stable").index
    X = X.loc[order].reset_index(drop=True)
    meta = meta.loc[order].reset_index(drop=True)
    return X, meta


def time_split(n: int, frac: float) -> int:
    """Index where the first `frac` of a time-ordered table ends."""
    return int(round(n * frac))
