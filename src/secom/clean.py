"""Unsupervised feature screening. Decisions are made on the training window only."""
import pandas as pd


def screen_features(X_train: pd.DataFrame, max_missing: float = 0.5, min_unique: int = 3):
    """Return (kept_columns, drop_log DataFrame).

    Steps, in order:
      1. drop features with > max_missing fraction missing
      2. drop constant features (one unique non-NaN value)
      3. drop near-constant features (< min_unique distinct values, or one value covers >= 95% of rows)
      4. drop exact duplicate columns (keep the first)
    """
    log = []
    cols = list(X_train.columns)

    miss = X_train[cols].isna().mean()
    for c in miss[miss > max_missing].index:
        log.append((c, "missing>50%", f"{miss[c]:.1%} missing"))
    cols = [c for c in cols if miss[c] <= max_missing]

    nun = X_train[cols].nunique(dropna=True)
    for c in nun[nun <= 1].index:
        log.append((c, "constant", "1 unique value"))
    cols = [c for c in cols if nun[c] > 1]

    near = []
    for c in cols:
        s = X_train[c].dropna()
        top_share = s.value_counts(normalize=True).iloc[0]
        if s.nunique() < min_unique or top_share >= 0.95:
            near.append(c)
            log.append((c, "near-zero-variance", f"{s.nunique()} unique, top value {top_share:.0%}"))
    cols = [c for c in cols if c not in near]

    dup_mask = X_train[cols].T.duplicated()
    dups = list(pd.Index(cols)[dup_mask.values])
    for c in dups:
        log.append((c, "duplicate", "identical to an earlier column"))
    cols = [c for c in cols if c not in dups]

    drop_log = pd.DataFrame(log, columns=["feature", "reason", "detail"])
    return cols, drop_log
