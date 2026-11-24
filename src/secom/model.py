"""Fail prediction and feature ranking.

Three models on a time-ordered split:
  * L1 logistic regression (impute -> scale -> L1-LR, class_weight='balanced'; C tuned by CV)
  * Sparse logistic regression (top-10 ANOVA features selected inside the pipeline)
  * HistGradientBoosting (impute -> HGB, class_weight='balanced')
Three rankings combined into a consensus:
  * univariate Mann-Whitney U (BH-FDR adjusted)
  * |L1 coefficient| on standardized features
  * mean |SHAP| from the boosted model
"""
import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

SEED = 42


def l1_pipeline(C: float = 0.1) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=5000)),
        ]
    )


def sparse_lr_pipeline(k: int = 10) -> Pipeline:
    """Top-k univariate (ANOVA F) features, chosen inside each fit, then ridge logistic regression."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        ]
    )


def cv_scores(model, X, y) -> dict:
    """Out-of-fold PR-AUC / ROC-AUC from stratified 5-fold CV inside the training window."""
    oof = cross_val_predict(model, X, y, cv=cv(), method="predict_proba")[:, 1]
    return {"cv_PR_AUC": average_precision_score(y, oof), "cv_ROC_AUC": roc_auc_score(y, oof)}


def hgb_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                    l2_regularization=1.0, class_weight="balanced", random_state=SEED,
                ),
            ),
        ]
    )


def cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def tune_l1(X, y) -> tuple[Pipeline, float]:
    gs = GridSearchCV(l1_pipeline(), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="average_precision", cv=cv())
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_["clf__C"]


def pick_threshold(model, X, y) -> float:
    """Threshold maximizing balanced accuracy on out-of-fold training predictions."""
    oof = cross_val_predict(model, X, y, cv=cv(), method="predict_proba")[:, 1]
    grid = np.unique(np.quantile(oof, np.linspace(0.5, 0.995, 200)))
    scores = [balanced_accuracy_score(y, oof >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(model, X, y, threshold: float) -> dict:
    p = model.predict_proba(X)[:, 1]
    pred = p >= threshold
    return {
        "PR_AUC": average_precision_score(y, p),
        "ROC_AUC": roc_auc_score(y, p),
        "recall_fail": recall_score(y, pred),
        "precision_fail": precision_score(y, pred, zero_division=0),
        "balanced_error_rate": 1 - balanced_accuracy_score(y, pred),
        "threshold": threshold,
        "flagged": int(pred.sum()),
        "n": len(y),
        "fails": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def mann_whitney_rank(X: pd.DataFrame, y) -> pd.DataFrame:
    y = np.asarray(y)
    rows = []
    for c in X.columns:
        a = X.loc[y == 1, c].dropna()
        b = X.loc[y == 0, c].dropna()
        if len(a) < 5 or a.nunique() + b.nunique() < 3:
            rows.append((c, np.nan, 1.0))
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        auc = stat / (len(a) * len(b))
        rows.append((c, auc, p))
    df = pd.DataFrame(rows, columns=["feature", "mw_auc", "mw_p"]).set_index("feature")
    df["mw_q"] = multipletests(df["mw_p"].fillna(1.0), method="fdr_bh")[1]
    df["mw_effect"] = (df["mw_auc"] - 0.5).abs()
    return df


def l1_importance(fitted: Pipeline, cols) -> pd.Series:
    return pd.Series(np.abs(fitted.named_steps["clf"].coef_.ravel()), index=cols, name="l1_abs_coef")


def shap_importance(fitted: Pipeline, X: pd.DataFrame) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    Xi = pd.DataFrame(fitted.named_steps["impute"].transform(X), columns=X.columns, index=X.index)
    explainer = shap.TreeExplainer(fitted.named_steps["clf"])
    sv = explainer.shap_values(Xi)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return pd.Series(np.abs(sv).mean(axis=0), index=X.columns, name="mean_abs_shap"), sv, Xi


def consensus(mw: pd.DataFrame, l1: pd.Series, sh: pd.Series) -> pd.DataFrame:
    df = pd.concat([mw, l1, sh], axis=1)
    df["rank_mw"] = df["mw_p"].rank(method="min")
    df["rank_l1"] = (-df["l1_abs_coef"]).rank(method="min")
    df["rank_shap"] = (-df["mean_abs_shap"]).rank(method="min")
    df["mean_rank"] = df[["rank_mw", "rank_l1", "rank_shap"]].mean(axis=1)
    df["consensus_rank"] = df["mean_rank"].rank(method="first").astype(int)
    return df.sort_values("consensus_rank")


def bootstrap_stability(X: pd.DataFrame, y, C: float, n_boot: int = 30, top_k: int = 15) -> pd.Series:
    """Fraction of stratified bootstrap refits in which each feature lands in the consensus top-k."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    counts = pd.Series(0, index=X.columns, dtype=float)
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        Xb, yb = X.iloc[idx].reset_index(drop=True), y[idx]
        l1 = l1_pipeline(C).fit(Xb, yb)
        hgb = hgb_pipeline().fit(Xb, yb)
        cons = consensus(mann_whitney_rank(Xb, yb), l1_importance(l1, X.columns), shap_importance(hgb, Xb)[0])
        counts[cons.index[:top_k]] += 1
    return (counts / n_boot).rename("top15_frequency")
