"""Run the full SECOM analysis end to end.

    .venv/bin/python run_all.py            # full run (includes 30 bootstrap refits)
    .venv/bin/python run_all.py --fast     # 5 bootstraps, for quick checks

Writes CSVs + results.json to reports/ and PNGs to reports/figures/.
"""
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")  # avoids OpenMP thread thrash in HistGradientBoosting on macOS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from secom import capability as capm  # noqa: E402
from secom import model as M  # noqa: E402
from secom import plots as P  # noqa: E402
from secom.clean import screen_features  # noqa: E402
from secom.load import load_secom, time_split  # noqa: E402
from secom.spc import imr_limits, moving_range, p_chart, run_rules  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FAST = "--fast" in sys.argv
TRAIN_FRAC = 0.70  # model train / test split
BASELINE_FRAC = 0.30  # SPC Phase I window
N_CHART = 4  # top predictors to chart
N_CONTROL = 2  # stable "control" sensors to chart
R = {}  # everything that goes into results.json


def main():
    P.setup()
    np.random.seed(M.SEED)
    X, meta = load_secom()
    n = len(X)
    y = meta["fail"].to_numpy()
    R["data"] = {
        "units": n,
        "sensors_raw": X.shape[1],
        "fails": int(y.sum()),
        "fail_rate": float(y.mean()),
        "start": str(meta.timestamp.min()),
        "end": str(meta.timestamp.max()),
        "days": int(meta.timestamp.dt.date.nunique()),
        "missing_cells_pct": float(X.isna().mean().mean()),
    }

    # ---------- Phase 1: audit & screening (decisions use the train window only) ----------
    k = time_split(n, TRAIN_FRAC)
    cols, drop_log = screen_features(X.iloc[:k])
    drop_log.to_csv(REP / "feature_drops.csv", index=False)
    R["screening"] = {
        "kept": len(cols),
        "dropped_by_reason": drop_log["reason"].value_counts().to_dict(),
        "train_units": k,
        "test_units": n - k,
        "train_fails": int(y[:k].sum()),
        "test_fails": int(y[k:].sum()),
        "train_end": str(meta.timestamp.iloc[k - 1]),
    }
    fig_missingness(X)

    # ---------- Phase 2: yield over time ----------
    yield_over_time(meta)

    # ---------- Phase 5: model + feature ranking ----------
    Xtr, Xte, ytr, yte = X.iloc[:k][cols], X.iloc[k:][cols], y[:k], y[k:]
    l1, best_C = M.tune_l1(Xtr, ytr)
    makers = {
        "L1_logistic": lambda: M.l1_pipeline(best_C),
        "Sparse_LR_top10": M.sparse_lr_pipeline,
        "HistGradientBoosting": M.hgb_pipeline,
    }
    fitted, metrics = {}, {}
    for name, make in makers.items():
        fitted[name] = make().fit(Xtr, ytr)
        thr = M.pick_threshold(make(), Xtr, ytr)
        metrics[name] = {**M.cv_scores(make(), Xtr, ytr), **M.evaluate(fitted[name], Xte, yte, thr)}
    hgb = fitted["HistGradientBoosting"]
    metrics["all_pass_baseline"] = {"accuracy": float(1 - yte.mean()), "recall_fail": 0.0, "PR_AUC": float(yte.mean())}
    sel = fitted["Sparse_LR_top10"].named_steps["select"].get_support()
    R["model"] = {
        "best_C": best_C,
        "l1_nonzero": int((l1.named_steps["clf"].coef_ != 0).sum()),
        "sparse_lr_features": list(np.array(cols)[sel]),
        "test": metrics,
    }
    pd.DataFrame(metrics).T.to_csv(REP / "model_metrics.csv")

    mw = M.mann_whitney_rank(Xtr, ytr)
    l1_imp = M.l1_importance(l1, cols)
    sh_imp, _, _ = M.shap_importance(hgb, Xtr)
    cons = M.consensus(mw, l1_imp, sh_imp)
    stab = M.bootstrap_stability(Xtr, ytr, best_C, n_boot=5 if FAST else 30)
    cons = cons.join(stab)
    miss = X[cols].isna().mean().rename("missing_frac")
    cons = cons.join(miss)
    top = cons.head(15)
    top[["consensus_rank", "rank_mw", "rank_l1", "rank_shap", "mw_auc", "mw_q", "l1_abs_coef", "mean_abs_shap",
         "top15_frequency", "missing_frac"]].round(4).to_csv(REP / "top_features.csv")
    R["top_features"] = top[["rank_mw", "rank_l1", "rank_shap", "mw_auc", "mw_q", "top15_frequency"]].round(4).reset_index().to_dict("records")
    R["fdr_significant_q05"] = int((mw["mw_q"] < 0.05).sum())
    fig_pr_curves(fitted, Xte, yte)
    fig_shap(hgb, Xte, list(top.index[:3]))
    fig_rank_table(top)
    top_driver_story(X, meta, top.index[0])

    # ---------- Phase 3 & 4: control charts + capability ----------
    chart_sensors = pick_chart_sensors(cons, X, meta)
    R["chart_sensors"] = chart_sensors
    spc_and_capability(X, meta, chart_sensors)

    with open(REP / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))


# --------------------------------------------------------------------------- helpers
def fig_missingness(X):
    m = X.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(range(len(m)), m.values, width=1.0, color=P.SERIES)
    ax.axhline(0.5, color=P.ACCENT, lw=1, ls="--", label="50% drop threshold")
    ax.set(xlabel="Sensors (sorted by missingness)", ylabel="Fraction missing", title="Missing data per sensor", xlim=(0, len(m)))
    ax.legend()
    fig.savefig(FIG / "01_missingness.png")
    plt.close(fig)


def yield_over_time(meta):
    day = meta.timestamp.dt.floor("D")
    g = meta.groupby(day)["fail"].agg(["sum", "count"])
    pc = p_chart(g["sum"], g["count"])
    pc.to_csv(REP / "p_chart_daily.csv")
    wk = meta.groupby(meta.timestamp.dt.to_period("W"))["fail"].agg(["count", "sum"])
    wk["first_pass_yield"] = 1 - wk["sum"] / wk["count"]
    wk.to_csv(REP / "weekly_yield.csv")
    R["yield"] = {
        "overall_fpy": float(1 - meta.fail.mean()),
        "p_bar": float(pc["center"].iloc[0]),
        "days_above_ucl": [str(d.date()) for d in pc.index[pc.out_high]],
        "days_above_ucl_detail": pc.loc[pc.out_high, ["n", "fails", "p", "ucl"]].round(3).reset_index().astype(str).to_dict("records"),
        "weekly_fpy_min": float(wk["first_pass_yield"].min()),
        "weekly_fpy_min_week": str(wk["first_pass_yield"].idxmin()),
        "weekly_fpy_max": float(wk["first_pass_yield"].max()),
    }

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.step(pc.index, pc["ucl"], where="mid", color=P.LIMIT, lw=1, ls="--", label="UCL (varies with n)")
    a1.axhline(pc["center"].iloc[0], color=P.LIMIT, lw=1, label=f"p̄ = {pc['center'].iloc[0]:.3f}")
    a1.plot(pc.index, pc["p"], "-", color=P.MUTED, lw=0.8)
    a1.plot(pc.index, pc["p"], "o", color=P.SERIES, ms=3)
    o = pc[pc.out_high]
    a1.plot(o.index, o["p"], "o", color=P.ACCENT, ms=6, label="Above UCL")
    a1.set(ylabel="Daily fail fraction", title="p-chart: daily line-test fail rate")
    a1.legend(loc="upper right")
    a2.bar(pc.index, pc["n"], color=P.MUTED, width=0.8)
    a2.set(ylabel="Units / day")
    fig.autofmt_xdate()
    fig.savefig(FIG / "02_p_chart.png")
    plt.close(fig)


def fig_pr_curves(fitted, Xte, yte):
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    styles = {"L1_logistic": ("L1 logistic", P.SERIES), "Sparse_LR_top10": ("Sparse LR (top 10)", P.LIMIT),
              "HistGradientBoosting": ("Gradient boosting", P.ACCENT)}
    for key, mdl in fitted.items():
        name, c = styles[key]
        p = mdl.predict_proba(Xte)[:, 1]
        pr, rc, _ = precision_recall_curve(yte, p)
        ap = R["model"]["test"][key]["PR_AUC"]
        ax.plot(rc, pr, color=c, lw=1.5, label=f"{name} (PR-AUC {ap:.2f})")
    ax.axhline(yte.mean(), color=P.MUTED, ls=":", lw=1, label=f"No-skill ({yte.mean():.2f})")
    ax.set(xlabel="Recall (fails caught)", ylabel="Precision", title="Held-out test window (last 30%)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    fig.savefig(FIG / "05_pr_curve.png")
    plt.close(fig)


def fig_shap(hgb, Xte, top3):
    _, sv, Xi = M.shap_importance(hgb, Xte)
    plt.figure()
    shap.summary_plot(sv, Xi, max_display=15, show=False, plot_size=(7, 5))
    plt.title("SHAP: top 15 sensors (test window)")
    plt.savefig(FIG / "06_shap_beeswarm.png")
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, f in zip(axes, top3):
        j = list(Xi.columns).index(f)
        ax.scatter(Xi[f], sv[:, j], s=8, color=P.SERIES, alpha=0.6)
        ax.axhline(0, color=P.MUTED, lw=0.8)
        ax.set(xlabel=f, ylabel="SHAP value (→ fail)", title=f"Dependence: {f}")
    fig.tight_layout()
    fig.savefig(FIG / "07_shap_dependence.png")
    plt.close(fig)


def fig_rank_table(top):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    t = top.iloc[::-1]
    ax.barh(t.index, t["top15_frequency"], color=P.SERIES)
    for i, (f, row) in enumerate(t.iterrows()):
        ax.text(row["top15_frequency"] + 0.01, i, f"MW #{int(row.rank_mw)} · L1 #{int(row.rank_l1)} · SHAP #{int(row.rank_shap)}",
                va="center", fontsize=7, color=P.INK)
    ax.set(xlim=(0, 1.45), xlabel="Share of bootstrap refits where sensor is in consensus top 15",
           title="Consensus top 15 fail predictors and ranking stability")
    fig.savefig(FIG / "08_top15_stability.png")
    plt.close(fig)


def pick_chart_sensors(cons, X, meta):
    """Top predictors, plus the most *stable* weakly-predictive sensors as an in-control contrast."""
    cont = [c for c in cons.index if X[c].isna().mean() <= 0.10 and X[c].nunique() > 30]
    top = cont[:N_CHART]
    b = time_split(len(X), BASELINE_FRAC)
    y = meta.fail.to_numpy()
    cand = []
    for c in cons.index[100:]:  # outside the top 100 by consensus rank
        x = X[c]
        if x.isna().mean() > 0.02 or x.nunique() < 100:
            continue
        lim = imr_limits(x.iloc[:b][y[:b] == 0].dropna())
        if lim.sigma == 0:
            continue
        cand.append((run_rules(x.iloc[b:].dropna().to_numpy(), lim.center, lim.sigma)["any"].mean(), c))
    calm = [c for _, c in sorted(cand)[:N_CONTROL]]
    return {"predictors": top, "controls": calm}


def top_driver_story(X, meta, s):
    """Fail rate by sensor quintile and the month-by-month drift of the #1 driver."""
    q = pd.qcut(X[s], 5)
    by_q = meta.groupby(q, observed=True)["fail"].agg(["mean", "count"])
    month = meta.timestamp.dt.to_period("M")
    by_m = pd.DataFrame({"sensor_mean": X[s], "fail_rate": meta.fail, "n": 1}).groupby(month).agg(
        sensor_mean=("sensor_mean", "mean"), fail_rate=("fail_rate", "mean"), n=("n", "sum"))
    by_m.to_csv(REP / "top_driver_monthly.csv")
    R["top_driver"] = {
        "sensor": s,
        "fail_rate_by_quintile": [{"bin": str(i), "fail_rate": round(float(r["mean"]), 4), "n": int(r["count"])} for i, r in by_q.iterrows()],
        "monthly": [{"month": str(i), "sensor_mean": round(float(r.sensor_mean), 3), "fail_rate": round(float(r.fail_rate), 4), "n": int(r.n)}
                    for i, r in by_m.iterrows()],
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    labels = [f"Q{i + 1}" for i in range(len(by_q))]
    a1.bar(labels, by_q["mean"], color=[P.MUTED] * (len(by_q) - 1) + [P.ACCENT])
    a1.axhline(meta.fail.mean(), color=P.LIMIT, ls="--", lw=1, label=f"Overall {meta.fail.mean():.1%}")
    for i, v in enumerate(by_q["mean"]):
        a1.text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=8)
    a1.set(ylabel="Fail rate", xlabel=f"{s} quintile (low → high)", title=f"Fail rate by {s} level")
    a1.legend()
    x = np.arange(len(by_m))
    a2.plot(x, by_m["sensor_mean"], "o-", color=P.SERIES, label=f"{s} monthly mean")
    a2.set_ylabel(f"{s} mean", color=P.SERIES)
    a2.set_xticks(x, [str(m) for m in by_m.index])
    a3 = a2.twinx()
    a3.plot(x, by_m["fail_rate"], "s--", color=P.ACCENT, label="Fail rate")
    a3.set_ylabel("Fail rate", color=P.ACCENT)
    a3.spines["right"].set_visible(True)
    a3.grid(False)
    a2.set_title(f"{s} drifted down and fail rate fell with it")
    fig.tight_layout()
    fig.savefig(FIG / "10_top_driver_drift.png")
    plt.close(fig)


def spc_and_capability(X, meta, sensors):
    b = time_split(len(X), BASELINE_FRAC)
    rows, link = [], []
    names = sensors["predictors"] + sensors["controls"]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 2.0 * len(names)), sharex=True)
    for ax, s in zip(axes, names):
        x = X[s]
        base = x.iloc[:b][meta.fail.iloc[:b] == 0].dropna()  # Phase I: pass units only
        lim = imr_limits(base)
        lsl, usl = np.percentile(base, [0.5, 99.5])

        ph2 = pd.DataFrame({"t": meta.timestamp.iloc[b:], "x": x.iloc[b:], "fail": meta.fail.iloc[b:]}).dropna()
        flags = run_rules(ph2["x"].to_numpy(), lim.center, lim.sigma)
        flags.index = ph2.index
        mr2 = moving_range(ph2["x"])

        # how well do SPC signals line up with fails?
        f_in = ph2.loc[flags["any"], "fail"]
        f_out = ph2.loc[~flags["any"], "fail"]
        link.append({
            "sensor": s, "role": "predictor" if s in sensors["predictors"] else "control",
            "phase2_points": len(ph2), "flagged_points": int(flags["any"].sum()),
            "flagged_pct": float(flags["any"].mean()),
            **{f"{r}_points": int(flags[r].sum()) for r in ["R1", "R2", "R3", "R4"]},
            "fail_rate_flagged": float(f_in.mean()) if len(f_in) else np.nan,
            "fail_rate_not_flagged": float(f_out.mean()),
            "mr_above_ucl": int((mr2 > lim.mr_ucl).sum()),
        })

        cap = capm.cp_cpk(ph2["x"], lsl, usl)
        pct = capm.percentile_capability(ph2["x"], lsl, usl)
        normal, ad = capm.is_normal(ph2["x"])
        lsl4, usl4 = lim.center - 4 * lim.sigma, lim.center + 4 * lim.sigma
        cap4 = capm.cp_cpk(ph2["x"], lsl4, usl4)
        in_control = flags["any"].mean() < 0.05 and flags["R1"].mean() < 0.01
        rows.append({
            "sensor": s, "role": link[-1]["role"], "LSL_proxy": lsl, "USL_proxy": usl, **cap,
            **pct, "AD_stat": ad, "normal_AD_5pct": normal, "Cpk_pm4sigma_spec": cap4["Cpk"],
            "Ppk_pm4sigma_spec": cap4["Ppk"], "in_control": in_control,
            # 0.5/99.5 pct specs = +/-2.576 sigma, so an unchanged normal process scores Pp = 2.576/3 = 0.859.
            # Ratio > 1: Phase II is tighter than the Phase I baseline; < 1: it got wider or shifted.
            "Pp_vs_unchanged_ref": cap["Pp"] / (2.576 / 3),
            "AIAG_band_proxy_specs": capm.verdict(cap["Ppk"] if normal else pct["Cpk_pct"]),
        })

        # chart (individuals only for the stacked overview; MR stats in CSV)
        bx = x.iloc[:b].dropna()
        ax.plot(meta.timestamp.iloc[bx.index], bx, color=P.MUTED, lw=0.6)
        ax.plot(ph2["t"], ph2["x"], color=P.SERIES, lw=0.6)
        ax.plot(ph2.loc[flags["any"], "t"], ph2.loc[flags["any"], "x"], "o", color=P.LIMIT, ms=2.5, mfc="none", mew=0.7,
                label="Run-rule flag")
        fl = ph2[ph2.fail == 1]
        ax.plot(fl["t"], fl["x"], "x", color=P.ACCENT, ms=4, mew=1, label="Line-test fail")
        for yv in (lim.ucl, lim.lcl):
            P.limit_line(ax, yv)
        ax.axhline(lim.center, color=P.LIMIT, lw=0.8)
        ax.axvline(meta.timestamp.iloc[b], color=P.INK, lw=0.8, ls=":")
        tag = "predictor" if s in sensors["predictors"] else "control"
        ax.set_title(f"{s} ({tag}) · {flags['any'].mean():.0%} of Phase II points flagged", loc="left", fontsize=9)
        lo, hi = np.nanpercentile(x, [0.2, 99.8])
        pad = (hi - lo) * 0.25
        ax.set_ylim(min(lo, lim.lcl) - pad, max(hi, lim.ucl) + pad)
    axes[0].legend(loc="upper right", ncol=2, fontsize=7)
    fig.suptitle("I-charts: limits from Phase I pass units (grey), applied to Phase II (dark); dotted line = phase boundary",
                 y=1.0, fontsize=9.5, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "03_imr_overview.png")
    plt.close(fig)

    # a full I-MR pair for the #1 predictor (the classic two-panel chart)
    fig_imr_pair(X, meta, sensors["predictors"][0], b)
    fig_capability_hist(X, meta, rows, b)

    # long-free, Minitab-friendly export: one row per unit, one column per charted sensor
    mt = pd.DataFrame({"timestamp": meta.timestamp, "phase": np.where(np.arange(len(X)) < b, "I", "II"),
                       "fail": meta.fail})
    mt = pd.concat([mt, X[names]], axis=1)
    mt.to_csv(REP / "minitab_export.csv", index=False)
    pd.DataFrame([{"sensor": r["sensor"], "LSL": r["LSL_proxy"], "USL": r["USL_proxy"]} for r in rows]).to_csv(
        REP / "minitab_spec_limits.csv", index=False)

    cap_df = pd.DataFrame(rows)
    cap_df.round(4).to_csv(REP / "capability_table.csv", index=False)
    link_df = pd.DataFrame(link)
    link_df.round(4).to_csv(REP / "spc_fail_linkage.csv", index=False)
    R["capability"] = cap_df.round(3).to_dict("records")
    R["spc_linkage"] = link_df.round(4).to_dict("records")


def fig_imr_pair(X, meta, s, b):
    base = X[s].iloc[:b][meta.fail.iloc[:b] == 0].dropna()
    lim = imr_limits(base)
    d = pd.DataFrame({"t": meta.timestamp, "x": X[s], "fail": meta.fail}).dropna()
    mr = np.r_[np.nan, moving_range(d["x"])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    a1.plot(d["t"], d["x"], color=P.SERIES, lw=0.6)
    out = (d["x"] > lim.ucl) | (d["x"] < lim.lcl)
    a1.plot(d.loc[out, "t"], d.loc[out, "x"], "o", color=P.LIMIT, ms=3, mfc="none", label="Beyond 3σ (R1)")
    f = d[d.fail == 1]
    a1.plot(f["t"], f["x"], "x", color=P.ACCENT, ms=4, label="Line-test fail")
    for yv, lab in [(lim.ucl, f"UCL {lim.ucl:.3g}"), (lim.lcl, f"LCL {lim.lcl:.3g}")]:
        P.limit_line(a1, yv)
        a1.text(d["t"].iloc[-1], yv, " " + lab, va="center", fontsize=7, color=P.LIMIT)
    a1.axhline(lim.center, color=P.LIMIT, lw=0.8)
    a1.axvline(meta.timestamp.iloc[b], color=P.INK, lw=0.8, ls=":")
    a1.set(ylabel=s, title=f"I-MR chart for {s} (top-ranked fail predictor)")
    a1.legend(loc="upper left", fontsize=7)
    a2.plot(d["t"], mr, color=P.SERIES, lw=0.6)
    P.limit_line(a2, lim.mr_ucl)
    a2.axhline(lim.mr_bar, color=P.LIMIT, lw=0.8)
    a2.set(ylabel="Moving range")
    a2.set_ylim(0, np.nanpercentile(mr, 99.5) * 1.3)
    fig.autofmt_xdate()
    fig.savefig(FIG / "04_imr_top_sensor.png")
    plt.close(fig)


def fig_capability_hist(X, meta, rows, b):
    from scipy import stats

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, r in zip(axes.ravel(), rows):
        x = X[r["sensor"]].iloc[b:].dropna()
        lo, hi = np.percentile(x, [0.5, 99.5])
        span = hi - lo
        xs = x[(x >= lo - span) & (x <= hi + span)]
        ax.hist(xs, bins=40, color=P.MUTED, density=True)
        g = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(g, stats.norm.pdf(g, r["mean"], r["sigma_overall"]), color=P.SERIES, lw=1.2)
        for v in (r["LSL_proxy"], r["USL_proxy"]):
            ax.axvline(v, color=P.ACCENT, lw=1, ls="--")
        ax.set_title(f"{r['sensor']} · Ppk {r['Ppk']:.2f} · Cpk {r['Cpk']:.2f}", fontsize=8.5)
        ax.set_yticks([])
    fig.suptitle("Phase II distributions vs proxy spec limits (red dashed = 0.5/99.5 pct of Phase I pass units)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "09_capability_hist.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
