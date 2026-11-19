"""Summaries and figures of the backtest in results/backtest_triangles.csv."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reserving import methods as m

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
METHODS = {"cl_paid": "Chain ladder, paid", "cl_incurred": "Chain ladder, incurred", "bf_paid": "Bornhuetter-Ferguson, paid",
           "cape_cod_paid": "Cape Cod, paid"}
SIGMA_RES = {"ppauto": 0.09, "comauto": 0.09, "othliab": 0.11}      # Solvency II standard formula, reserve risk by line
NAMES = {"ppauto": "Private passenger auto", "comauto": "Commercial auto", "wkcomp": "Workers' compensation", "othliab": "Other liability"}


def actual(g: pd.DataFrame, k: str) -> pd.Series:
    return g["actual_incurred_reserve"] if k == "cl_incurred" else g["actual_reserve"]


def method_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, g in d.groupby("line"):
        for k, label in METHODS.items():
            act = actual(g, k)
            err = (g[f"pred_{k}"] - act) / g["premium_1989_1997"]
            latest = g["latest_incurred"] if k == "cl_incurred" else g["latest_paid"]
            rows.append({"line": line, "method": label, "triangles": len(g),
                         "median_error_pct_of_premium": 100 * float(err.median()),
                         "mean_error_pct_of_premium": 100 * float(err.mean()),
                         "rmse_pct_of_premium": 100 * float(np.sqrt((err ** 2).mean())),
                         "share_underestimated": float((err < 0).mean()),
                         "median_ultimate_ratio": float(((g[f"pred_{k}"] + latest) / (act + latest)).median())})
    return pd.DataFrame(rows)


def winners(d: pd.DataFrame) -> pd.DataFrame:
    paid = ["cl_paid", "bf_paid", "cape_cod_paid"]
    out = []
    for line, g in d.groupby("line"):
        e = pd.DataFrame({k: (g[f"pred_{k}"] - g["actual_reserve"]).abs() for k in paid})
        best = e.idxmin(axis=1).value_counts(normalize=True)
        out.append({"line": line, **{METHODS[k]: float(best.get(k, 0.0)) for k in paid}})
    return pd.DataFrame(out)


def coverage(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, g in d.groupby("line"):
        a = g["actual_reserve"]
        rows.append({"line": line, "triangles": len(g),
                     "mack_lognormal_50": float(((a >= g["mack_q25"]) & (a <= g["mack_q75"])).mean()),
                     "mack_lognormal_90": float(((a >= g["mack_q05"]) & (a <= g["mack_q95"])).mean()),
                     "bootstrap_50": float(((a >= g["boot_q25"]) & (a <= g["boot_q75"])).mean()),
                     "bootstrap_90": float(((a >= g["boot_q05"]) & (a <= g["boot_q95"])).mean()),
                     "above_mack_995": float((a > g["mack_q995"]).mean()), "above_bootstrap_995": float((a > g["boot_q995"]).mean()),
                     "actual_above_prediction": float((a > g["pred_cl_paid"]).mean())})
    return pd.DataFrame(rows)


def solvency2(d: pd.DataFrame) -> pd.DataFrame:
    big = d[d["premium_1997"] >= d.groupby("line")["premium_1997"].transform("median")]
    rows = []
    for scope, frame in (("all triangles", d), ("larger half by premium", big)):
        for line, g in frame.groupby("line"):
            if line not in SIGMA_RES:
                continue
            sg = SIGMA_RES[line]
            rho = m.solvency2_reserve_capital(1.0, sg)
            a = g["actual_reserve"]
            rows.append({"scope": scope, "line": line, "triangles": len(g), "sigma_reserve": sg,
                         "capital_995_pct_of_reserve": 100 * rho,
                         "median_mack_cv": float(g["mack_cv"].median()), "median_bootstrap_cv": float(g["boot_cv"].median()),
                         "median_bootstrap_995_pct_above_reserve": 100 * float((g["boot_q995"] / g["pred_cl_paid"] - 1).median()),
                         "actual_above_standard_formula_995": float((a > g["pred_cl_paid"] * (1 + rho)).mean()),
                         "share_with_bootstrap_cv_above_sigma": float((g["boot_cv"] > sg).mean())})
    return pd.DataFrame(rows)


def figures(d: pd.DataFrame, c: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, (line, g) in zip(axs, d.groupby("line")):
        data = [100 * ((g[f"pred_{k}"] - actual(g, k)) / g["premium_1989_1997"]).clip(-60, 60) for k in METHODS]
        ax.boxplot(data, showfliers=False)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(1, 5), ["CL paid", "CL inc.", "BF", "Cape Cod"], rotation=30)
        ax.set_title(NAMES[line], fontsize=9)
    axs[0].set_ylabel("Reserve error, % of premium 1989-1997")
    fig.tight_layout()
    fig.savefig(FIG / "reserve_error.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(c))
    for j, (col, lab) in enumerate([("mack_lognormal_90", "Mack, lognormal"), ("bootstrap_90", "Bootstrap")]):
        ax.bar(x + (j - 0.5) * 0.32, c[col], width=0.3, label=lab)
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xticks(x, [NAMES[ln] for ln in c["line"]], rotation=15, fontsize=8)
    ax.set_ylabel("Share of triangles with the actual\nreserve inside the 90% interval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "interval_coverage.png", dpi=150)
    plt.close(fig)


def main() -> None:
    d = pd.read_csv(RES / "backtest_triangles.csv")
    t, w, c, s = method_table(d), winners(d), coverage(d), solvency2(d)
    t.to_csv(RES / "backtest_by_method.csv", index=False, float_format="%.4g")
    w.to_csv(RES / "backtest_winners.csv", index=False, float_format="%.3f")
    c.to_csv(RES / "backtest_coverage.csv", index=False, float_format="%.3f")
    s.to_csv(RES / "backtest_solvency2.csv", index=False, float_format="%.4g")
    figures(d, c)
    print(len(d), "triangles\n", t.round(3).to_string(), "\n", w.round(3).to_string(), "\n", c.round(3).to_string(), "\n", s.round(3).to_string())


if __name__ == "__main__":
    main()
