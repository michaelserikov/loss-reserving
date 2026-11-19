"""Backtest of the reserving methods on all usable company triangles of four lines of business.

Each method sees only the upper triangle (calendar years to 1997) and predicts the reserve of accident years 1989 to 1997;
the outcome is the lower triangle of the same database (development year 10). Writes results/backtest_triangles.csv and
the summary tables; about 10 minutes on 6 cores for 1,000 bootstrap draws per triangle.

    python scripts/run_backtest.py [n_bootstrap]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from reserving import methods as m
from reserving import triangle as tri

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SIGMA_RES = {"ppauto": 0.09, "comauto": 0.09, "othliab": 0.11}      # Solvency II standard formula, reserve risk per line
ROWS = range(1, 10)                                                    # accident years 1989 to 1997 have a reserve


def one(t: tri.Triangle) -> dict | None:
    paid, inc, prem = t.upper(t.paid), t.upper(t.incurred), t.premium
    if not tri.usable(t):
        return None
    try:
        cl = m.chain_ladder(paid)
        cli = m.chain_ladder(inc)
        mk = m.mack(paid)
    except Exception:
        return None
    if not np.isfinite(cl["cdf"]).all() or cl["reserve"][1:].sum() <= 0:
        return None
    diag_p = m.latest_diagonal(paid)
    actual_res = (t.paid[:, -1] - diag_p)[1:].sum()
    actual_inc_res = (t.incurred[:, -1] - m.latest_diagonal(inc))[1:].sum()
    # a priori loss ratio for BF: chain ladder ultimate loss ratio of accident years 1988 to 1990
    elr = float(cl["ultimate"][:3].sum() / prem[:3].sum())
    bf = m.bornhuetter_ferguson(paid, prem, elr)
    cc = m.cape_cod(paid, prem)
    pm = float(prem[1:].sum())
    pred = {"cl_paid": cl["reserve"][1:].sum(), "cl_incurred": cli["reserve"][1:].sum(), "bf_paid": bf["reserve"][1:].sum(),
            "cape_cod_paid": cc["reserve"][1:].sum()}
    b = m.odp_bootstrap(paid, n_sim=N_BOOT, seed=t.company)
    q = np.quantile(b["reserves"], [0.05, 0.25, 0.5, 0.75, 0.95, 0.995])
    se, mean = mk["se_total"], mk["reserve_total"]
    ln = {p: m.lognormal_quantile(mean, se, p) for p in (0.05, 0.25, 0.75, 0.95, 0.995)}
    return {"line": t.line, "company": t.company, "name": t.name, "premium_1989_1997": pm, "premium_1997": float(prem[-1]),
            "latest_paid": float(diag_p[1:].sum()), "latest_incurred": float(m.latest_diagonal(inc)[1:].sum()),
            "actual_reserve": float(actual_res),
            "actual_incurred_reserve": float(actual_inc_res), **{f"pred_{k}": float(v) for k, v in pred.items()},
            "mack_se": float(se), "mack_cv": float(se / mean), "boot_sd": b["sd"], "boot_mean": b["mean"], "boot_cv": float(b["sd"] / b["cl_reserve"]),
            "boot_q05": q[0], "boot_q25": q[1], "boot_q50": q[2], "boot_q75": q[3], "boot_q95": q[4], "boot_q995": q[5],
            "mack_q05": ln[0.05], "mack_q25": ln[0.25], "mack_q75": ln[0.75], "mack_q95": ln[0.95], "mack_q995": ln[0.995],
            "elr_apriori": elr, "elr_capecod": cc["elr"], "phi": b["phi"]}


def main() -> dict:
    RES.mkdir(exist_ok=True)
    tris = [t for line in tri.LINES for t in tri.load(line)]
    print(len(tris), "triangles")
    with Pool(6) as pool:
        rows = [r for r in pool.imap_unordered(one, tris, chunksize=4) if r is not None]
    df = pd.DataFrame(rows).sort_values(["line", "company"])
    df.to_csv(RES / "backtest_triangles.csv", index=False, float_format="%.6g")
    print(len(df), "usable triangles")
    return {"triangles": len(df)}


if __name__ == "__main__":
    main()
