"""CAS Loss Reserving Database (Schedule P, ten accident years 1988 to 1997, ten development years): triangles.

The database holds the full 10 x 10 square for every company and line, so a method can be run on the upper triangle
(what was known at the end of 1997) and judged against the lower triangle (what happened later). Amounts are in
thousand dollars, net of reinsurance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
LINES = {"ppauto": "Private passenger auto", "comauto": "Commercial auto", "wkcomp": "Workers' compensation",
         "othliab": "Other liability (occurrence)"}
N = 10


@dataclass
class Triangle:
    """Cumulative paid and incurred square and earned premium of one company and line."""

    company: int
    name: str
    line: str
    paid: np.ndarray        # (10, 10), rows accident years 1988..1997, columns development years 1..10
    incurred: np.ndarray
    premium: np.ndarray     # net earned premium by accident year

    def upper(self, square: np.ndarray) -> np.ndarray:
        """The triangle known at the end of 1997: accident year i has i + 1 development years (0-based: columns <= N - 1 - i)."""
        out = square.astype(float).copy()
        for i in range(N):
            out[i, N - i:] = np.nan
        return out


def _columns(df: pd.DataFrame) -> dict:
    suffix = [c for c in df.columns if c.startswith("CumPaidLoss_")][0].split("_", 1)[1]
    return {"incurred": f"IncurLoss_{suffix}", "paid": f"CumPaidLoss_{suffix}", "premium": f"EarnedPremNet_{suffix}"}


def load(line: str, raw: Path = RAW) -> list[Triangle]:
    df = pd.read_csv(raw / f"{line}_pos.csv")
    cols = _columns(df)
    out = []
    for code, g in df.groupby("GRCODE"):
        if len(g) != N * N:
            continue
        g = g.sort_values(["AccidentYear", "DevelopmentLag"])
        paid = g[cols["paid"]].to_numpy(float).reshape(N, N)
        inc = g[cols["incurred"]].to_numpy(float).reshape(N, N)
        prem = g.groupby("AccidentYear")[cols["premium"]].first().to_numpy(float)
        out.append(Triangle(int(code), str(g["GRNAME"].iloc[0]), line, paid, inc, prem))
    return out


def usable(t: Triangle, min_premium: float = 500.0) -> bool:
    """Positive premium and a paid triangle that is positive, increasing enough for factors to exist."""
    up = t.upper(t.paid)
    first = up[:, 0]
    return bool(np.all(t.premium >= min_premium) and np.all(first > 0) and np.all(t.paid[:, -1] > 0))


def incremental(c: np.ndarray) -> np.ndarray:
    """Incremental amounts of a cumulative triangle (NaN preserved)."""
    inc = c.copy()
    inc[:, 1:] = c[:, 1:] - c[:, :-1]
    return inc
