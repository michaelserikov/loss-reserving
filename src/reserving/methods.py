"""Reserving methods on a cumulative run-off triangle with NaN below the latest diagonal.

Chain ladder with volume-weighted factors, the distribution-free standard error of Mack (1993), the over-dispersed
Poisson bootstrap of England and Verrall (1999, 2002), Bornhuetter-Ferguson, Cape Cod and the lognormal approximation
used for the Solvency II reserve risk. Everything is plain numpy so that each formula can be tested against a published
example (Taylor and Ashe, 1983).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def latest_diagonal(c: np.ndarray) -> np.ndarray:
    n = c.shape[0]
    return np.array([c[i, n - 1 - i] for i in range(n)])


def development_factors(c: np.ndarray, weights: str = "volume") -> np.ndarray:
    """Age-to-age factors f_k, k = 1..n-1, volume weighted (sum of next column over sum of this column) or simple average."""
    n = c.shape[1]
    f = np.ones(n - 1)
    for k in range(n - 1):
        rows = ~np.isnan(c[:, k + 1])
        if weights == "volume":
            f[k] = c[rows, k + 1].sum() / c[rows, k].sum()
        else:
            f[k] = np.mean(c[rows, k + 1] / c[rows, k])
    return f


def cumulative_factors(f: np.ndarray, tail: float = 1.0) -> np.ndarray:
    """Factor to ultimate from each development year: cdf_k = tail * prod_{j >= k} f_j (length n)."""
    return np.append(np.cumprod(f[::-1])[::-1], 1.0) * tail


def chain_ladder(c: np.ndarray, weights: str = "volume", tail: float = 1.0) -> dict:
    n = c.shape[0]
    f = development_factors(c, weights)
    cdf = cumulative_factors(f, tail)
    diag = latest_diagonal(c)
    age = np.array([n - 1 - i for i in range(n)])          # 0-based development year of the latest diagonal
    ultimate = diag * cdf[age]
    return {"factors": f, "cdf": cdf, "latest": diag, "ultimate": ultimate, "reserve": ultimate - diag}


def mack(c: np.ndarray) -> dict:
    """Mack's distribution-free standard error of the chain ladder reserve, by accident year and in total."""
    n = c.shape[0]
    f = development_factors(c, "volume")
    sigma2 = np.zeros(n - 1)
    for k in range(n - 1):
        rows = ~np.isnan(c[:, k + 1])
        m = rows.sum()
        if m > 1:
            sigma2[k] = (c[rows, k] * (c[rows, k + 1] / c[rows, k] - f[k]) ** 2).sum() / (m - 1)
    if n >= 3 and sigma2[n - 2] == 0:                                    # last factor rests on one observation
        a, b = sigma2[n - 3], sigma2[n - 4] if n >= 4 else sigma2[n - 3]
        sigma2[n - 2] = min(a * a / b, min(a, b)) if b > 0 else a
    cl = chain_ladder(c)
    ult, diag = cl["ultimate"], cl["latest"]
    age = [n - 1 - i for i in range(n)]
    col_sum = [np.nansum(c[: n - 1 - k, k]) for k in range(n - 1)]       # sum of column k over rows that have column k+1
    mse = np.zeros(n)
    for i in range(1, n):
        s = 0.0
        for k in range(age[i], n - 1):
            ck = c[i, age[i]] * np.prod(f[age[i]:k]) if k > age[i] else c[i, age[i]]
            s += sigma2[k] / f[k] ** 2 * (1.0 / ck + 1.0 / col_sum[k])
        mse[i] = ult[i] ** 2 * s
    # covariance between accident years (Mack's formula for the total)
    total = mse.sum()
    for i in range(1, n):
        for j in range(i + 1, n):
            s = 0.0
            for k in range(age[i], n - 1):
                s += sigma2[k] / f[k] ** 2 / col_sum[k]
            total += 2 * ult[i] * ult[j] * s
    reserve = cl["reserve"]
    return {**cl, "sigma2": sigma2, "se_by_year": np.sqrt(mse), "se_total": float(np.sqrt(total)),
            "reserve_total": float(reserve.sum()), "cv_total": float(np.sqrt(total) / reserve.sum()) if reserve.sum() > 0 else np.nan}


def lognormal_quantile(mean: float, sd: float, q: float) -> float:
    """Quantile of a lognormal with the given mean and standard deviation."""
    if mean <= 0 or sd <= 0:
        return mean
    s2 = np.log(1 + (sd / mean) ** 2)
    return float(mean * np.exp(-0.5 * s2 + np.sqrt(s2) * norm.ppf(q)) )


def odp_bootstrap(c: np.ndarray, n_sim: int = 2000, seed: int = 1, process_error: bool = True) -> dict:
    """Over-dispersed Poisson bootstrap with Pearson residuals (England and Verrall).

    Fitted incremental values come from the chain ladder (identical to the Poisson GLM with row and column effects);
    residuals are adjusted for degrees of freedom and resampled; each pseudo triangle is re-projected with the chain
    ladder; process error is added as a gamma draw with variance phi * mean."""
    rng = np.random.default_rng(seed)
    n = c.shape[0]
    cl = chain_ladder(c)
    f = cl["factors"]
    # back-fit cumulative values from the latest diagonal with the volume factors
    fitted_c = np.full_like(c, np.nan)
    for i in range(n):
        last = n - 1 - i
        fitted_c[i, last] = c[i, last]
        for k in range(last - 1, -1, -1):
            fitted_c[i, k] = fitted_c[i, k + 1] / f[k]
    m = fitted_c.copy()
    m[:, 1:] = fitted_c[:, 1:] - fitted_c[:, :-1]
    inc = c.copy()
    inc[:, 1:] = c[:, 1:] - c[:, :-1]
    mask = ~np.isnan(inc)
    ok = mask & (m > 0)
    resid = np.where(ok, (inc - m) / np.sqrt(np.where(ok, m, 1.0)), np.nan)
    n_obs, p = int(mask.sum()), 2 * n - 1
    phi = float(np.nansum(resid ** 2) / (n_obs - p))
    adj = np.sqrt(n_obs / (n_obs - p)) * resid[ok]
    adj = adj - adj.mean()
    reserves = np.empty(n_sim)
    for s in range(n_sim):
        r = np.where(ok, rng.choice(adj, size=inc.shape), 0.0)
        pseudo_inc = np.where(mask, m + r * np.sqrt(np.where(m > 0, m, 0.0)), np.nan)
        pseudo = np.where(mask, np.nancumsum(np.where(mask, pseudo_inc, 0.0), axis=1), np.nan)
        f_s = development_factors(pseudo, "volume")
        proj = chain_ladder_projection(pseudo, f_s, m, phi, rng, process_error)
        reserves[s] = proj
    return {"reserves": reserves, "phi": phi, "mean": float(reserves.mean()), "sd": float(reserves.std(ddof=1)),
            "cl_reserve": float(cl["reserve"].sum())}


def chain_ladder_projection(pseudo: np.ndarray, f_s: np.ndarray, m_fit: np.ndarray, phi: float, rng, process_error: bool) -> float:
    """Reserve of a pseudo triangle: expected future increments from the refitted chain ladder, plus process error.

    The expected increment of every lower cell is projected without noise; process error is an independent gamma draw per
    cell with mean equal to that increment and variance phi times the mean (England and Verrall)."""
    n = pseudo.shape[0]
    total = 0.0
    for i in range(1, n):
        last = n - 1 - i
        cum = pseudo[i, last]
        for k in range(last, n - 1):
            mean_inc = cum * (f_s[k] - 1.0)
            if process_error and mean_inc > 0 and phi > 0:
                total += rng.gamma(mean_inc / phi, phi)
            else:
                total += mean_inc
            cum += mean_inc
    return float(total)


def bornhuetter_ferguson(c: np.ndarray, premium: np.ndarray, elr: np.ndarray | float, f: np.ndarray | None = None) -> dict:
    """Ultimate = latest + (1 - 1/cdf) * premium * expected loss ratio. `elr` is a scalar or one value per accident year."""
    n = c.shape[0]
    f = development_factors(c) if f is None else f
    cdf = cumulative_factors(f)
    diag = latest_diagonal(c)
    age = np.array([n - 1 - i for i in range(n)])
    pct_unpaid = 1 - 1 / cdf[age]
    ult = diag + pct_unpaid * premium * elr
    return {"ultimate": ult, "reserve": ult - diag}


def cape_cod(c: np.ndarray, premium: np.ndarray, f: np.ndarray | None = None) -> dict:
    """Cape Cod: the expected loss ratio is the paid (or incurred) to date over the premium 'used up' by the pattern."""
    n = c.shape[0]
    f = development_factors(c) if f is None else f
    cdf = cumulative_factors(f)
    diag = latest_diagonal(c)
    age = np.array([n - 1 - i for i in range(n)])
    used = premium / cdf[age]
    elr = float(diag.sum() / used.sum())
    out = bornhuetter_ferguson(c, premium, elr, f)
    return {**out, "elr": elr}


def solvency2_reserve_capital(reserve: float, sigma: float, level: float = 0.995) -> float:
    """Capital of the standard formula for reserve risk: reserve * (lognormal VaR factor - 1)."""
    s = np.sqrt(np.log(1 + sigma ** 2))
    return float(reserve * (np.exp(norm.ppf(level) * s) / np.sqrt(1 + sigma ** 2) - 1))
