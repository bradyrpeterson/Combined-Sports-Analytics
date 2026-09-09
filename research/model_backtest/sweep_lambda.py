"""
Tune the single parameter of the ridge-to-prior backbone.

lambda controls how hard this season's results have to argue to move a team off
its preseason prior. It replaces the hand-picked harmonic decay constant
(PRIOR_DECAY_K=6) in production. Large lambda = trust the prior, good in week 1
but too sticky by week 10; small lambda = trust the data, which is what produces
degenerate week-2 ratings. We want one value that behaves in BOTH regimes.

Reported per lambda and per window: MAE, straight-up %, and the tail metric that
actually gates shipping -- sign flips against a double-digit market favourite.
"""
import os

import numpy as np
import pandas as pd

import models as M
from early_backtest import build_prior, FIXED_HFA, SEASONS, EARLY_WEEKS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

LAMBDAS = [1, 2, 4, 8, 15, 30, 60, 120]


def shrunk_ratings_multi(train, prior, lambdas):
    """Build the Massey design matrix once, then solve the ridge-to-prior fit for
    every lambda. Returns {lambda: ratings Series}."""
    if len(train) == 0:
        return {lam: prior.copy() for lam in lambdas}

    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(train), len(teams)
    X = np.zeros((n, k))
    y = np.zeros(n)
    for i, (_, row) in enumerate(train.iterrows()):
        X[i, idx[row["homeTeam"]]] = 1.0
        X[i, idx[row["awayTeam"]]] = -1.0
        hfa = 0.0 if bool(row.get("neutralSite", False)) else FIXED_HFA
        y[i] = (row["homePoints"] - row["awayPoints"]) - hfa

    p = prior.reindex(teams).fillna(0.0).values
    resid = y - X @ p
    XtX = X.T @ X
    Xtr = X.T @ resid

    out = {}
    base = prior.reindex(prior.index.union(teams)).fillna(0.0)
    for lam in lambdas:
        d = np.linalg.solve(XtX + lam * np.eye(k), Xtr)
        s = base.copy()
        s.loc[teams] = p + d
        out[lam] = s
    return out


def run():
    fbs = M.load_fbs_teams()
    rows = []

    for year in SEASONS:
        df = M.load_games_df(year)
        completed = df[df["seasonType"] == "regular"].dropna(subset=["homePoints", "awayPoints"]).copy()
        last_week = int(completed["week"].max())
        lines = M.build_line_lookup(M.load_json(f"lines_{year}.json"))
        prior, _, _ = build_prior(year, fbs)

        for W in range(1, last_week + 1):
            train = completed[completed["week"] < W]
            test = completed[completed["week"] == W]
            if len(test) == 0:
                continue
            rat = shrunk_ratings_multi(train, prior, LAMBDAS)

            for _, g in test.iterrows():
                home, away = g["homeTeam"], g["awayTeam"]
                if home not in fbs or away not in fbs:
                    continue
                neutral = bool(g.get("neutralSite", False))
                actual = float(g["homePoints"] - g["awayPoints"])
                spread = lines.get((home, away))
                rec = {"season": year, "week": W, "actual": actual, "spread": spread}
                ok = True
                for lam in LAMBDAS:
                    r = rat[lam]
                    if home not in r.index or away not in r.index:
                        ok = False
                        break
                    rec[f"lam_{lam}"] = float(r[home] - r[away] + (0.0 if neutral else FIXED_HFA))
                if ok:
                    rows.append(rec)
        print(f"  swept {year}")

    tbl = pd.DataFrame(rows)

    out = []
    for label, sub in (("weeks 1-4", tbl[tbl["week"].isin(EARLY_WEEKS)]),
                       ("weeks 5+", tbl[~tbl["week"].isin(EARLY_WEEKS)])):
        for lam in LAMBDAS:
            col = f"lam_{lam}"
            mae = float((sub[col] - sub["actual"]).abs().mean())
            su = float((np.sign(sub[col]) == np.sign(sub["actual"])).mean() * 100)
            wl = sub.dropna(subset=["spread"]).copy()
            wl["vegas"] = -wl["spread"]
            big = wl[wl["vegas"].abs() >= 10]
            flips = int((np.sign(big[col]) != np.sign(big["vegas"])).sum())
            out.append({
                "window": label, "lambda": lam, "n": len(sub),
                "MAE": round(mae, 2), "SU_%": round(su, 1),
                "flips": flips,
                "flip_rate_%": round(100 * flips / len(big), 2) if len(big) else np.nan,
                "vs_line_max": round(float((wl[col] - wl["vegas"]).abs().max()), 1),
            })

    res = pd.DataFrame(out)
    res.to_csv(os.path.join(RESULTS_DIR, "lambda_sweep.csv"), index=False)
    for label in ("weeks 1-4", "weeks 5+"):
        print(f"\n=== {label.upper()} ===")
        print(res[res["window"] == label].drop(columns=["window"]).to_string(index=False))

    # Per-week flips for the most promising values, to confirm week 2 is handled.
    print("\n=== flips vs a 10pt market favourite, by week ===")
    hdr = "  week  " + "".join(f"lam={l:<6}" for l in LAMBDAS)
    print(hdr)
    for w in range(1, 9):
        sub = tbl[tbl["week"] == w].dropna(subset=["spread"]).copy()
        if len(sub) == 0:
            continue
        sub["vegas"] = -sub["spread"]
        big = sub[sub["vegas"].abs() >= 10]
        line = f"  {w:<6}"
        for lam in LAMBDAS:
            f = int((np.sign(big[f"lam_{lam}"]) != np.sign(big["vegas"])).sum()) if len(big) else 0
            line += f"{f:<10}"
        print(line)


if __name__ == "__main__":
    run()
