"""
Can anything be published before week 5?

The original backtest started at week 5, so weeks 1-4 -- the regime that produced
the LSU/Louisiana Tech blowup -- has never been measured. This script measures it.

The headline metric here is NOT average accuracy. It is tail behaviour: how often
does a model say something indefensible? The specific failure we care about is a
"sign flip against a big favourite" -- the model picking the other side of a game
where the market has a double-digit favourite (LSU -35.5 predicted as a La Tech
win). A model can have fine MAE and still be unshippable if it does that.

Leakage control:
  - SP+ prior for season Y is season Y-1's FINAL SP+ (never same-season, which
    would embed the whole season being predicted). This makes the prior here
    somewhat WEAKER than production's, which fetches a genuine preseason
    projection -- so these numbers are a conservative lower bound.
  - "own last year" prior for season Y is a Massey fit on season Y-1 games only.
  - In-season fits for week W use only weeks < W.
  - The residual ML model for season Y week W trains only on seasons < Y plus
    weeks < W of season Y.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

import models as M
from backtest import grade_pick, BET_STAKE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEASONS = [2022, 2023, 2024, 2025]
EARLY_WEEKS = [1, 2, 3, 4]
SP_WEIGHT = 0.5
FIXED_HFA = 2.5          # home field is unidentifiable on small samples; pin it
#Tuned via sweep_lambda.py. The optimum is broad and flat (1-4 all behave), and it
#agrees with the value theory suggests: lambda ~= (game-margin noise variance) /
#(prior error variance) ~= 13^2 / 7^2 ~= 3.5. Data and theory landing in the same
#place is the reason to trust this rather than a hand-picked decay constant.
RIDGE_TO_PRIOR_LAMBDA = 3.0
RESID_CLIP = 10.0
MIN_RESID_ROWS = 400

FEATURE_NAMES = [
    "rating_diff", "abs_rating_diff", "h_off_vs_a_def", "a_off_vs_h_def",
    "third_diff", "to_diff", "home_field",
]


def load_sp(year):
    """Season `year`'s final SP+ ratings, mean-centred."""
    raw = M.load_json(f"sp_{year}.json")
    s = pd.Series({r["team"]: r["rating"] for r in raw if r.get("rating") is not None})
    return s - s.mean()


def build_prior(year, fbs):
    """Preseason prior for `year`: 50/50 blend of prior-season SP+ and a Massey fit
    on prior-season games. Mirrors the production blend, but leak-free."""
    sp = load_sp(year - 1)

    prev = M.load_games_df(year - 1)
    prev = prev.dropna(subset=["homePoints", "awayPoints"]).copy()
    prev["margin"] = prev["homePoints"] - prev["awayPoints"]
    own, _ = M.fit_massey_ratings(prev)

    teams = set(fbs) | set(sp.index) | set(own.index)
    sp_f = sp.reindex(teams).fillna(0.0)
    own_f = own.reindex(teams).fillna(0.0)
    return SP_WEIGHT * sp_f + (1 - SP_WEIGHT) * own_f, sp_f, own_f


def fit_ridge_to_prior(train, prior, lam=RIDGE_TO_PRIOR_LAMBDA):
    """Massey ratings shrunk toward the preseason prior rather than toward zero.

    Minimises ||y - Xb||^2 + lam*||b - prior||^2. With no games this returns the
    prior exactly; with many games it converges to the ordinary Massey fit. This
    replaces the hand-tuned harmonic decay schedule -- the blend falls out of how
    much evidence there is, not a constant someone picked.
    """
    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"])) if len(train) else []
    if len(train) == 0 or len(teams) == 0:
        return prior.copy()

    X = pd.DataFrame(0, index=np.arange(len(train)), columns=teams)
    y = np.empty(len(train))
    t = train.reset_index(drop=True)
    for i, row in t.iterrows():
        X.loc[i, row["homeTeam"]] = 1
        X.loc[i, row["awayTeam"]] = -1
        hfa = 0.0 if bool(row.get("neutralSite", False)) else FIXED_HFA
        y[i] = (row["homePoints"] - row["awayPoints"]) - hfa

    p = prior.reindex(teams).fillna(0.0).values
    offset = X.values @ p
    d = Ridge(alpha=lam, fit_intercept=False).fit(X.values, y - offset).coef_

    # Schedules include non-FBS opponents the prior has no entry for; give them a
    # 0.0 (average) starting rating so the fit can still solve for them.
    out = prior.reindex(prior.index.union(teams)).fillna(0.0)
    out.loc[teams] = p + d
    return out


def rating_pred(ratings, home, away, neutral):
    if home not in ratings.index or away not in ratings.index:
        return None
    return float(ratings[home] - ratings[away] + (0.0 if neutral else FIXED_HFA))


def feats(ratings, stats_idx, home, away, neutral):
    if home not in ratings.index or away not in ratings.index:
        return None
    rd = float(ratings[home] - ratings[away])
    if stats_idx is None or home not in stats_idx.index or away not in stats_idx.index:
        # No box-score stats yet (week 1): neutral values, so the residual model
        # can still contribute the parts that depend only on ratings.
        v = [rd, abs(rd), 0.0, 0.0, 0.0, 0.0, 0.0 if neutral else 1.0]
        return v
    h, a = stats_idx.loc[home], stats_idx.loc[away]
    v = [
        rd, abs(rd),
        float(h["yardsPerPlay_off"] - a["yardsPerPlay_def"]),
        float(a["yardsPerPlay_off"] - h["yardsPerPlay_def"]),
        float(h["thirdDownPct"] - a["thirdDownPct"]),
        float(h["turnoverMargin"] - a["turnoverMargin"]),
        0.0 if neutral else 1.0,
    ]
    return None if any(pd.isna(x) for x in v) else v


def stats_or_none(year, upto_week):
    if upto_week < 1:
        return None
    try:
        return M.build_stats_clean(year, 1, upto_week).set_index("team")
    except Exception:
        return None


def build_table():
    """Walk-forward over EVERY week of every season, recording baseline predictions
    and residual-model features. Later weeks are included so the residual model has
    a real training pool; only weeks 1-4 are graded at the end."""
    fbs = M.load_fbs_teams()
    rows = []

    for year in SEASONS:
        df = M.load_games_df(year)
        completed = df[df["seasonType"] == "regular"].dropna(subset=["homePoints", "awayPoints"]).copy()
        last_week = int(completed["week"].max())
        lines = M.build_line_lookup(M.load_json(f"lines_{year}.json"))
        prior, sp_f, own_f = build_prior(year, fbs)

        for W in range(1, last_week + 1):
            train = completed[completed["week"] < W]
            test = completed[completed["week"] == W]
            if len(test) == 0:
                continue

            stats_idx = stats_or_none(year, W - 1)

            # --- backbones ---
            r_shrunk = fit_ridge_to_prior(train, prior)

            # production's harmonic-decay blend (what is live today)
            weeks_done = int(train["week"].max()) if len(train) else 0
            pw = 6 / (6 + weeks_done)
            if len(train):
                cur, cur_hf = M.fit_massey_ratings(train)
            else:
                cur, cur_hf = pd.Series(dtype=float), 0.0
            allt = set(prior.index) | set(cur.index)
            prod_r = pw * prior.reindex(allt).fillna(0.0) + (1 - pw) * cur.reindex(allt).fillna(0.0)
            prod_hf = pw * 2.2 + (1 - pw) * cur_hf

            # production's stage-2 ML, fit exactly as production fits it
            prod_ml = None
            if stats_idx is not None and len(train):
                tr2 = train.copy()
                sc = stats_idx.reset_index()
                prod_ml = M.fit_stage2_model(tr2, sc, prod_r, LinearRegression())

            for _, g in test.iterrows():
                home, away = g["homeTeam"], g["awayTeam"]
                if home not in fbs or away not in fbs:
                    continue
                neutral = bool(g.get("neutralSite", False))
                actual = float(g["homePoints"] - g["awayPoints"])

                p_prior = rating_pred(prior, home, away, neutral)
                p_sp = rating_pred(sp_f, home, away, neutral)
                p_own = rating_pred(own_f, home, away, neutral)
                p_shrunk = rating_pred(r_shrunk, home, away, neutral)
                if p_prior is None or p_shrunk is None:
                    continue

                p_prod_nm = None
                if home in prod_r.index and away in prod_r.index:
                    p_prod_nm = float(prod_r[home] - prod_r[away] + (0 if neutral else prod_hf))
                p_prod = M.predict_stage2(prod_ml, home, away, neutral) if prod_ml else None
                if p_prod is None:
                    p_prod = p_prod_nm

                f = feats(r_shrunk, stats_idx, home, away, neutral)
                if f is None:
                    continue

                rows.append({
                    "season": year, "week": W, "home": home, "away": away,
                    "actual": actual, "spread": lines.get((home, away)),
                    "prior_only": p_prior,
                    "sp_only": p_sp,
                    "lastyear_only": p_own,
                    "shrunk": p_shrunk,
                    "production_now": p_prod,
                    "production_no_ml": p_prod_nm,
                    "residual_target": actual - p_shrunk,
                    **{f"f_{n}": v for n, v in zip(FEATURE_NAMES, f)},
                })
        print(f"  built {year} (weeks 1-{last_week})")

    return pd.DataFrame(rows)


def add_residual_model(tbl):
    fcols = [f"f_{n}" for n in FEATURE_NAMES]
    tbl = tbl.sort_values(["season", "week"]).reset_index(drop=True)
    tbl["shrunk_plus_ml"] = np.nan
    for season, week in tbl[["season", "week"]].drop_duplicates().values.tolist():
        past = tbl[(tbl["season"] < season) | ((tbl["season"] == season) & (tbl["week"] < week))]
        cur = tbl[(tbl["season"] == season) & (tbl["week"] == week)]
        if len(past) < MIN_RESID_ROWS:
            continue
        mdl = Ridge(alpha=10.0).fit(past[fcols].values, past["residual_target"].values)
        adj = np.clip(mdl.predict(cur[fcols].values), -RESID_CLIP, RESID_CLIP)
        tbl.loc[cur.index, "shrunk_plus_ml"] = cur["shrunk"].values + adj
    return tbl


MODELS = ["production_now", "production_no_ml", "prior_only", "sp_only",
          "lastyear_only", "shrunk", "shrunk_plus_ml"]


def evaluate(g, label):
    out = []
    for col in MODELS:
        sub = g.dropna(subset=[col, "actual"])
        if len(sub) == 0:
            continue
        err = (sub[col] - sub["actual"]).abs()
        recs = [grade_pick(r[col], r["spread"], r["actual"], r["actual"] > 0) for _, r in sub.iterrows()]
        gd = pd.DataFrame(recs)
        dec = gd[gd["ats_result"].isin(["win", "loss"])]
        n_dec = len(dec)

        # --- tail / embarrassment metrics, measured only where a line exists ---
        wl = sub.dropna(subset=["spread"]).copy()
        wl["vegas"] = -wl["spread"]
        disagree = (wl[col] - wl["vegas"]).abs()
        # sign flip against a double-digit market favourite: the LSU signature
        big = wl[wl["vegas"].abs() >= 10]
        flips = int((np.sign(big[col]) != np.sign(big["vegas"])).sum())

        out.append({
            "window": label, "model": col, "n": len(sub),
            "MAE": round(float(err.mean()), 2),
            "SU_%": round(100 * gd["straight_up_correct"].mean(), 1),
            "ats_n": n_dec,
            "ats_%": round(100 * (dec["ats_result"] == "win").sum() / n_dec, 1) if n_dec else np.nan,
            "roi_%": round(100 * gd["profit"].sum() / (n_dec * BET_STAKE), 1) if n_dec else np.nan,
            "vs_line_mean": round(float(disagree.mean()), 1),
            "vs_line_p95": round(float(disagree.quantile(0.95)), 1),
            "vs_line_max": round(float(disagree.max()), 1),
            "flips_vs_10pt_fav": flips,
            "flip_rate_%": round(100 * flips / len(big), 1) if len(big) else np.nan,
        })
    return pd.DataFrame(out)


def run():
    print("Building walk-forward table (all weeks, all seasons)...")
    tbl = build_table()
    tbl = add_residual_model(tbl)
    print(f"\n{len(tbl)} games total\n")

    early = tbl[tbl["week"].isin(EARLY_WEEKS)]
    late = tbl[~tbl["week"].isin(EARLY_WEEKS)]

    res = pd.concat([evaluate(early, "weeks 1-4"), evaluate(late, "weeks 5+")], ignore_index=True)
    res.to_csv(os.path.join(RESULTS_DIR, "early_season_comparison.csv"), index=False)

    for lab in ("weeks 1-4", "weeks 5+"):
        print(f"=== {lab.upper()} ===")
        print(res[res["window"] == lab].drop(columns=["window"]).to_string(index=False))
        print()

    print("=== WEEKS 1-4, BY WEEK (MAE / flips vs a 10pt market favourite) ===")
    for w in EARLY_WEEKS:
        sub = tbl[tbl["week"] == w]
        parts = []
        for col in MODELS:
            s = sub.dropna(subset=[col])
            if len(s) == 0:
                parts.append(f"{col}=n/a")
                continue
            mae = (s[col] - s["actual"]).abs().mean()
            wl = s.dropna(subset=["spread"]).copy()
            wl["vegas"] = -wl["spread"]
            big = wl[wl["vegas"].abs() >= 10]
            fl = int((np.sign(big[col]) != np.sign(big["vegas"])).sum()) if len(big) else 0
            parts.append(f"{col}={mae:.1f}/{fl}")
        print(f"  week {w} (n={len(sub)}): " + "  ".join(parts))

    tbl.to_csv(os.path.join(RESULTS_DIR, "early_season_detail.csv"), index=False)
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
