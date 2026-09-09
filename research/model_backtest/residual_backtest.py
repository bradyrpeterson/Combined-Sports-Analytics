"""
Tests the "rating backbone + bounded ML residual" architecture against plain
rating models, using the same walk-forward discipline and grading as backtest.py.

The question this answers: the existing stage-2 model failed because it was asked
to predict margin using a rating that already predicts margin (rank-deficient
features, ~51 training rows, no regularization). Does ML help if instead it is
asked to predict only what Massey gets WRONG, trained on a pooled multi-season
sample, with its output bounded so it can adjust but never override?

Models compared:
  massey_uncapped   -- plain Massey, no cap (never actually graded standalone before)
  massey_capped     -- cap=21, the previous backtest winner
  massey_huber      -- Huber loss instead of least squares: outlier-robust with no
                       arbitrary cap constant
  residual_ridge    -- massey_uncapped + Ridge on the residual, clipped to +/-RESID_CLIP
  residual_gbr      -- massey_uncapped + GradientBoosting on the residual, same clip

Leakage control: to predict (year Y, week W), residual models train ONLY on
residual observations from seasons < Y plus weeks < W of season Y. Each residual
observation itself uses the Massey fit and stats snapshot that were available
before its own game was played.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor

import models as M
from backtest import grade_pick, BET_STAKE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEASONS = [2022, 2023, 2024, 2025]
FIRST_WEEK = 5          # cached stats only go back to snapshot 1_4
MIN_TRAIN_GAMES = 30
RESID_CLIP = 10.0       # ML may adjust the backbone by at most this many points
MIN_RESID_ROWS = 400    # don't fit a residual model on a thin pool


def fit_massey_huber(train):
    """Massey ratings fit with Huber loss -- down-weights blowout outliers without
    imposing a hard margin cap."""
    teams = sorted(set(train["homeTeam"]).union(train["awayTeam"]))
    X, y = M._design_matrix(train, teams)
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500, fit_intercept=False)
    model.fit(X.values, y)
    coefs = pd.Series(model.coef_, index=X.columns)
    hf = coefs["home_field"]
    r = coefs.drop("home_field")
    r -= r.mean()
    return r, hf


def residual_features(ratings, stats_idx, core_prior, home, away, neutral):
    """Features for the residual model. Deliberately NON-redundant (no value that is
    an exact linear combination of others) and weighted toward information the
    margin-only Massey fit cannot see."""
    if home not in ratings.index or away not in ratings.index:
        return None
    if home not in stats_idx.index or away not in stats_idx.index:
        return None
    h, a = stats_idx.loc[home], stats_idx.loc[away]
    rating_diff = float(ratings[home] - ratings[away])

    core_diff = 0.0
    if core_prior is not None and home in core_prior.index and away in core_prior.index:
        core_diff = float(core_prior.loc[home, "overall"] - core_prior.loc[away, "overall"])

    vec = [
        rating_diff,
        abs(rating_diff),                                      # is this a mismatch game?
        float(h["yardsPerPlay_off"] - a["yardsPerPlay_def"]),   # home offense vs away defense
        float(a["yardsPerPlay_off"] - h["yardsPerPlay_def"]),   # away offense vs home defense
        float(h["thirdDownPct"] - a["thirdDownPct"]),
        float(h["turnoverMargin"] - a["turnoverMargin"]),
        core_diff,                                             # prior-season strength
        0.0 if neutral else 1.0,
    ]
    if any(pd.isna(v) for v in vec):
        return None
    return vec


FEATURE_NAMES = [
    "rating_diff", "abs_rating_diff", "h_off_vs_a_def", "a_off_vs_h_def",
    "third_diff", "to_diff", "core_prior_diff", "home_field",
]


def build_week_table():
    """One pass over every (season, week): fit the rating models on that week's
    train slice and record, for each game, the baseline predictions, the residual
    features, and the actual result. Everything downstream reads this table."""
    games_by_year = {y: M.load_games_df(y) for y in SEASONS}
    rows = []

    for year in SEASONS:
        df = games_by_year[year]
        regular = df[df["seasonType"] == "regular"].copy()
        completed = regular.dropna(subset=["homePoints", "awayPoints"]).copy()
        last_week = int(completed["week"].max())

        lines = M.build_line_lookup(M.load_json(f"lines_{year}.json"))
        core_prior = M.load_core_prior(year - 1)
        fbs = M.load_fbs_teams()

        for W in range(FIRST_WEEK, last_week + 1):
            train = completed[completed["week"] < W]
            test = completed[completed["week"] == W]
            if len(train) < MIN_TRAIN_GAMES or len(test) == 0:
                continue

            stats_idx = M.build_stats_clean(year, 1, W - 1).set_index("team")

            r_unc, hf_unc = M.fit_massey_ratings(train)
            r_cap, hf_cap = M.fit_massey_ratings(train, cap=21)
            r_hub, hf_hub = fit_massey_huber(train)

            for _, g in test.iterrows():
                home, away = g["homeTeam"], g["awayTeam"]
                if home not in fbs or away not in fbs:
                    continue
                neutral = bool(g.get("neutralSite", False))
                actual = float(g["homePoints"] - g["awayPoints"])

                base_unc = M.predict_rating_only(r_unc, hf_unc, home, away, neutral)
                if base_unc is None:
                    continue
                base_cap = M.predict_rating_only(r_cap, hf_cap, home, away, neutral)
                base_hub = M.predict_rating_only(r_hub, hf_hub, home, away, neutral)

                feats = residual_features(r_unc, stats_idx, core_prior, home, away, neutral)
                if feats is None:
                    continue

                rows.append({
                    "season": year, "week": W, "home": home, "away": away,
                    "neutral": neutral, "actual": actual,
                    "spread": lines.get((home, away)),
                    "massey_uncapped": base_unc,
                    "massey_capped": base_cap,
                    "massey_huber": base_hub,
                    "residual": actual - base_unc,
                    **{f"f_{n}": v for n, v in zip(FEATURE_NAMES, feats)},
                })
            print(f"  built {year} week {W}: train={len(train)} test={len(test)}")

    return pd.DataFrame(rows)


def run():
    print("Building walk-forward week table...")
    tbl = build_week_table()
    print(f"\n{len(tbl)} graded games across {tbl['season'].nunique()} seasons\n")

    fcols = [f"f_{n}" for n in FEATURE_NAMES]
    tbl = tbl.sort_values(["season", "week"]).reset_index(drop=True)

    # Walk-forward residual predictions: for each (season, week), fit on strictly
    # earlier observations only.
    for name in ("residual_ridge", "residual_gbr"):
        tbl[name] = np.nan

    keys = tbl[["season", "week"]].drop_duplicates().values.tolist()
    for season, week in keys:
        past = tbl[(tbl["season"] < season) | ((tbl["season"] == season) & (tbl["week"] < week))]
        cur = tbl[(tbl["season"] == season) & (tbl["week"] == week)]
        if len(past) < MIN_RESID_ROWS:
            continue
        Xp, yp = past[fcols].values, past["residual"].values
        Xc = cur[fcols].values

        ridge = Ridge(alpha=10.0).fit(Xp, yp)
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ).fit(Xp, yp)

        adj_r = np.clip(ridge.predict(Xc), -RESID_CLIP, RESID_CLIP)
        adj_g = np.clip(gbr.predict(Xc), -RESID_CLIP, RESID_CLIP)
        tbl.loc[cur.index, "residual_ridge"] = cur["massey_uncapped"].values + adj_r
        tbl.loc[cur.index, "residual_gbr"] = cur["massey_uncapped"].values + adj_g

    model_cols = ["massey_uncapped", "massey_capped", "massey_huber",
                  "residual_ridge", "residual_gbr"]

    # Grade every model on the identical subset of games (those where every model
    # produced a prediction), so the comparison is apples to apples.
    graded = tbl.dropna(subset=model_cols).copy()
    print(f"Comparable subset (all models predicted): {len(graded)} games\n")

    summary = []
    for col in model_cols:
        recs = []
        for _, r in graded.iterrows():
            g = grade_pick(r[col], r["spread"], r["actual"], r["actual"] > 0)
            recs.append(g)
        gd = pd.DataFrame(recs)
        dec = gd[gd["ats_result"].isin(["win", "loss"])]
        n_dec = len(dec)
        ats_w = (dec["ats_result"] == "win").sum()
        profit = gd["profit"].sum()
        mae = float(np.mean(np.abs(graded[col] - graded["actual"])))
        summary.append({
            "model": col,
            "MAE": round(mae, 3),
            "straight_up_%": round(100 * gd["straight_up_correct"].mean(), 1),
            "ats_n": n_dec,
            "ats_win_%": round(100 * ats_w / n_dec, 1) if n_dec else np.nan,
            "profit_$": round(profit, 2),
            "roi_%": round(100 * profit / (n_dec * BET_STAKE), 1) if n_dec else np.nan,
        })

    out = pd.DataFrame(summary).sort_values("MAE")
    out.to_csv(os.path.join(RESULTS_DIR, "residual_comparison.csv"), index=False)
    print("=== RATING BACKBONE vs BOUNDED ML RESIDUAL ===")
    print(out.to_string(index=False))

    # Per-season MAE, to check the result is not driven by one season.
    per = graded.groupby("season").apply(
        lambda gp: pd.Series({c: round(float(np.mean(np.abs(gp[c] - gp["actual"]))), 3) for c in model_cols}),
        include_groups=False,
    )
    per.to_csv(os.path.join(RESULTS_DIR, "residual_mae_by_season.csv"))
    print("\n=== MAE by season ===")
    print(per.to_string())

    # How much is the ML layer actually moving the backbone?
    adj = graded["residual_ridge"] - graded["massey_uncapped"]
    adj_g = graded["residual_gbr"] - graded["massey_uncapped"]
    print(f"\nridge adjustment: mean |adj| = {adj.abs().mean():.2f} pts, max = {adj.abs().max():.2f}")
    print(f"gbr   adjustment: mean |adj| = {adj_g.abs().mean():.2f} pts, max = {adj_g.abs().max():.2f}")

    graded.to_csv(os.path.join(RESULTS_DIR, "residual_picks_detail.csv"), index=False)
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
