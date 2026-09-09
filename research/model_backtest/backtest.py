"""
Walk-forward backtest of several modeling paradigms against college football
outcomes, graded exactly the way tracking.py grades the live site:
  - straight_up_correct: predicted winner == actual winner
  - ats_result: win/loss/push against the betting line, but ONLY on games where
    the model's margin differs from the betting line by >= 3 points (the same
    "recommended" edge threshold app.py/tracking.py use)
  - profit: flat $10 stake at -110 (win pays $9.09, loss costs $10, push is a no-op)

No leakage: every model for week W of season Y is fit using ONLY games from
weeks < W of that season (or, for the postseason bucket, all regular-season
games), and ONLY team stats accumulated through week W-1 (via CFBD's
startWeek/endWeek truncation). Elo ratings are strictly sequential. CORE
ratings are always the PRIOR season's final ratings, never the current season's.
"""
import collections
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge

import models as M

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEASONS = [2022, 2023, 2024, 2025]
MIN_TRAIN_GAMES = 30
FIRST_PREDICT_WEEK = 5  # need >=4 weeks of games before trusting ratings/stats

BET_STAKE = 10.0
BET_TO_WIN = 9.09


def edge_class(model_margin, betting_spread):
    betting_margin = -betting_spread
    diff = abs(model_margin - betting_margin)
    if diff >= 5:
        return "edge-big"
    elif diff >= 3:
        return "edge-medium"
    return None


def grade_pick(model_margin, betting_spread, actual_home_margin, actual_winner_is_home):
    """Returns dict with straight_up_correct, ats_result ('win'/'loss'/'push'/None), profit."""
    predicted_winner_is_home = model_margin > 0
    straight_up_correct = predicted_winner_is_home == actual_winner_is_home

    ats_result = None
    profit = 0.0
    recommended = False
    if betting_spread is not None:
        ec = edge_class(model_margin, betting_spread)
        recommended = ec is not None
        if recommended:
            betting_home_margin = -betting_spread
            recommended_side_home = model_margin > betting_home_margin
            cover_margin = actual_home_margin + betting_spread
            if cover_margin == 0:
                ats_result = "push"
            else:
                home_covered = cover_margin > 0
                ats_result = "win" if (home_covered == recommended_side_home) else "loss"
                profit = BET_TO_WIN if ats_result == "win" else -BET_STAKE

    return {
        "straight_up_correct": straight_up_correct,
        "recommended": recommended,
        "ats_result": ats_result,
        "profit": profit,
    }


def main():
    fbs_teams = M.load_fbs_teams()

    print("Loading games for all seasons...")
    games_by_year = {y: M.load_games_df(y) for y in SEASONS}

    print("Computing sequential Elo across full timeline...")
    elo_snapshots, elo_records = M.compute_elo(games_by_year)

    rows = []
    provider_counter_by_year = {}

    for year in SEASONS:
        df = games_by_year[year]
        regular = df[df["seasonType"] == "regular"].copy()
        completed_regular = regular.dropna(subset=["homePoints", "awayPoints"]).copy()
        last_week = int(completed_regular["week"].max())
        postseason = df[df["seasonType"] == "postseason"].dropna(subset=["homePoints", "awayPoints"]).copy()

        lines_reg_raw = M.load_json(f"lines_{year}.json")
        lines_post_raw = M.load_json(f"lines_post_{year}.json")
        provider_counter = collections.Counter()
        lines_lookup = M.build_line_lookup(lines_reg_raw, provider_counter)
        lines_lookup.update(M.build_line_lookup(lines_post_raw, provider_counter))
        provider_counter_by_year[year] = provider_counter

        core_prior = M.load_core_prior(year - 1)

        week_buckets = list(range(FIRST_PREDICT_WEEK, last_week + 1)) + ["post"]

        for W in week_buckets:
            if W == "post":
                train = completed_regular
                test = postseason
                stats_key = (year, 1, last_week)
                elo_snap = elo_snapshots[(year, "post")]
                elo_coef = M.elo_scale_factor(elo_records, year, "post")
            else:
                train = completed_regular[completed_regular["week"] < W]
                test = completed_regular[completed_regular["week"] == W]
                stats_key = (year, 1, W - 1)
                elo_snap = elo_snapshots[(year, W)]
                elo_coef = M.elo_scale_factor(elo_records, year, W)

            if len(train) < MIN_TRAIN_GAMES or len(test) == 0:
                continue

            stats_clean = M.build_stats_clean(*stats_key)

            # --- fit all models on this week's train slice ---
            massey_ratings, massey_hf = M.fit_massey_ratings(train)
            current_state = M.fit_stage2_model(train, stats_clean, massey_ratings, LinearRegression())
            bayes_feat_state = M.fit_stage2_model(train, stats_clean, massey_ratings, BayesianRidge())
            core_fn = M.core_extra_feature_fn(core_prior)
            core_state = M.fit_stage2_model(train, stats_clean, massey_ratings, LinearRegression(), extra_feature_fn=core_fn) if core_fn else None
            massey_capped_ratings, massey_capped_hf = M.fit_massey_ratings(train, cap=21)
            colley_state = M.fit_colley(train)
            bayes_rating_ratings, bayes_rating_hf = M.fit_bayes_rating(train)
            gbr_state = M.fit_gbr(train, stats_clean, massey_ratings)

            for _, g in test.iterrows():
                home, away = g["homeTeam"], g["awayTeam"]
                if home not in fbs_teams or away not in fbs_teams:
                    continue
                neutral = bool(g.get("neutralSite", False))
                actual_home_margin = g["homePoints"] - g["awayPoints"]
                actual_winner_is_home = actual_home_margin > 0
                spread = lines_lookup.get((home, away))

                preds = {}
                preds["current_baseline"] = M.predict_stage2(current_state, home, away, neutral)
                preds["bayesian_regression"] = M.predict_stage2(bayes_feat_state, home, away, neutral)
                if core_state is not None:
                    preds["current_plus_core"] = M.predict_stage2(core_state, home, away, neutral)
                preds["massey_capped"] = M.predict_rating_only(massey_capped_ratings, massey_capped_hf, home, away, neutral)
                preds["colley"] = M.predict_colley(colley_state, home, away, neutral)
                preds["bayesian_rating_ridge"] = M.predict_rating_only(bayes_rating_ratings, bayes_rating_hf, home, away, neutral)
                preds["elo"] = M.predict_elo(elo_snap, elo_coef, home, away, neutral)
                preds["gradient_boosting"] = M.predict_stage2(gbr_state, home, away, neutral)

                ensemble_parts = [preds.get("current_baseline"), preds.get("elo"), preds.get("bayesian_regression")]
                if all(p is not None for p in ensemble_parts):
                    preds["ensemble_3way"] = float(np.mean(ensemble_parts))

                for model_name, margin in preds.items():
                    if margin is None or (isinstance(margin, float) and np.isnan(margin)):
                        continue
                    grade = grade_pick(margin, spread, actual_home_margin, actual_winner_is_home)
                    rows.append({
                        "season": year, "week": W, "model": model_name,
                        "home": home, "away": away, "neutral": neutral,
                        "model_margin": round(margin, 2),
                        "actual_home_margin": actual_home_margin,
                        "betting_spread": spread,
                        **grade,
                    })

            print(f"  {year} week {W}: train={len(train)} test={len(test)} done")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(RESULTS_DIR, "picks_detail.csv"), index=False)

    # --- provider usage transparency ---
    with open(os.path.join(RESULTS_DIR, "line_provider_usage.txt"), "w") as f:
        for year, counter in provider_counter_by_year.items():
            f.write(f"{year}: {dict(counter)}\n")

    # --- aggregate summary: per season x model, and overall x model ---
    def summarize(group):
        total = len(group)
        su_wins = group["straight_up_correct"].sum()
        rec = group[group["recommended"] & group["ats_result"].isin(["win", "loss"])]
        ats_wins = (rec["ats_result"] == "win").sum()
        ats_losses = (rec["ats_result"] == "loss").sum()
        ats_decided = ats_wins + ats_losses
        profit = group["profit"].sum()
        return pd.Series({
            "total_picks": total,
            "straight_up_win_pct": round(100 * su_wins / total, 1) if total else np.nan,
            "ats_decided": ats_decided,
            "ats_win_pct": round(100 * ats_wins / ats_decided, 1) if ats_decided else np.nan,
            "total_profit": round(profit, 2),
            "roi_pct": round(100 * profit / (ats_decided * BET_STAKE), 1) if ats_decided else np.nan,
        })

    per_season = results.groupby(["model", "season"]).apply(summarize, include_groups=False).reset_index()
    per_season.to_csv(os.path.join(RESULTS_DIR, "summary_per_season.csv"), index=False)

    overall = results.groupby("model").apply(summarize, include_groups=False).reset_index()
    overall = overall.sort_values("total_profit", ascending=False)
    overall.to_csv(os.path.join(RESULTS_DIR, "summary_overall.csv"), index=False)

    print("\n=== OVERALL (all seasons combined) ===")
    print(overall.to_string(index=False))
    print("\nDone. Results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
