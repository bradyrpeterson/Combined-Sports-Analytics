"""
Follow-up analysis #1: break down every model's backtest results by the site's own
edge tiers (football/predictor.py's calculate_edge_highlight):
    diff = |model_margin - (-betting_spread)|
    diff >= 5        -> 'edge-big'    (high / green)
    3 <= diff < 5     -> 'edge-medium' (medium / yellow)
    diff < 3          -> unflagged     (low edge, control group)
    diff >= 3         -> 'combined'    (high+medium, == the site's "recommended")

Reuses research/model_backtest/results/picks_detail.csv from the prior run --
no new CFBD API calls.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

BET_STAKE = 10.0
BET_TO_WIN = 9.09
BREAKEVEN = 110 / 210  # 0.5238...


def load():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "picks_detail.csv"))
    df = df.dropna(subset=["betting_spread"]).copy()
    df["betting_margin"] = -df["betting_spread"]
    df["diff"] = (df["model_margin"] - df["betting_margin"]).abs()
    df["picked_home"] = df["model_margin"] > df["betting_margin"]
    df["cover_margin"] = df["actual_home_margin"] + df["betting_spread"]
    df["home_covered"] = df["cover_margin"] > 0
    df["is_push"] = df["cover_margin"] == 0
    df["ats_win"] = np.where(df["is_push"], np.nan, df["home_covered"] == df["picked_home"])

    def tier(d):
        if d >= 5:
            return "high (green, >=5pt)"
        elif d >= 3:
            return "medium (yellow, 3-5pt)"
        else:
            return "low (unflagged, <3pt)"

    df["tier"] = df["diff"].apply(tier)
    df["combined_tier"] = np.where(df["diff"] >= 3, "combined (high+medium, >=3pt)", "n/a")
    return df


def grade(group):
    n = len(group)
    su = group["straight_up_correct"].mean() * 100
    decided = group.dropna(subset=["ats_win"])
    ats_n = len(decided)
    ats_wins = int(decided["ats_win"].sum())
    ats_pct = 100 * ats_wins / ats_n if ats_n else np.nan
    profit = np.where(decided["ats_win"], BET_TO_WIN, -BET_STAKE).sum() if ats_n else 0.0
    p_val = np.nan
    if ats_n >= 5:
        p_val = binomtest(ats_wins, ats_n, BREAKEVEN).pvalue
    return pd.Series({
        "n_picks": n, "straight_up_%": round(su, 1),
        "ats_n": ats_n, "ats_win_%": round(ats_pct, 1) if ats_n else np.nan,
        "profit_$": round(profit, 2),
        "roi_%": round(100 * profit / (ats_n * BET_STAKE), 1) if ats_n else np.nan,
        "p_vs_breakeven(52.4%)": round(p_val, 3) if ats_n >= 5 else np.nan,
    })


def main():
    df = load()
    models = sorted(df["model"].unique())

    print("=" * 100)
    print("PART 1a: results by edge tier (high / medium / combined / low), per model, ALL SEASONS")
    print("=" * 100)
    all_rows = []
    for m in models:
        sub = df[df["model"] == m]
        for tier_name, tsub in [
            ("high (green, >=5pt)", sub[sub["tier"] == "high (green, >=5pt)"]),
            ("medium (yellow, 3-5pt)", sub[sub["tier"] == "medium (yellow, 3-5pt)"]),
            ("combined (>=3pt, ='recommended')", sub[sub["diff"] >= 3]),
            ("low (unflagged, <3pt)", sub[sub["diff"] < 3]),
        ]:
            if len(tsub) == 0:
                continue
            row = grade(tsub)
            row["model"] = m
            row["tier"] = tier_name
            all_rows.append(row)
    tier_df = pd.DataFrame(all_rows)[["model", "tier", "n_picks", "straight_up_%", "ats_n", "ats_win_%", "profit_$", "roi_%", "p_vs_breakeven(52.4%)"]]
    tier_df.to_csv(os.path.join(RESULTS_DIR, "tier_breakdown_overall.csv"), index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(tier_df.to_string(index=False))

    print()
    print("=" * 100)
    print("PART 1b: high-edge tier sample size PER SEASON (checking for tiny-n seasons)")
    print("=" * 100)
    high_by_season = []
    for m in models:
        sub = df[(df["model"] == m) & (df["diff"] >= 5)]
        for season, ssub in sub.groupby("season"):
            row = grade(ssub)
            row["model"] = m
            row["season"] = season
            high_by_season.append(row)
    high_season_df = pd.DataFrame(high_by_season)[["model", "season", "n_picks", "straight_up_%", "ats_n", "ats_win_%", "profit_$"]]
    high_season_df.to_csv(os.path.join(RESULTS_DIR, "tier_high_by_season.csv"), index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(high_season_df.to_string(index=False))

    print()
    print("Combined-tier sanity check: does it reproduce last run's recommended/ATS numbers?")
    combined = tier_df[tier_df["tier"] == "combined (>=3pt, ='recommended')"].sort_values("roi_%", ascending=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
