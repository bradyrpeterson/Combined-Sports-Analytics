"""
Follow-up analysis #2: compare three pick-selection rules on the same underlying
model predictions (reusing picks_detail.csv, no new API calls):

  1. POINT   -- the site's existing rule: |model_margin - betting_margin| >= 3
  2. PERCENT -- |model_margin - betting_margin| / |betting_margin| >= 0.10,
               only evaluated when |betting_margin| >= 3 (percentages of a
               near-pick'em line are degenerate/explode near zero, so those
               games are excluded from this rule rather than auto-flagged --
               noted explicitly, this is a judgment call).
  3. EV      -- convert the model's predicted margin into a probability that the
               model's picked side covers the spread, via a Normal approximation:
               actual_home_margin ~ Normal(model_margin, sigma), so
               P(home covers) = Phi((model_margin + betting_spread) / sigma).
               sigma is the model's own forecast residual std, estimated
               LEAVE-ONE-SEASON-OUT (i.e. for grading season Y, sigma is computed
               from the other 3 seasons' residuals only -- avoids using that
               season's own outcome spread to judge its own bets).
               EV per $1 staked at -110 = p*(100/110) - (1-p)*1. Flag if EV > 0.

Run for current_baseline and massey_capped (best performer in the "combined"
edge tier from tier_analysis.py).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

BET_STAKE = 10.0
BET_TO_WIN = 9.09
BREAKEVEN = 110 / 210

MODELS_OF_INTEREST = ["current_baseline", "massey_capped"]


def load_full():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "picks_detail.csv"))
    return df


def leave_one_season_out_sigma(df_model):
    """dict season -> sigma computed from the OTHER 3 seasons' residuals (all picks,
    not just ones with a spread -- residual = forecast error, doesn't need a line)."""
    df_model = df_model.copy()
    df_model["resid"] = df_model["actual_home_margin"] - df_model["model_margin"]
    seasons = sorted(df_model["season"].unique())
    sigmas = {}
    for s in seasons:
        other = df_model[df_model["season"] != s]
        sigmas[s] = other["resid"].std(ddof=1)
    return sigmas


def prep(df, model_name):
    sub = df[df["model"] == model_name].dropna(subset=["betting_spread"]).copy()
    sub["betting_margin"] = -sub["betting_spread"]
    sub["diff"] = (sub["model_margin"] - sub["betting_margin"]).abs()
    sub["picked_home"] = sub["model_margin"] > sub["betting_margin"]
    sub["cover_margin"] = sub["actual_home_margin"] + sub["betting_spread"]
    sub["home_covered"] = sub["cover_margin"] > 0
    sub["is_push"] = sub["cover_margin"] == 0
    sub["ats_win"] = np.where(sub["is_push"], np.nan, sub["home_covered"] == sub["picked_home"])

    # POINT rule
    sub["flag_point"] = sub["diff"] >= 3

    # PERCENT rule
    sub["pct_diff"] = sub["diff"] / sub["betting_margin"].abs()
    sub["flag_percent"] = (sub["betting_margin"].abs() >= 3) & (sub["pct_diff"] >= 0.10)

    # EV rule (leave-one-season-out sigma computed from ALL predictions of this model,
    # not just the spread-covered subset, for a less noisy sigma estimate)
    sigmas = leave_one_season_out_sigma(df[df["model"] == model_name])
    sub["sigma"] = sub["season"].map(sigmas)
    sub["p_home_covers"] = norm.cdf((sub["model_margin"] + sub["betting_spread"]) / sub["sigma"])
    sub["p_pick_covers"] = np.where(sub["picked_home"], sub["p_home_covers"], 1 - sub["p_home_covers"])
    sub["ev_per_dollar"] = sub["p_pick_covers"] * (100 / 110) - (1 - sub["p_pick_covers"])
    sub["flag_ev"] = sub["ev_per_dollar"] > 0

    return sub


def grade(group):
    n = len(group)
    if n == 0:
        return None
    su = group["straight_up_correct"].mean() * 100
    decided = group.dropna(subset=["ats_win"])
    ats_n = len(decided)
    ats_wins = int(decided["ats_win"].sum())
    ats_pct = 100 * ats_wins / ats_n if ats_n else np.nan
    profit = np.where(decided["ats_win"], BET_TO_WIN, -BET_STAKE).sum() if ats_n else 0.0
    p_val = np.nan
    if ats_n >= 5:
        p_val = binomtest(ats_wins, ats_n, BREAKEVEN).pvalue
    return {
        "n_flagged": n, "straight_up_%": round(su, 1),
        "ats_n": ats_n, "ats_win_%": round(ats_pct, 1) if ats_n else np.nan,
        "profit_$": round(profit, 2),
        "roi_%": round(100 * profit / (ats_n * BET_STAKE), 1) if ats_n else np.nan,
        "p_vs_breakeven": round(p_val, 3) if ats_n >= 5 else np.nan,
    }


def main():
    df = load_full()
    all_results = []
    overlap_notes = []

    for model_name in MODELS_OF_INTEREST:
        sub = prep(df, model_name)
        total_eligible = len(sub)  # games with a betting line available

        for rule_name, flag_col in [("POINT (>=3pt, current site rule)", "flag_point"),
                                     ("PERCENT (>=10%, |line|>=3)", "flag_percent"),
                                     ("EV (Normal-approx cover prob, EV>0)", "flag_ev")]:
            flagged = sub[sub[flag_col]]
            res = grade(flagged)
            if res is None:
                continue
            res["model"] = model_name
            res["rule"] = rule_name
            res["pct_of_eligible_games_flagged"] = round(100 * len(flagged) / total_eligible, 1)
            all_results.append(res)

        # overlap between rules
        n_point = sub["flag_point"].sum()
        n_percent = sub["flag_percent"].sum()
        n_ev = sub["flag_ev"].sum()
        overlap_pp = (sub["flag_point"] & sub["flag_percent"]).sum()
        overlap_pe = (sub["flag_point"] & sub["flag_ev"]).sum()
        overlap_all = (sub["flag_point"] & sub["flag_percent"] & sub["flag_ev"]).sum()
        overlap_notes.append(
            f"{model_name}: total_eligible={total_eligible} | POINT={n_point} PERCENT={n_percent} EV={n_ev} | "
            f"POINT&PERCENT={overlap_pp} POINT&EV={overlap_pe} all_three={overlap_all}"
        )

    out = pd.DataFrame(all_results)[[
        "model", "rule", "n_flagged", "pct_of_eligible_games_flagged", "straight_up_%",
        "ats_n", "ats_win_%", "profit_$", "roi_%", "p_vs_breakeven",
    ]]
    out.to_csv(os.path.join(RESULTS_DIR, "edge_rule_comparison.csv"), index=False)

    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(out.to_string(index=False))
    print()
    print("Overlap between rules (selectivity check):")
    for line in overlap_notes:
        print(" ", line)

    # also report sigma per model per season for transparency
    print()
    print("Leave-one-season-out sigma (residual std, points) used for EV rule:")
    for model_name in MODELS_OF_INTEREST:
        sigmas = leave_one_season_out_sigma(df[df["model"] == model_name])
        print(f"  {model_name}: {sigmas}")


if __name__ == "__main__":
    main()
