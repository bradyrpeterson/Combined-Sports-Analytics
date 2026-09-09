"""Snapshot predictions before games are played, settle them against final scores,
and aggregate the results into a public track record.

Firestore collection: tracked_picks
Doc id: {sport}_{date}_{home}_{away}  (deterministic -> idempotent snapshotting)
"""
import os
from datetime import datetime, timezone

import requests
from google.cloud.firestore_v1.base_query import FieldFilter
import pytz

BET_STAKE = 10
BET_TO_WIN = 9.09  # standard -110 odds on a flat $10 bet

#Only games where our number differs from the market by at least this much count
#toward the public record. Picks are still snapshotted and graded at the 3-point
#threshold the site highlights -- this filter is applied when reading, so the
#stored history keeps its full detail and the cut can be revisited.
#Backtesting 2022-2025 found the >=5 tier was the only one that graded at or
#above break-even; the 3-5 tier lost money in every model tested.
RECOMMENDED_MIN_EDGE = 5

#The lower threshold the site flags at. Games between the two thresholds are graded
#into their own bucket by get_track_record (the "yellow_*" keys) so the two tiers can
#be compared later without mixing the weaker one into the headline record.
FLAGGED_MIN_EDGE = 3


def _slug(name):
    return name.replace(" ", "-").replace("/", "-")


def _doc_id(sport, date, home, away):
    return f"{sport}_{date}_{_slug(home)}_{_slug(away)}"


def _model_home_margin(pick):
    """Recover the signed, home-positive model margin from a predictions-df row."""
    return pick["margin"] if pick["predicted_winner"] == pick["home"] else -pick["margin"]


def _football_season_for_date(date_str):
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year - 1 if month == 1 else year  # January games belong to the prior season label


def _basketball_season_for_date(date_str):
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year + 1 if month >= 8 else year  # season is labeled by its ending year


def _get_football_final(headers, date_str, home, away):
    season = _football_season_for_date(date_str)
    url = f"https://api.collegefootballdata.com/games?year={season}&team={home}"
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    for game in resp.json():
        if game.get("homeTeam") != home or game.get("awayTeam") != away:
            continue
        if (game.get("startDate") or "")[:10] != date_str:
            continue
        home_points, away_points = game.get("homePoints"), game.get("awayPoints")
        if home_points is not None and away_points is not None:
            return home_points, away_points
    return None


def _get_basketball_final(headers, date_str, home, away):
    season = _basketball_season_for_date(date_str)
    est = pytz.timezone("America/New_York")
    day = est.localize(datetime.strptime(date_str, "%Y-%m-%d"))
    start_utc = day.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(pytz.UTC)
    end_utc = day.replace(hour=23, minute=59, second=59, microsecond=999999).astimezone(pytz.UTC)
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_str = end_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    url = (
        "https://api.collegebasketballdata.com/games"
        f"?season={season}&startDateRange={start_str}&endDateRange={end_str}"
    )
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    for game in resp.json():
        if game.get("homeTeam") == home and game.get("awayTeam") == away and game.get("status") == "final":
            return game.get("homePoints"), game.get("awayPoints")
    return None


def snapshot_todays_picks(db, football_predictor=None, basketball_predictor=None):
    """Log any not-yet-seen upcoming picks as pending tracked_picks docs."""
    saved = 0
    predictors = [("football", football_predictor), ("basketball", basketball_predictor)]
    for sport, predictor in predictors:
        if predictor is None:
            continue
        try:
            predictions_df = predictor.get_upcoming_predictions()
        except Exception as e:
            print(f"[tracking] Error getting {sport} predictions to snapshot: {e}")
            continue

        for _, pick in predictions_df.iterrows():
            date = pick.get("date")
            if not date:
                continue

            doc_id = _doc_id(sport, date, pick["home"], pick["away"])
            doc_ref = db.collection("tracked_picks").document(doc_id)
            if doc_ref.get().exists:
                continue

            betting_spread = pick.get("betting_spread")
            recommended = bool(pick.get("edge_class"))
            recommended_side = None
            if betting_spread is not None and recommended:
                model_home_margin = _model_home_margin(pick)
                betting_home_margin = -betting_spread
                recommended_side = "home" if model_home_margin > betting_home_margin else "away"

            doc_ref.set({
                "sport": sport,
                "date": date,
                "season": (
                    _football_season_for_date(date) if sport == "football"
                    else _basketball_season_for_date(date)
                ),
                "home": pick["home"],
                "away": pick["away"],
                "predicted_winner": pick["predicted_winner"],
                "model_margin": float(pick["margin"]),
                "win_prob": float(pick["prob"]),
                "betting_spread": None if betting_spread is None else float(betting_spread),
                "edge": None if pick.get("spread_diff") is None else abs(float(pick["spread_diff"])),
                "recommended": recommended,
                "recommended_side": recommended_side,
                "status": "pending",
                "source": "live",
                "created_at": datetime.now(timezone.utc),
            })
            saved += 1
    return saved


def settle_pending_picks(db, football_predictor=None, basketball_predictor=None):
    """Grade any pending picks whose games have finished."""
    headers = {
        "football": getattr(football_predictor, "headers", None),
        "basketball": getattr(basketball_predictor, "headers", None),
    }
    settled = 0
    pending_docs = db.collection("tracked_picks").where(filter=FieldFilter("status", "==", "pending")).stream()

    for doc in pending_docs:
        data = doc.to_dict()
        sport = data["sport"]
        if headers.get(sport) is None:
            continue

        try:
            if sport == "football":
                result = _get_football_final(headers["football"], data["date"], data["home"], data["away"])
            else:
                result = _get_basketball_final(headers["basketball"], data["date"], data["home"], data["away"])
        except Exception as e:
            print(f"[tracking] Error fetching final score for {data['home']} vs {data['away']}: {e}")
            continue

        if result is None:
            continue  # game hasn't finished yet, try again next run

        home_points, away_points = result
        actual_winner = data["home"] if home_points > away_points else data["away"]
        straight_up_correct = actual_winner == data["predicted_winner"]

        ats_result = None
        if data.get("recommended") and data.get("betting_spread") is not None:
            actual_home_margin = home_points - away_points
            cover_margin = actual_home_margin + data["betting_spread"]
            if cover_margin == 0:
                ats_result = "push"
            else:
                home_covered = cover_margin > 0
                picked_home = data["recommended_side"] == "home"
                ats_result = "win" if (home_covered == picked_home) else "loss"

        doc.reference.update({
            "status": "final",
            "actual_home_score": home_points,
            "actual_away_score": away_points,
            "straight_up_correct": straight_up_correct,
            "ats_result": ats_result,
            "settled_at": datetime.now(timezone.utc),
        })
        settled += 1

    return settled


def get_recent_results(db, sport=None, limit=10, min_edge=FLAGGED_MIN_EDGE):
    """Most recently settled *recommended* picks, newest first -- both the yellow
    (3-5 pt) and green (>=5 pt) tiers -- with how each one finished on the
    moneyline and against the spread. Powers the results panel on the landing page."""
    query = db.collection("tracked_picks").where(filter=FieldFilter("status", "==", "final"))
    if sport:
        query = query.where(filter=FieldFilter("sport", "==", sport))

    picks = [d.to_dict() for d in query.stream()]
    picks = [
        p for p in picks
        if p.get("actual_home_score") is not None
        and p.get("recommended")
        and (p.get("edge") or 0) >= min_edge
    ]
    picks.sort(key=lambda p: p.get("date", ""), reverse=True)

    results = []
    for p in picks[:limit]:
        home_score, away_score = p["actual_home_score"], p["actual_away_score"]
        winner = p["home"] if home_score > away_score else p["away"]
        results.append({
            "sport": p.get("sport"),
            "date": p.get("date"),
            "home": p["home"],
            "away": p["away"],
            "home_score": home_score,
            "away_score": away_score,
            "actual_winner": winner,
            "actual_margin": abs(home_score - away_score),
            "predicted_winner": p.get("predicted_winner"),
            "model_margin": p.get("model_margin"),
            "win_prob": p.get("win_prob"),
            #Moneyline: did we call the outright winner? Spread: did the side cover?
            "ml_correct": bool(p.get("straight_up_correct")),
            #Green (>=5) picks are the ones that count toward the tracked record;
            #yellow (3-5) picks are shown here but held out of it.
            "tier": "high" if (p.get("edge") or 0) >= RECOMMENDED_MIN_EDGE else "medium",
            "counts_in_record": (p.get("edge") or 0) >= RECOMMENDED_MIN_EDGE,
            "recommended_side": p.get("recommended_side"),
            "recommended_team": (
                p["home"] if p.get("recommended_side") == "home"
                else p["away"] if p.get("recommended_side") == "away" else None
            ),
            "betting_spread": p.get("betting_spread"),
            "ats_result": p.get("ats_result"),
            "edge": p.get("edge"),
        })
    return results


def get_track_record(db, sport=None, season=None):
    """Aggregate settled picks + weekly aggregate summaries into headline stats and a
    cumulative flat-bet profit series.

    Most weeks are graded per game (tracked_picks). Some weeks are only trustworthy as a
    bottom-line total (e.g. a manually-tracked week where individual game grading couldn't
    be verified but the week's own reported totals could be) -- those live in
    weekly_summaries and contribute one lump event to the totals/chart instead of one
    event per game.
    """
    pick_query = db.collection("tracked_picks").where(filter=FieldFilter("status", "==", "final"))
    summary_query = db.collection("weekly_summaries")
    if sport:
        pick_query = pick_query.where(filter=FieldFilter("sport", "==", sport))
        summary_query = summary_query.where(filter=FieldFilter("sport", "==", sport))
    if season:
        pick_query = pick_query.where(filter=FieldFilter("season", "==", season))
        summary_query = summary_query.where(filter=FieldFilter("season", "==", season))

    events = [{"date": p["date"], "kind": "pick", "data": p} for p in (doc.to_dict() for doc in pick_query.stream())]
    events += [{"date": s["date"], "kind": "summary", "data": s} for s in (doc.to_dict() for doc in summary_query.stream())]
    events.sort(key=lambda e: e["date"])

    total = 0
    straight_up_wins = 0
    ats_wins = 0
    ats_losses = 0
    profit = 0.0
    profit_series = []
    yellow_wins = 0
    yellow_losses = 0
    yellow_profit = 0.0

    for event in events:
        data = event["data"]
        if event["kind"] == "pick":
            total += 1
            if data.get("straight_up_correct"):
                straight_up_wins += 1
            edge = data.get("edge") or 0
            graded = data.get("recommended") and data.get("ats_result") in ("win", "loss")
            won = data.get("ats_result") == "win"
            if graded and edge >= RECOMMENDED_MIN_EDGE:
                if won:
                    ats_wins += 1
                    profit += BET_TO_WIN
                else:
                    ats_losses += 1
                    profit -= BET_STAKE
                profit_series.append({"date": data["date"], "profit": round(profit, 2)})
            elif graded and edge >= FLAGGED_MIN_EDGE:
                if won:
                    yellow_wins += 1
                    yellow_profit += BET_TO_WIN
                else:
                    yellow_losses += 1
                    yellow_profit -= BET_STAKE
        else:
            #Weekly summaries are bottom-line totals with no per-game edge stored, so
            #they can't be split by the RECOMMENDED_MIN_EDGE cut. They count toward
            #games graded, but are left out of the recommended record and ROI rather
            #than silently mixing 3-5 point picks into a >=5 point number.
            total += data["total_picks"]
            straight_up_wins += data["straight_up_wins"]

    ats_decided = ats_wins + ats_losses
    yellow_decided = yellow_wins + yellow_losses
    return {
        "total_picks": total,
        "straight_up_win_pct": round(100 * straight_up_wins / total, 1) if total else None,
        "recommended_count": ats_decided,
        "ats_win_pct": round(100 * ats_wins / ats_decided, 1) if ats_decided else None,
        "profit_series": profit_series,
        "total_profit": round(profit, 2),
        "roi_pct": round(100 * profit / (ats_decided * BET_STAKE), 1) if ats_decided else None,
        #The 3-5 point tier, graded on its own and kept out of every number above.
        "yellow_count": yellow_decided,
        "yellow_ats_win_pct": round(100 * yellow_wins / yellow_decided, 1) if yellow_decided else None,
        "yellow_profit": round(yellow_profit, 2),
        "yellow_roi_pct": round(100 * yellow_profit / (yellow_decided * BET_STAKE), 1) if yellow_decided else None,
    }
