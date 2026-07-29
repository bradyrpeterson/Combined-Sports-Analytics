"""Snapshot predictions before games are played, settle them against final scores,
and aggregate the results into a public track record.

Firestore collection: tracked_picks
Doc id: {sport}_{date}_{home}_{away}  (deterministic -> idempotent snapshotting)
"""
import os
from datetime import datetime, timezone

import requests
import pytz

BET_STAKE = 10
BET_TO_WIN = 9.09  # standard -110 odds on a flat $10 bet


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
    pending_docs = db.collection("tracked_picks").where("status", "==", "pending").stream()

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


def get_track_record(db, sport=None, season=None):
    """Aggregate settled picks + weekly aggregate summaries into headline stats and a
    cumulative flat-bet profit series.

    Most weeks are graded per game (tracked_picks). Some weeks are only trustworthy as a
    bottom-line total (e.g. a manually-tracked week where individual game grading couldn't
    be verified but the week's own reported totals could be) -- those live in
    weekly_summaries and contribute one lump event to the totals/chart instead of one
    event per game.
    """
    pick_query = db.collection("tracked_picks").where("status", "==", "final")
    summary_query = db.collection("weekly_summaries")
    if sport:
        pick_query = pick_query.where("sport", "==", sport)
        summary_query = summary_query.where("sport", "==", sport)
    if season:
        pick_query = pick_query.where("season", "==", season)
        summary_query = summary_query.where("season", "==", season)

    events = [{"date": p["date"], "kind": "pick", "data": p} for p in (doc.to_dict() for doc in pick_query.stream())]
    events += [{"date": s["date"], "kind": "summary", "data": s} for s in (doc.to_dict() for doc in summary_query.stream())]
    events.sort(key=lambda e: e["date"])

    total = 0
    straight_up_wins = 0
    ats_wins = 0
    ats_losses = 0
    profit = 0.0
    profit_series = []

    for event in events:
        data = event["data"]
        if event["kind"] == "pick":
            total += 1
            if data.get("straight_up_correct"):
                straight_up_wins += 1
            if data.get("recommended") and data.get("ats_result") in ("win", "loss"):
                if data["ats_result"] == "win":
                    ats_wins += 1
                    profit += BET_TO_WIN
                else:
                    ats_losses += 1
                    profit -= BET_STAKE
                profit_series.append({"date": data["date"], "profit": round(profit, 2)})
        else:
            total += data["total_picks"]
            straight_up_wins += data["straight_up_wins"]
            ats_wins += data["ats_wins"]
            ats_losses += data["ats_losses"]
            profit += data["ats_wins"] * BET_TO_WIN - data["ats_losses"] * BET_STAKE
            profit_series.append({"date": data["date"], "profit": round(profit, 2)})

    ats_decided = ats_wins + ats_losses
    return {
        "total_picks": total,
        "straight_up_win_pct": round(100 * straight_up_wins / total, 1) if total else None,
        "recommended_count": ats_decided,
        "ats_win_pct": round(100 * ats_wins / ats_decided, 1) if ats_decided else None,
        "profit_series": profit_series,
        "total_profit": round(profit, 2),
    }
