"""One-off import of Week 11 & 12 (2025 football) as aggregate weekly summaries.

Unlike Weeks 9-10 (imported per-game in import_historical_picks.py, where every
game's grading was cross-verified against real final scores), Weeks 11-12 only
came through as screenshots dense enough that per-game transcription couldn't be
verified to match the tracker's own totals exactly. Rather than guess at which
specific games won or lost, these are imported as a single lump event per week
using the bottom-line totals from the tracker itself -- trusted as-is per the
user. See tracking.get_track_record() for how these merge with per-game picks.

Run once: python3 scripts/import_weekly_summaries.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import firebase_admin
from firebase_admin import credentials, firestore

SEASON = 2025

# (week, date, sport, total_picks, straight_up_wins, ats_wins, ats_losses)
WEEKLY_SUMMARIES = [
    (11, "2025-11-08", "football", 48, 30, 32, 16),
    (12, "2025-11-15", "football", 57, 48, 37, 20),
]


def main():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    for week, date, sport, total_picks, straight_up_wins, ats_wins, ats_losses in WEEKLY_SUMMARIES:
        doc_id = f"{sport}_{SEASON}_week{week}"
        db.collection("weekly_summaries").document(doc_id).set({
            "sport": sport,
            "season": SEASON,
            "week": week,
            "date": date,
            "total_picks": total_picks,
            "straight_up_wins": straight_up_wins,
            "ats_wins": ats_wins,
            "ats_losses": ats_losses,
            "source": "manual_aggregate",
        })
        print(f"Wrote {doc_id}: {total_picks} picks, {straight_up_wins} SU, {ats_wins}-{ats_losses} ATS")

    import tracking
    record = tracking.get_track_record(db, sport="football", season=SEASON)
    print("Post-import track record (football, 2025):", record)


if __name__ == "__main__":
    main()
