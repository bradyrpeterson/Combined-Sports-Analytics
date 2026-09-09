"""
Data fetching + caching layer for the model backtest.

All CFBD API calls are cached to disk under research/model_backtest/data/ so that
re-running the backtest never re-hits the API. Includes basic rate-limit courtesy
(sleep between calls) and 429 backoff, since this key is shared with the live site.
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

load_dotenv(os.path.join(REPO_ROOT, ".env"))
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not found -- did you copy .env into the worktree?")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE = "https://api.collegefootballdata.com"

SLEEP_BETWEEN_CALLS = 1.2  # be courteous, shared production key


def _get(url, params, retries=6):
    for attempt in range(retries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            wait = min(60, 2 ** attempt * 2)
            print(f"  [429] rate limited, backing off {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(SLEEP_BETWEEN_CALLS)
        return resp.json()
    raise RuntimeError(f"Failed after {retries} retries: {url} {params}")


def _cached(cache_name, fetch_fn):
    path = os.path.join(DATA_DIR, cache_name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"Fetching {cache_name} ...")
    data = fetch_fn()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def fetch_games(year):
    return _cached(f"games_{year}.json", lambda: _get(f"{BASE}/games", {"year": year}))


def fetch_stats(year, start_week, end_week):
    name = f"stats_{year}_{start_week}_{end_week}.json"
    return _cached(
        name,
        lambda: _get(
            f"{BASE}/stats/season",
            {"year": year, "startWeek": start_week, "endWeek": end_week},
        ),
    )


def fetch_lines(year):
    return _cached(
        f"lines_{year}.json",
        lambda: _get(f"{BASE}/lines", {"year": year, "seasonType": "regular"}),
    )


def fetch_lines_postseason(year):
    return _cached(
        f"lines_post_{year}.json",
        lambda: _get(f"{BASE}/lines", {"year": year, "seasonType": "postseason"}),
    )


def fetch_core(year):
    return _cached(f"core_{year}.json", lambda: _get(f"{BASE}/ratings/core", {"year": year}))


if __name__ == "__main__":
    SEASONS = [2022, 2023, 2024, 2025]

    for year in SEASONS:
        games = fetch_games(year)
        reg_weeks = sorted({g["week"] for g in games if g["seasonType"] == "regular"})
        last_week = max(reg_weeks)
        print(f"{year}: regular weeks {reg_weeks[0]}-{last_week}")

        # Stats for each walk-forward prediction week: predict week W using
        # stats accumulated through week W-1 (startWeek=1, endWeek=W-1).
        # Predictions start at week 5 (needs weeks 1-4 of data).
        for w in range(5, last_week + 1):
            fetch_stats(year, 1, w - 1)
        # Full regular-season stats snapshot, used for postseason predictions.
        fetch_stats(year, 1, last_week)

        fetch_lines(year)
        fetch_lines_postseason(year)

    # Prior-season CORE ratings (year-1) for each season being backtested, used
    # only as a prior-season prior -- never same-season (that would leak).
    for year in SEASONS:
        fetch_core(year - 1)

    print("Done fetching.")
