"""Fetch the extra data needed to backtest weeks 1-4, which the original
backtest never covered (it started at week 5).

Adds:
  - stats_{year}_1_1 .. _1_3   (stats available before weeks 2, 3, 4)
  - sp_{year}.json             (SP+ ratings -- the preseason prior the production
                                model uses but the backtest has never tested)
  - games_2021.json            (needed for 2022's "own model last year" prior)
"""
from fetch_data import BASE, _cached, _get, fetch_games, fetch_stats

SEASONS = [2022, 2023, 2024, 2025]


def fetch_sp(year):
    return _cached(f"sp_{year}.json", lambda: _get(f"{BASE}/ratings/sp", {"year": year}))


if __name__ == "__main__":
    # Prior-season games for the earliest backtested season's last-year prior.
    fetch_games(2021)

    for year in SEASONS:
        for w in (1, 2, 3):
            fetch_stats(year, 1, w)
        fetch_sp(year)

    print("Done fetching early-season data.")
