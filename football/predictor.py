#Import needed packages
import pandas as pd
import numpy as np
import cfbd #College Football Data API
import requests #For pulling data from CFBD
import json #For handling json files
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
#Using configuration suggested by CFBD turn the games into a dataset
configuration = cfbd.Configuration(
    access_token = api_key)
headers = {"Authorization": f"Bearer {api_key}"}

#Only care about games where one of the teams was FBS
# Load list of FBS teams (static roster file, doesn't need refreshing)
with open("football/fbs_teams_2026.json", "r") as f:
    fbs_teams = json.load(f)

_raw_stat_cols = ["team", "totalYards", "rushingAttempts", "passAttempts",
                  "totalYardsOpponent", "rushingAttemptsOpponent", "passAttemptsOpponent",
                  "thirdDownConversions", "thirdDowns", "turnoversOpponent", "turnovers"]

need_cols = ["season","seasonType","week","startDate","startTimeTBD","venue","venueId",
             "homeTeam","awayTeam","homePoints","awayPoints","homeConference","awayConference","neutralSite"]

#--- Preseason prior weighting ----------------------------------------------
#The prior itself is a 50/50 blend of SP+ (this year's advanced-metric
#projection) and our own rating regression run on last season's complete
#results. Backtested across 2022-2025: an even blend of the two beat using
#either alone in every single season tested (e.g. weeks-1-4 MAE of 14.2 for
#the 50/50 blend vs 17.3 for SP+ alone and 15.1 for our own model alone).
#SP+ alone actually did worse than expected early on -- it's a whole-season
#average, so a team that starts slow and turns it on in October drags that
#average away from what actually happened in week 1. Our own model doesn't
#share that blind spot, so combining the two cancels out some of each one's
#individual misses rather than just being "a little of both for variety."
SP_PLUS_WEIGHT = 0.5
OWN_MODEL_WEIGHT = 0.5

#Now fade the (already-blended) prior into this season's own results as
#harmonic_k / (harmonic_k + weeks_completed). This decay schedule was picked
#the same way: it beat every other schedule tried (hard cutoffs, static
#blends, faster/slower linear decays) both early in the season AND well past
#week 4 -- with only this year's games, the ratings regression above stays
#noisy for a lot longer than 4 weeks because most FBS teams have only played
#a handful of common opponents by then. The weight never truly reaches zero
#(~0.6 at week 4, ~0.3 by week 12) -- forcing it to zero sooner tested worse
#in every season checked.
PRIOR_DECAY_K = 6  # weight = k / (k + weeks_completed); tuned via backtest

#How many weeks of this season's own games we want in hand before trusting
#predictions enough to count them toward the tracked recommended record --
#below this, ratings are still mostly the preseason SP+/last-year prior.
MODEL_FULLY_TRAINED_MIN_WEEKS = 4

#Don't hit the CFBD API more than once per this many seconds -- refresh() is
#called at the top of every page load (see app.py) so the site never shows
#stale scores without a redeploy, but a burst of page views (or one visitor
#clicking between pages) shouldn't turn into a burst of redundant API calls.
MIN_REFRESH_INTERVAL_SECONDS = 3600
_last_refresh_time = 0.0


def _fit_team_ratings(games_df):
    """Home/away indicator regression on margin -- same method used for both
    the live in-season ratings below and the backtest that picked the preseason
    blend (see PRESEASON PRIOR section). Returns (ratings Series, home_field)."""
    if len(games_df) == 0:
        return pd.Series(dtype=float), 0.0

    #Build the design matrix
    teams_in = sorted(set(games_df["homeTeam"]).union(games_df["awayTeam"]))
    #Home teams get a +1 value and -1 is for away
    #This setup allows for the regression to assign each team a numeric rating
    X = pd.DataFrame(0, index=np.arange(len(games_df)), columns=teams_in)
    for i, row in games_df.reset_index(drop=True).iterrows():
        X.loc[i, row["homeTeam"]] = 1    # +1 for home team
        X.loc[i, row["awayTeam"]] = -1   # -1 for away team

    #Add home field column
    X["home_field"] = 1

    #Time to fit the margin
    #The regressions finds coefficients that best fit the margins
    #Don't worry about an intercept cause we have home field
    y = games_df["margin"].reset_index(drop=True)

    #Create the linear regression based on the margins
    #Essentially finds a set of ratings that makes the predicted scores as close
    #as possible to what actually happened
    model = LinearRegression(fit_intercept=False)
    #Fit a model that predicts y (point margin) as a linear
    #combination of the columns in X
    model.fit(X, y)

    #Ensure home field is counted for in the team ratings
    #Every team gets a numeric rating
    #home_field isolates the average value that being at home bears
    #Substrating the mean centers the ratings at 0 so the average team has that rating
    coefs = pd.Series(model.coef_, index=X.columns)
    hf = coefs["home_field"]
    r = coefs.drop("home_field")
    #Make the average team=0
    r -= r.mean()
    return r, hf


def train_prediction_model(completed, ratings, stats_clean, fbs_teams):
    """
    Train ML model to learn optimal weights from historical data.
    Called fresh by load_data() every refresh so it always fits this
    season's latest results.
    """

    features = []
    targets = []

    for _, game in completed.iterrows():
        home = game['homeTeam']
        away = game['awayTeam']
        margin = game['homePoints'] - game['awayPoints']
        neutral = game.get('neutralSite', False)

        # Skip if missing data
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_clean['team'].values or away not in stats_clean['team'].values:
            continue
        # Skip non-FBS opponents -- lopsided "buy games" would teach the
        # model the wrong thing about what a given rating gap actually means.
        if home not in fbs_teams or away not in fbs_teams:
            continue

        # Get features
        h_stats = stats_clean[stats_clean['team'] == home].iloc[0]
        a_stats = stats_clean[stats_clean['team'] == away].iloc[0]

        rating_diff = ratings[home] - ratings[away]
        ypp_diff = h_stats['yardsPerPlay_off'] - a_stats['yardsPerPlay_def']
        third_diff = h_stats['thirdDownPct'] - a_stats['thirdDownPct']
        to_diff = h_stats['turnoverMargin'] - a_stats['turnoverMargin']
        hc = 0 if neutral else 1

        # Skip if NaN
        if pd.isna([rating_diff, ypp_diff, third_diff, to_diff]).any():
            continue

        feature_vector = [
            rating_diff,
            ypp_diff,
            third_diff,
            to_diff,
            hc,
            ratings[home],
            ratings[away],
            h_stats['yardsPerPlay_off'],
            a_stats['yardsPerPlay_off'],
            h_stats['yardsPerPlay_def'],
            a_stats['yardsPerPlay_def']
        ]

        features.append(feature_vector)
        targets.append(margin)

    if len(features) == 0:
        # No games yet have both a result and stats to train on (e.g. stats
        # haven't been published this early in the season) -- nothing to fit.
        return None

    X = np.array(features)
    y = np.array(targets)

    # Remove any NaN rows (safety check)
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X = X[~nan_mask]
        y = y[~nan_mask]

    if len(X) == 0:
        return None

    # Train model
    ml_model = LinearRegression()
    ml_model.fit(X, y)

    return ml_model


def load_data():
    """Fetch everything fresh from CFBD (games, venues, SP+, last season, box-score
    stats, betting-relevant metadata) and retrain the model on it. This is the
    entire body of what used to run once at import time -- now it's callable, so
    refresh() can re-run it on every page load instead of only on process start."""
    with cfbd.ApiClient(configuration) as api_client:
        api_instance = cfbd.GamesApi(api_client)
        games = api_instance.get_games(year=2026)

        venues_api = cfbd.VenuesApi(api_client)
        try:
            venues = venues_api.get_venues()
        except Exception as e:
            print(f"Error fetching venues: {e}")
            venues = []

        #SP+ (Bill Connelly's advanced rating) for the current season -- this early,
        #before many/any games are played, it *is* the preseason projection. Used
        #below as half of the preseason prior for team ratings.
        try:
            sp_plus = cfbd.RatingsApi(api_client).get_sp(year=2026)
        except Exception as e:
            print(f"Error fetching SP+ ratings: {e}")
            sp_plus = []

        #Last season's own completed games -- the other half of the preseason
        #prior (see PRESEASON PRIOR below). Backtested against using SP+ alone
        #across 2022-2025: blending in our own last-year rating beat SP+ alone
        #(and beat our own model alone) in every one of those seasons.
        try:
            last_season_games = cfbd.GamesApi(api_client).get_games(year=2025)
            last_season_fbs = [t.school for t in cfbd.TeamsApi(api_client).get_fbs_teams(year=2025)]
        except Exception as e:
            print(f"Error fetching last season's games: {e}")
            last_season_games, last_season_fbs = [], []

    #Map venue id -> "City, State" so game locations can be shown alongside the venue name
    venue_location = {}
    for v in venues:
        v = v.to_dict()
        city_state = ", ".join(p for p in [v.get("city"), v.get("state")] if p)
        if v.get("id") is not None and city_state:
            venue_location[v["id"]] = city_state

    #Using requests pull all the statistical data from the data set
    stats_url = "https://api.collegefootballdata.com/stats/season?year=2026"
    #Convert the API response into a json then a dataframe for easy use
    stats_response = requests.get(stats_url, headers=headers)
    stats_data = stats_response.json()

    #CFBD's season-stats aggregate lags kickoff by a day or more, so early in the
    #season (or before it starts) this comes back with zero rows. Fall back to an
    #empty frame with the right columns instead of crashing the pivot below --
    #get_upcoming_predictions already skips any game missing stats, so games just
    #don't show up until real stats are published.
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        #Reshape stats to have one row per team
        stats_wide = stats_df.pivot(index="team", columns="statName", values="statValue").reset_index()
        for _col in _raw_stat_cols:
            if _col not in stats_wide.columns:
                stats_wide[_col] = np.nan
    else:
        stats_wide = pd.DataFrame(columns=_raw_stat_cols)

    # Create efficiency stats using the dataset
    stats_wide["yardsPerPlay_off"] = stats_wide["totalYards"] / (stats_wide["rushingAttempts"] + stats_wide["passAttempts"])
    stats_wide["yardsPerPlay_def"] = stats_wide["totalYardsOpponent"] / (stats_wide["rushingAttemptsOpponent"] + stats_wide["passAttemptsOpponent"])
    stats_wide["thirdDownPct"] = stats_wide["thirdDownConversions"] / stats_wide["thirdDowns"]
    stats_wide["turnoverMargin"] = stats_wide["turnoversOpponent"] - stats_wide["turnovers"]

    # Keep only the stats/columns that I plan on using
    useful = ["team", "yardsPerPlay_off", "yardsPerPlay_def", "thirdDownPct", "turnoverMargin"]
    stats_clean = stats_wide[useful]

    #Each game is turned into a dictionary then into a dataframe
    df = pd.DataFrame([g.to_dict() for g in games])
    #Only keep the columns that matter
    df = df[need_cols].copy()
    #Only keep games if it involved an FBS team
    df = df[df["homeTeam"].isin(fbs_teams) | df["awayTeam"].isin(fbs_teams)]
    df = df.reset_index(drop=True)

    #Need to make an upcoming data frame as well as a completed data frame
    completed = df.dropna(subset=["homePoints", "awayPoints"]).reset_index(drop=True)
    upcoming = df[df["homePoints"].isna() | df["awayPoints"].isna()].reset_index(drop=True)
    regular_upcoming = upcoming[upcoming["seasonType"] == "regular"]

    if len(regular_upcoming) > 0:
        next_week = int(regular_upcoming["week"].dropna().sort_values().unique()[0])
    else:
        next_week = 17

    #define what margin is
    #sort the dataframe to have a line of home and away teams
    completed = completed.copy()
    completed["margin"] = completed["homePoints"] - completed["awayPoints"]

    #Early in the season (or before it starts) there may be zero completed FBS
    #games yet -- can't fit a regression on zero rows, so this comes back empty.
    #The preseason-prior blend below covers that gap.
    current_ratings, current_home_field = _fit_team_ratings(completed)

    #Last season's own final ratings, computed with this same method -- the
    #other half of the preseason prior (see below).
    last_season_df = pd.DataFrame([g.to_dict() for g in last_season_games])
    if len(last_season_df) > 0:
        last_season_df = last_season_df[
            last_season_df["homeTeam"].isin(last_season_fbs) | last_season_df["awayTeam"].isin(last_season_fbs)
        ]
        last_season_df = last_season_df.dropna(subset=["homePoints", "awayPoints"]).reset_index(drop=True)
        last_season_df["margin"] = last_season_df["homePoints"] - last_season_df["awayPoints"]
    own_last_year_rating, own_last_year_home_field = _fit_team_ratings(last_season_df)

    preseason_rating_sp = pd.Series(
        {t.team: t.rating for t in sp_plus if t.team in fbs_teams}
    )
    if len(preseason_rating_sp) > 0:
        preseason_rating_sp -= preseason_rating_sp.mean()

    all_prior_teams = set(fbs_teams) | set(preseason_rating_sp.index) | set(own_last_year_rating.index)
    sp_full = preseason_rating_sp.reindex(all_prior_teams).fillna(0.0)
    own_full = own_last_year_rating.reindex(all_prior_teams).fillna(0.0)
    preseason_rating = SP_PLUS_WEIGHT * sp_full + OWN_MODEL_WEIGHT * own_full
    preseason_home_field = SP_PLUS_WEIGHT * 2.2 + OWN_MODEL_WEIGHT * own_last_year_home_field
    #2.2 = SP+'s implied home-field share -- it doesn't publish a home-field
    #number of its own, so this uses the average home_field coefficient observed
    #across the 2022-2025 backtest seasons in its place.

    regular_completed_weeks = completed[completed["seasonType"] == "regular"]["week"].dropna()
    weeks_completed = int(regular_completed_weeks.max()) if len(regular_completed_weeks) > 0 else 0

    prior_weight = PRIOR_DECAY_K / (PRIOR_DECAY_K + weeks_completed)

    all_rated_teams = set(fbs_teams) | set(current_ratings.index) | set(preseason_rating.index)
    current_full = current_ratings.reindex(all_rated_teams).fillna(0.0)
    prior_full = preseason_rating.reindex(all_rated_teams).fillna(0.0)

    ratings = prior_weight * prior_full + (1 - prior_weight) * current_full
    home_field = prior_weight * preseason_home_field + (1 - prior_weight) * current_home_field

    # Create FBS-only rankings
    FBS_rankings = pd.DataFrame({'team': ratings.index, 'rating': ratings.values})
    FBS_rankings = FBS_rankings[FBS_rankings['team'].isin(fbs_teams)]
    FBS_rankings = FBS_rankings.sort_values(by='rating', ascending=False).reset_index(drop=True)

    prediction_model = train_prediction_model(completed, ratings, stats_clean, fbs_teams)

    model_fully_trained = weeks_completed >= MODEL_FULLY_TRAINED_MIN_WEEKS and prediction_model is not None

    return {
        "games": games,
        "venue_location": venue_location,
        "stats_clean": stats_clean,
        "completed": completed,
        "upcoming": upcoming,
        "next_week": next_week,
        "ratings": ratings,
        "home_field": home_field,
        "FBS_rankings": FBS_rankings,
        "prediction_model": prediction_model,
        "weeks_completed": weeks_completed,
        "model_fully_trained": model_fully_trained,
    }


def refresh(force=False):
    """Re-fetch everything from CFBD and retrain the model, replacing this
    module's data in place. Call this before serving any page (see app.py)
    so the site always reflects the current games/scores instead of whatever
    was live when the process last started -- no redeploy required.

    Throttled to once every MIN_REFRESH_INTERVAL_SECONDS so back-to-back page
    loads (or several visitors at once) don't turn into a burst of redundant
    CFBD calls; pass force=True to bypass that (e.g. an admin refresh button)."""
    global _last_refresh_time
    now = time.monotonic()
    if not force and (now - _last_refresh_time) < MIN_REFRESH_INTERVAL_SECONDS:
        return
    globals().update(load_data())
    _last_refresh_time = now


# Load data once at import so the module works even if nobody calls refresh()
# (e.g. a script importing this directly). app.py calls refresh() again at
# the top of every request.
refresh(force=True)


#Prediciton function
def predict_game(home, away, neutral_site=False):
    rating_diff = ratings[home] - ratings[away]
    #Whether or not home field advantage is applied
    home_advantage = 0 if neutral_site else 1

    #Pull the home and away team stats and compare them, if we have them yet.
    #Early in the season (or before it starts) CFBD's season-stats aggregate
    #is empty -- fall back to a plain rating + home-field margin instead of
    #crashing or refusing to predict the game at all.
    h_stats_rows = stats_clean.loc[stats_clean["team"] == home]
    a_stats_rows = stats_clean.loc[stats_clean["team"] == away]

    if prediction_model is not None and len(h_stats_rows) > 0 and len(a_stats_rows) > 0:
        h_stats = h_stats_rows.iloc[0]
        a_stats = a_stats_rows.iloc[0]

        # Compute stat differences between the two teams
        ypp_diff = h_stats["yardsPerPlay_off"] - a_stats["yardsPerPlay_def"]
        third_down_diff = h_stats["thirdDownPct"] - a_stats["thirdDownPct"]
        turnover_diff = h_stats["turnoverMargin"] - a_stats["turnoverMargin"]

        #Old arbitrary weighting system before ML model
        #w_rating=0.7
        #_ypp=0.1
        #w_third=0.05
        #w_turnover=0.15
        #margin=(w_rating*rating_diff+(w_ypp*ypp_diff*10)+(w_third*third_down_diff*20)+(w_turnover*turnover_diff)+home_advantage)
        # Built feature vector
        features = np.array([[
            rating_diff,
            ypp_diff,
            third_down_diff,
            turnover_diff,
            home_advantage,
            ratings[home],
            ratings[away],
            h_stats['yardsPerPlay_off'],
            a_stats['yardsPerPlay_off'],
            h_stats['yardsPerPlay_def'],
            a_stats['yardsPerPlay_def']
        ]])
        if not np.isnan(features).any():
            # Predict margin using the trained model
            margin = prediction_model.predict(features)[0]
            #Calculate probabiliy based on the idea that a team favored by 7
            #has a 75% chance to win
            prob = 1 / (1 + np.exp(-margin / 7))  # rough logistic
            return margin, prob

    # No box-score stats for this matchup yet (or no trained model) -- predict
    # off ratings (preseason-prior-blended early in the season) + home field alone.
    margin = rating_diff + home_field * home_advantage
    prob = 1 / (1 + np.exp(-margin / 7))
    return margin, prob

def get_betting_lines(week, year=2026, season_type="regular"):
   #Get draftkings specific betting lines for the week
    if season_type == "postseason":
        # Postseason doesn't use week numbers
        lines_url = f"https://api.collegefootballdata.com/lines?year={year}&seasonType=postseason"
    else:
        lines_url = f"https://api.collegefootballdata.com/lines?year={year}&week={week}&seasonType=regular"

    try:
        response = requests.get(lines_url, headers=headers)
        response.raise_for_status()
        lines_data = response.json()

        betting_lines = {}

        for game in lines_data:
            home = game.get("homeTeam")
            away = game.get("awayTeam")
            lines = game.get("lines", [])

            if not home or not away or not lines:
                continue

            # Look specifically for DraftKings line
            draftkings_spread = None
            for line in lines:
                if line.get("provider") == "DraftKings":
                    draftkings_spread = line.get("spread")
                    break

            if draftkings_spread is not None:
                betting_lines[(home, away)] = draftkings_spread

        return betting_lines

    except Exception as e:
        print(f"Error fetching betting lines: {e}")
        return {}

def calculate_edge_highlight(model_margin, betting_spread):
    # Convert betting spread to match our model's convention
    # If betting spread is +7, that means home is favored by 7
    # If betting spread is -7, that means away is favored by 7
    # We need to flip it to match our model's convention (positive = home favored)
    betting_margin = -betting_spread

    # Calculate the difference
    difference = abs(model_margin - betting_margin)

    if difference >= 5:
        return 'edge-big'
    elif difference >= 3:
        return 'edge-medium'
    else:
        return None

def get_upcoming_predictions(week=None,conference=None):
    # Use the upcoming games dataset (no scores yet)
    games_to_predict = upcoming.copy()

    if week is not None:
        if int(week)>16:
            games_to_predict = games_to_predict[games_to_predict["seasonType"]== "postseason"]
        else:
            games_to_predict = games_to_predict[(games_to_predict["week"].astype(int) == int(week))&(games_to_predict["seasonType"]=="regular")]
    else:
        games_to_predict =games_to_predict[games_to_predict["seasonType"]=="regular"]
    #Filter by conferences
    if conference is not None:
        games_to_predict = games_to_predict[
            (games_to_predict["homeConference"] == conference) |
            (games_to_predict["awayConference"] == conference)
        ]
    # Fetch betting lines for this week
    if week and int(week) > 16:
        betting_lines = get_betting_lines(week, season_type="postseason")
    else:
        betting_lines = get_betting_lines(week if week else next_week, season_type="regular")

    predictions = []
    for _, game in games_to_predict.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]

        is_neutral = game.get("neutralSite",False)

        #skip games where data
        #Helps avoid faulty data by skipping games missing a team rating.
        #(Box-score stats aren't required -- predict_game falls back to
        #ratings + home field when they're not published yet.)
        #Also requires both teams to be FBS: a non-FBS opponent (FCS "buy
        #games" etc.) still gets a coefficient out of the ratings regression
        #since it played an FBS team, but that number means nothing -- it's
        #fit on a handful of lopsided games and centered against FBS
        #competition, not against its own level. That's what produced spreads
        #20-35 points off Vegas for exactly these matchups.
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in fbs_teams or away not in fbs_teams:
            continue

        try:
            margin, prob = predict_game(home, away, neutral_site=is_neutral)
            winner = home if margin > 0 else away

            # Get betting line for this game
            betting_spread = betting_lines.get((home, away), None)

            # Calculate edge highlight
            edge_class = None
            spread_diff = None
            if betting_spread is not None:
                edge_class = calculate_edge_highlight(margin, betting_spread)
                # Calculate the actual difference for display
                betting_margin = -betting_spread
                spread_diff = round(margin - betting_margin, 1)

            # While the model is still mostly running on the preseason prior
            # (not enough of this season's own games yet -- see
            # model_fully_trained above), don't let these picks count toward
            # the tracked recommended record. tracking.py only counts a
            # snapshotted pick as "recommended" when edge_class is set, so
            # forcing it to None here is what keeps it out of that tally --
            # the game still shows on the page with its edge highlighted,
            # it just isn't graded.
            if not model_fully_trained:
                edge_class = None

            game_date = pd.to_datetime(game.get("startDate"), utc=True, errors="coerce")
            game_date_et = game_date.tz_convert("America/New_York") if pd.notna(game_date) else None
            is_tbd = bool(game.get("startTimeTBD", False))

            venue_name = game.get("venue")
            city_state = venue_location.get(game.get("venueId"))
            if venue_name and city_state:
                location = f"{venue_name} — {city_state}"
            else:
                location = venue_name or city_state

            predictions.append({
                "home": home,
                "away": away,
                "predicted_winner": winner,
                "margin": round(abs(margin), 2),
                "prob": round(prob * 100, 1) if margin > 0 else round((1 - prob) * 100, 1),
                "betting_spread": betting_spread,
                "edge_class": edge_class,
                "spread_diff": spread_diff,
                "neutral_site": is_neutral,
                "date": game_date.strftime("%Y-%m-%d") if pd.notna(game_date) else None,
                "game_date_display": game_date_et.strftime("%a, %b %-d") if game_date_et is not None else None,
                "game_time_display": ("TBD" if is_tbd else game_date_et.strftime("%-I:%M %p ET")) if game_date_et is not None else None,
                "kickoff_iso": game_date_et.isoformat() if game_date_et is not None else None,
                "location": location,
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    return pd.DataFrame(predictions)



# How to print if I wasn't using the app
#m, p = predict_game("Florida State", "Ohio State")
#print(f"\nPredicted margin {m:.2f}, win probability {p*100:.1f}%")
