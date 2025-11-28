##Import needed packages
import pandas as pd
import numpy as np
import requests
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta

load_dotenv()
api_key = os.getenv("API_KEY")

# Headers for API requests - EXACTLY LIKE FOOTBALL
headers = {"Authorization": f"Bearer {api_key}"}

#Load d1 teams
with open("basketball/d1_teams_2025.json", "r") as f:
    d1_teams = json.load(f)

#Have to use requests since no python library for CBBD yet
games_url = "https://api.collegebasketballdata.com/games?season=2026"
#Convert the API response into a json then a dataframe for easy use
games_response = requests.get(games_url, headers=headers)
games_data = games_response.json()
games_df = pd.DataFrame(games_data)
needed_cols = ["season","status","startDate","homeTeam","awayTeam","homePoints","awayPoints","homeConference","awayConference"]
games_df=games_df[needed_cols].copy()

#Only care about games where one of the teams was D1
games_df=games_df[games_df["homeTeam"].isin(d1_teams) | games_df["awayTeam"].isin(d1_teams)].reset_index(drop=True)

#Completed games = final status
completed = games_df[(games_df["status"] == "final")].reset_index(drop=True)

#Upcoming games = scheduled but not yet played
upcoming = games_df[(games_df["status"] != "final")].reset_index(drop=True)

#Create the margin column in completed
completed["margin"] = completed["homePoints"] - completed["awayPoints"]

need_cols = ["season","startDate","status","homeTeam","awayTeam","homePoints","awayPoints","homeConference","awayConference"]
games_df=games_df[need_cols].copy()

#Using requests pull all the statistical data from the data set
stats_url = "https://api.collegebasketballdata.com/stats/team/season?season=2026"
#Convert the API response into a json then a dataframe for easy use
stats_response = requests.get(stats_url, headers=headers)
stats_data = stats_response.json()
stats_df = pd.DataFrame(stats_data)
team_stats = pd.json_normalize(stats_df["teamStats"])
team_stats.columns = ["teamStats_" + c.replace(".", "_") for c in team_stats.columns]

opp_stats = pd.json_normalize(stats_df["opponentStats"])
opp_stats.columns = ["opponentStats_" + c.replace(".", "_") for c in opp_stats.columns]

#Combine the stats back into one dataframe
stats_df = pd.concat(
    [stats_df.drop(["teamStats", "opponentStats"], axis=1),
     team_stats, opp_stats],
    axis=1
)

# Create efficiency stats using the dataset
stats_df["off_eff"] = stats_df["teamStats_points_total"] / stats_df["teamStats_possessions"]
stats_df["def_eff"] = stats_df["opponentStats_points_total"] / stats_df["opponentStats_possessions"]
stats_df["tov_rate"] = stats_df["teamStats_fourFactors_turnoverRatio"]

# Keep only the stats/columns that I plan on using
useful = ["team", "off_eff", "def_eff", "tov_rate"]
stats_clean = stats_df[useful].copy()

#define what margin is
#sort the dataframe to have a line of home and away teams
#Build the design matrix
df = completed.dropna(subset=["homeTeam", "awayTeam", "homePoints", "awayPoints"]).copy()
df = df.dropna(subset=["margin"]).reset_index(drop=True)

teams = sorted(set(df["homeTeam"]).union(df["awayTeam"]))
#Home teams get a +1 value and -1 is for away
#This setup allows for the regression to assign each team a numeric rating
X = pd.DataFrame(0, index=np.arange(len(df)), columns=teams)
for i, row in df.iterrows():
    X.loc[i, row["homeTeam"]] = 1    # +1 for home team
    X.loc[i, row["awayTeam"]] = -1   # -1 for away team

#Add home court column
X["home_court"] = 1

X = X.fillna(0)

#Time to fit the margin
y = df["margin"]

#Create the linear regression based on the margins
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

#Ensure home field is counted for in the team ratings
coefs = pd.Series(model.coef_, index=X.columns)
home_court = coefs["home_court"]
ratings = coefs.drop("home_court")
#Make the average team=0
ratings -= ratings.mean()   

# Sort ratings index for clean dropdown in Flask
ratings = ratings.sort_index()


# Create D1-only rankings
D1_rankings = pd.DataFrame({
    'team': ratings.index,
    'rating': ratings.values
})

# Filter for only FBS teams
D1_rankings = D1_rankings[D1_rankings['team'].isin(d1_teams)].reset_index(drop=True)

# Sort by rating descending
D1_rankings = D1_rankings.sort_values(by='rating', ascending=False).reset_index(drop=True)
#Prediction function - EXACTLY LIKE FOOTBALL
def predict_game(home, away, neutral_site=False):
    rating_diff = ratings[home] - ratings[away]
    #Pull the home and away team stats and compare them
    h_stats = stats_clean.loc[stats_clean["team"] == home].iloc[0]
    a_stats = stats_clean.loc[stats_clean["team"] == away].iloc[0]
    
    # Compute stat differences between the two teams
    oeff_diff = h_stats["off_eff"] - a_stats["off_eff"]
    deff_diff = h_stats["def_eff"] - a_stats["def_eff"]
    turnover_diff = h_stats["tov_rate"] - a_stats["tov_rate"]

    #Whether or not home court advntage is applied
    home_advantage = 0 if neutral_site else home_court
    #Different weights of each
    w_rating=0.7
    w_oeff=0.125
    w_deff=0.125
    w_tov=0.05
    margin=(w_rating*rating_diff+(w_oeff*oeff_diff)+(w_deff*deff_diff)+(w_tov*turnover_diff)+home_advantage)

    #Calculate probability
    prob = 1 / (1 + np.exp(-margin / 5))  # rough logistic
    return margin, prob

#Function to get betting lines for today's games
def get_betting_lines(season=2026):
    

    # Get today's date in EST
    import pytz
    est = pytz.timezone('America/New_York')
    today_est = datetime.now(est)
    
    # Start of day in EST, then convert to UTC for API
    start_of_day_est = today_est.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day_est = today_est.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Convert EST to UTC (API expects UTC with Z)
    start_of_day_utc = start_of_day_est.astimezone(pytz.UTC)
    end_of_day_utc = end_of_day_est.astimezone(pytz.UTC)
    
    # Format as ISO 8601 with Z
    start_date = start_of_day_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    end_date = end_of_day_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    # Basketball lines endpoint
    lines_url = f"https://api.collegebasketballdata.com/lines?season={season}&startDateRange={start_date}&endDateRange={end_date}"
    
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
            
            # Look for ESPN BET spreads
            espn_spread = None
            for line in lines:
                if line.get("provider") == "ESPN BET":
                    espn_spread = line.get("spread")
                    break
            
            if espn_spread is not None:
                betting_lines[(home, away)] = espn_spread
        
        return betting_lines
    
    except Exception as e:
        print(f"Error fetching betting lines: {e}")
        return {}

# EXACTLY LIKE FOOTBALL
def calculate_edge_highlight(model_margin, betting_spread):
    # Convert betting spread to match our model's convention
    # If betting spread is +7, that means home is favored by 7
    # If betting spread is -7, that means away is favored by 7
    # We need to flip it to match our model's convention (positive = home favored)
    betting_margin = -betting_spread
    
    # Calculate the difference
    difference = abs(model_margin - betting_margin)
    
    if difference >= 10:
        return 'edge-big'
    elif difference >= 5:
        return 'edge-medium'
    else:
        return None

#Get upcoming games in order to print them
def get_upcoming_predictions(conference=None):
    # Use the upcoming games dataset (no scores yet)
    games_to_predict = upcoming.copy()
    
    # Only care about games that are happening today in EST
    games_to_predict["startDate"] = pd.to_datetime(
        games_to_predict["startDate"], utc=True, errors="coerce"
    )

    # Convert UTC → EST
    games_to_predict["startDate_EST"] = games_to_predict["startDate"].dt.tz_convert("America/New_York")

    # Get today's date in EST
    import pytz
    est = pytz.timezone('America/New_York')
    today_est = datetime.now(est).date()

    # Filter only today's games
    games_to_predict = games_to_predict[
        games_to_predict["startDate_EST"].dt.date == today_est
    ]

    # Filter by conferences
    if conference is not None and conference != "All":
        games_to_predict = games_to_predict[
            (games_to_predict["homeConference"] == conference) | 
            (games_to_predict["awayConference"] == conference)
        ]
    
    # Fetch betting lines for today from ESPN BET
    betting_lines = get_betting_lines(season=2026)

    predictions = []
    for _, game in games_to_predict.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]

        # Skip games where data is missing
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_clean["team"].values or away not in stats_clean["team"].values:
            continue

        try:
            margin, prob = predict_game(home, away)
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
            
            predictions.append({
                "home": home,
                "away": away,
                "predicted_winner": winner,
                "margin": round(abs(margin), 2),
                "prob": round(prob * 100, 1) if margin > 0 else round((1 - prob) * 100, 1),
                "betting_spread": betting_spread,
                "edge_class": edge_class,
                "spread_diff": spread_diff
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    # Return DataFrame - EXACTLY LIKE FOOTBALL
    return pd.DataFrame(predictions)