##Import needed packages
import pandas as pd
import numpy as np
import requests
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
#Load d1 teams
with open("basketball/d1_teams_2025.json", "r") as f:
    d1_teams = json.load(f)
#Have to use requests since no python library for CBBD yet
games_url = "https://api.collegebasketballdata.com/games?season=2026"
headers = {"Authorization": f"Bearer {api_key}"} 
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
headers = {"Authorization": f"Bearer {api_key}"} 
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
#In case I need to see what columns the game dataset has to offer
#print("Columns available:", df.columns.tolist())


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
#The regressions finds coefficients that best fit the margins
#Don't worry about an intercept cause we have home field
y = df["margin"]

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
home_court = coefs["home_court"]
ratings = coefs.drop("home_court")
#Make the average team=0
ratings -= ratings.mean()   

# Sort ratings index for clean dropdown in Flask
ratings = ratings.sort_index()



# In case I want to print the home field advantage calculation
#print("Estimated home-field advantage (points):", round(home_field, 2))
#In case I went to print the best teams strictly based on my powerindex
#print(ratings.sort_values(ascending=False).head(5))

#Prediciton function
def predict_game(home, away):

    rating_diff = ratings[home] - ratings[away]
    #Pull the home and away team stats and compare them
    h_stats = stats_clean.loc[stats_clean["team"] == home].iloc[0]
    a_stats = stats_clean.loc[stats_clean["team"] == away].iloc[0]
    
    # Compute stat differences between the two teams
    oeff_diff = h_stats["off_eff"] - a_stats["off_eff"]
    deff_diff = h_stats["def_eff"] - a_stats["def_eff"]
    turnover_diff = h_stats["tov_rate"] - a_stats["tov_rate"]

    #Different weights of each
    w_rating=0.7
    w_oeff=0.125
    w_deff=0.125
    w_tov=0.05
    margin=(w_rating*rating_diff+(w_oeff*oeff_diff)+(w_deff*deff_diff)+(w_tov*turnover_diff)+home_court)
    #margin = (w_rating * rating_diff)+home_court
    #Calculate probabiliy based on the idea that a team favored by 7 
    #has a 75% chance to win 
    prob = 1 / (1 + np.exp(-margin / 5))  # rough logistic
    return margin, prob

from datetime import datetime, timezone
def get_upcoming_predictions():
    # Use the upcoming games dataset (no scores yet)

    #Only care about games that are happening today in EST
    games_to_predict = upcoming.copy()
    games_to_predict["startDate"] = pd.to_datetime(
        games_to_predict["startDate"], utc=True, errors="coerce"
    )

    # Convert UTC → EST
    games_to_predict["startDate_EST"] = games_to_predict["startDate"].dt.tz_convert("America/New_York")

    # Get today's date in EST
    today_est = datetime.now(tz=pd.Timestamp.now("America/New_York").tzinfo).date()

    # Filter only today's games
    games_to_predict = games_to_predict[
        games_to_predict["startDate_EST"].dt.date == today_est
    ]


    predictions = []
    for _, game in games_to_predict.iterrows():
        home, away = game["homeTeam"], game["awayTeam"]

        #skip games where data
        #Helps avoid faulty data by skipping the games or stats are missing
        if home not in ratings.index or away not in ratings.index:
            continue
        if home not in stats_clean["team"].values or away not in stats_clean["team"].values:
            continue

       
        try:
            margin, prob = predict_game(home, away)
            winner = home if margin > 0 else away
            predictions.append({
                "home": home,
                "away": away,
                "predicted_winner": winner,
                "margin": round(abs(margin), 2),
                "prob": round(prob * 100, 1) if margin > 0 else round((1 - prob) * 100, 1)
            })
        except Exception as e:
            print(f"Error predicting {home} vs {away}: {e}")
            continue

    return predictions



#D1_CONFERENCES = [
 #   "ACC", "Big 12", "Big Ten", "SEC", "Big East",
  #  "A-10", "American", "Mountain West", "WCC", "CUSA", "Horizon", "Ivy", "MAAC", "MAC", "MEAC",
  #  "MVC", "NEC", "OVC", "Patriot", "SoCon",
  #  "Southland", "Summit", "Sun Belt", "SWAC", "WAC", 
  #  "Big Sky", "Big South", "Am. East", "ASUN", "Big West","CAA"
#]

