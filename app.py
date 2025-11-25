# app.py
# Unified Flask app for Football and Basketball predictors

from flask import Flask, render_template, request
import sys
import json
import os
import pandas as pd

# Add both sport folders to Python path
sys.path.append('./football')
sys.path.append('./basketball')

app = Flask(__name__)

# Try to import football predictor
try:
    from football import predictor as football_predictor
    
    # Load football data files
    football_dir = os.path.join(os.path.dirname(__file__), 'football')
    with open(os.path.join(football_dir, "fbs_teams_2025.json"), "r") as f:
        football_teams = json.load(f)
    with open(os.path.join(football_dir, "team_logos.json"), "r") as f:
        football_logos = json.load(f)
    with open(os.path.join(football_dir, "team_color.json"), "r") as f:
        football_colors = json.load(f)
    with open(os.path.join(football_dir, "conferences.json"), "r") as f:
        football_conferences = json.load(f)
    FOOTBALL_AVAILABLE = True
except Exception as e:
    print(f"Football predictor not available: {e}")
    FOOTBALL_AVAILABLE = False
    football_teams = []
    football_logos = {}
    football_colors = {}

# Try to import basketball predictor
try:
    from basketball import predictor as basketball_predictor
    
    # Load basketball data files
    basketball_dir = os.path.join(os.path.dirname(__file__), 'basketball')
    with open(os.path.join(basketball_dir, "d1_teams_2025.json"), "r") as f:
        basketball_teams = json.load(f)
    with open(os.path.join(basketball_dir, "conferences.json"), "r") as f:
        basketball_conferences = json.load(f)
    BASKETBALL_AVAILABLE = True
except Exception as e:
    print(f"Basketball predictor not available: {e}")
    BASKETBALL_AVAILABLE = False
    basketball_teams = []
    basketball_conferences = []


@app.route("/")
def landing():
    """Landing page - choose your sport"""
    return render_template(
        "landing.html",
        football_available=FOOTBALL_AVAILABLE,
        basketball_available=BASKETBALL_AVAILABLE
    )


@app.route("/football", methods=["GET", "POST"])
def football():
   #Football predictor page
    if not FOOTBALL_AVAILABLE:
        return "Football predictor not available. Check football/predictor.py", 404
    
    result = None
    home_team = None
    away_team = None
    home_logo = None
    away_logo = None
    winner_color = None
    upcoming_predictions = None
    selected_conference = request.args.get("conference", "All")
    
    # Get upcoming predictions
    try:
        upcoming_predictions = football_predictor.get_upcoming_predictions(
            week=football_predictor.next_week,
            conference=selected_conference if selected_conference != 'All' else None
        )
    except Exception as e:
        print(f"Error getting football predictions: {e}")
        upcoming_predictions = pd.DataFrame()  # Empty DataFrame instead of None
    
    if request.method == "POST":
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        
        try:
            margin, prob = football_predictor.predict_game(home_team, away_team)
            winner = home_team if margin > 0 else away_team
            winner_prob = prob if margin > 0 else 1 - prob
            
            result = f"{winner} has a {winner_prob*100:.2f}% chance to win and is predicted to win by {abs(margin):.2f}"
            
            winner_color = football_colors.get(winner)
            home_logo = football_logos.get(home_team)
            away_logo = football_logos.get(away_team)
        except Exception as e:
            result = f"Error making prediction: {e}"
    
    return render_template(
        "football.html",
        teams=football_teams,
        result=result,
        home_team=home_team,
        away_team=away_team,
        home_logo=home_logo,
        away_logo=away_logo,
        winner_color=winner_color,
        predictions=upcoming_predictions,
        week=football_predictor.next_week if FOOTBALL_AVAILABLE else None,
        conferences=football_conferences,
        selected_conference=selected_conference
    )


@app.route("/basketball", methods=["GET", "POST"])
def basketball():
    # Basketball predictor page
    if not BASKETBALL_AVAILABLE:
        return "Basketball predictor not available. Check basketball/predictor.py", 404
    
    result = None
    home_team = None
    away_team = None
    upcoming_predictions = pd.DataFrame()  # Initialize as empty DataFrame
    selected_conference = request.args.get("conference", "All")
    
    # Get upcoming predictions (today's games)
    try:
        predictions_result = basketball_predictor.get_upcoming_predictions(
            conference=selected_conference if selected_conference != 'All' else None
        )
        
        # Ensure it's a DataFrame
        if isinstance(predictions_result, pd.DataFrame):
            upcoming_predictions = predictions_result
        elif isinstance(predictions_result, list):
            upcoming_predictions = pd.DataFrame(predictions_result)
        else:
            upcoming_predictions = pd.DataFrame()
            
    except Exception as e:
        print(f"Error getting basketball predictions: {e}")
        import traceback
        traceback.print_exc()
        upcoming_predictions = pd.DataFrame()
    
    if request.method == "POST":
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        
        # Check if teams have stats
        if home_team not in basketball_predictor.ratings.index or away_team not in basketball_predictor.ratings.index:
            result = f"Stats not available for matchup: {home_team} vs {away_team}"
        else:
            try:
                margin, prob = basketball_predictor.predict_game(home_team, away_team)
                winner = home_team if margin > 0 else away_team
                winner_prob = prob if margin > 0 else 1 - prob
                
                result = f"{winner} has a {winner_prob*100:.2f}% chance to win and is predicted to win by {abs(margin):.2f} points."
            except Exception as e:
                result = f"Error making prediction: {e}"
    
    # Sort teams for dropdown
    sorted_teams = sorted(basketball_teams)
    
    return render_template(
        "basketball.html",
        teams=sorted_teams,
        result=result,
        home_team=home_team,
        away_team=away_team,
        predictions=upcoming_predictions,
        conferences=basketball_conferences,
        selected_conference=selected_conference
    )


if __name__ == "__main__":
    print("=" * 80)
    print("🏆 SPORTS ANALYTICS HUB")
    print("=" * 80)
    print(f"Football Predictor: {'✓ Available' if FOOTBALL_AVAILABLE else '✗ Not Available'}")
    print(f"Basketball Predictor: {'✓ Available' if BASKETBALL_AVAILABLE else '✗ Not Available'}")
    print("=" * 80)
    print("🌐 Starting server")
    print("=" * 80)
    
    # Get port from environment variable (for deployment) or use 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)