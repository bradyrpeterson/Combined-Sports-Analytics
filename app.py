# app.py
# Unified Flask app for Football and Basketball predictors

from flask import Flask, render_template, request, url_for, session, redirect, jsonify
import sys
import json
import os
import pandas as pd
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime
# Add both sport folders to Python path
sys.path.append('./football')
sys.path.append('./basketball')

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY',"asdfaDFdf23423@#@!$!@#@!$#@!$#@!$#@!")

#Initialize Firebase Admin
if not firebase_admin._apps:
    if os.path.exists('firebase_credentials.json'):
        cred = credentials.Certificate('firebase_credentials.json')
    else:
        service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
        cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
#Initialize my firestore DB
db = firestore.client()
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if 'user_id' not in session:
                return redirect('/login')
            
            # Check user status
            user_ref = db.collection("users").document(session["user_id"])
            user_doc = user_ref.get()

            if not user_doc.exists:
                return redirect(url_for('login'))
            
            user_data = user_doc.to_dict()
            status = user_data.get("status", "pending")
            expires = user_data.get("subscription_expires")
            
            # Check if active subscription has expired
            if status == "active" and expires:
                # Make datetime timezone-aware for comparison
                from datetime import timezone
                now = datetime.now(timezone.utc)
                
                # Handle different datetime types
                if isinstance(expires, datetime):
                    # If expires doesn't have timezone, make it UTC
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    
                    if now > expires:
                        user_ref.update({"status": "expired"})
                        return redirect(url_for("payment_pending"))
                
                # Handle Firestore timestamp
                elif hasattr(expires, 'seconds'):
                    expire_datetime = datetime.fromtimestamp(expires.seconds, tz=timezone.utc)
                    if now > expire_datetime:
                        user_ref.update({"status": "expired"})
                        return redirect(url_for("payment_pending"))
            
            # Check if subscription is pending or expired
            if status in ["pending", "expired"]:
                return redirect(url_for("payment_pending"))
            
            return f(*args, **kwargs)
            
        except Exception as e:
            print(f"Error in login_required: {e}")
            import traceback
            traceback.print_exc()
            return redirect(url_for('login'))
    return decorated_function


@app.route("/login")
def login():
    """Login page"""
    doc = db.collection('season_records').document('current').get()
    records = doc.to_dict()

    return render_template("login.html", records=records)

VALID_USERS = {
    "brady": "password123",
    "test": "test123"
}

@app.route("/simple-login", methods=["POST"])
def simple_login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    if username in VALID_USERS and VALID_USERS[username] == password:
        session['user'] = {'username': username, 'email': f'{username}@local.test'}
        return redirect('/')
    return redirect('/login')

@app.route("/auth-callback",methods=["POST"])
def auth_callback():
    """Handle authentication callback"""
    data = request.get_json()
    uid = data.get("uid")
    email = data.get("email")

    #Check if the user exists in my database
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    if not user_doc.exists:
        user_ref.set({
            "email": email,
            "status":"pending",
            "created_at": datetime.now(),
            "approved_at":None,
            "subscription_expires":None
        })
    
    session["user_id"] = uid
    session["email"] = email
    
    return jsonify({'success':True}),200
    

@app.route("/logout")
def logout():
    """Logout user"""
    session.clear()
    return render_template("logout.html")

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
@login_required
def landing():
    """Landing page - choose your sport"""
    if 'user_id' not in session:
        return redirect('/login')
    return render_template(
        "landing.html",
        football_available=FOOTBALL_AVAILABLE,
        basketball_available=BASKETBALL_AVAILABLE
    )
@app.route("/payment-pending")
def payment_pending():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('payment_pending.html', email=session.get('email'))

@app.route("/football", methods=["GET", "POST"])
@login_required
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
    neutral_site = False
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
        neutral_site = request.form.get("neutral_site") == "on"
        
        try:
            margin, prob = football_predictor.predict_game(home_team, away_team, neutral_site=neutral_site)
            winner = home_team if margin > 0 else away_team
            winner_prob = prob if margin > 0 else 1 - prob
            
            #Add neutral site info for the result
            site_info = "(neutral site)" if neutral_site else ""
            result = f"{winner} has a {winner_prob*100:.2f}% chance to win and is predicted to win by {abs(margin):.2f}{site_info}"
            
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
        selected_conference=selected_conference,
        neutral_site=neutral_site
    )


@app.route("/basketball", methods=["GET", "POST"])
@login_required
def basketball():
    # Basketball predictor page
    if not BASKETBALL_AVAILABLE:
        return "Basketball predictor not available. Check basketball/predictor.py", 404
    
    result = None
    home_team = None
    away_team = None
    upcoming_predictions = None
    neutral_site = False
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
        neutral_site = request.form.get("neutral_site") == "on"
        
        # Check if teams have stats
        if home_team not in basketball_predictor.ratings.index or away_team not in basketball_predictor.ratings.index:
            result = f"Stats not available for matchup: {home_team} vs {away_team}"
        else:
            try:
                margin, prob = basketball_predictor.predict_game(home_team, away_team, neutral_site=neutral_site)
                winner = home_team if margin > 0 else away_team
                winner_prob = prob if margin > 0 else 1 - prob
                
                site_info = "(neutral site)" if neutral_site else ""
                result = f"{winner} has a {winner_prob*100:.2f}% chance to win and is predicted to win by {abs(margin):.2f} points.{site_info}"
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
        selected_conference=selected_conference,
        neutral_site=neutral_site
    )

#Create flask route for football rankings
@app.route("/football/rankings")
@login_required
def football_rankings():
    if not FOOTBALL_AVAILABLE:
        return "Football predictor not available", 404
    
    # Access the FBS_rankings you already created in predictor.py
    rankings_df = football_predictor.FBS_rankings.copy()
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    rankings_df['rating'] = rankings_df['rating'].round(2)
    
    return render_template(
        "rankings.html",
        sport="Football",
        rankings=rankings_df,
        back_url=url_for('football')
    )

#Create flask route for basketball rankings
@app.route("/basketball/rankings")
@login_required
def basketball_rankings():
    if not BASKETBALL_AVAILABLE:
        return "Basketball predictor not available", 404
    
    # Access the D1_rankings you already created in predictor.py
    rankings_df = basketball_predictor.D1_rankings.copy()
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    rankings_df['rating'] = rankings_df['rating'].round(2)
    
    return render_template(
        "rankings.html",
        sport="Basketball",
        rankings=rankings_df,
        back_url=url_for('basketball')
    )
if __name__ == "__main__":
    print("=" * 80)
    print("SPORTS ANALYTICS HUB")
    print("=" * 80)
    print(f"Football Predictor: {'✓ Available' if FOOTBALL_AVAILABLE else '✗ Not Available'}")
    print(f"Basketball Predictor: {'✓ Available' if BASKETBALL_AVAILABLE else '✗ Not Available'}")
    print("=" * 80)
    print("Starting server")
    print("=" * 80)
    
    #Get port from environment variable (for deployment) or use 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)