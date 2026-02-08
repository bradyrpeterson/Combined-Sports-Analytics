# app.py - COMPLETELY FIXED VERSION
# Unified Flask app for Football and Basketball predictors

from flask import Flask, render_template, request, url_for, session, redirect, jsonify
import sys
import json
import os
import pandas as pd
import numpy as np
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime

# Add both sport folders to Python path
sys.path.append('./football')
sys.path.append('./basketball')

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY',"asdfaDFdf23423@#@!$!@#@!$#@!$#@!$#@!")

# Initialize Firebase Admin
if not firebase_admin._apps:
    if os.path.exists('firebase_credentials.json'):
        cred = credentials.Certificate('firebase_credentials.json')
    else:
        service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
        cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)

# Initialize Firestore DB
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
                from datetime import timezone
                now = datetime.now(timezone.utc)
                
                if isinstance(expires, datetime):
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    
                    if now > expires:
                        user_ref.update({"status": "expired"})
                        return redirect(url_for("payment_pending"))
                
                elif hasattr(expires, 'seconds'):
                    expire_datetime = datetime.fromtimestamp(expires.seconds, tz=timezone.utc)
                    if now > expire_datetime:
                        user_ref.update({"status": "expired"})
                        return redirect(url_for("payment_pending"))
            
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
    return render_template("login.html")

@app.route("/auth-callback", methods=["POST"])
def auth_callback():
    """Handle authentication callback"""
    data = request.get_json()
    uid = data.get("uid")
    email = data.get("email")

    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    if not user_doc.exists:
        user_ref.set({
            "email": email,
            "status": "pending",
            "created_at": datetime.now(),
            "approved_at": None,
            "subscription_expires": None
        })
    
    session["user_id"] = uid
    session["email"] = email
    
    return jsonify({'success': True}), 200

@app.route("/logout")
def logout():
    """Logout user"""
    session.clear()
    return render_template("logout.html")

# Try to import football predictor
try:
    from football import predictor as football_predictor
    
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
    football_conferences = []

# Try to import basketball predictor
try:
    from basketball import predictor as basketball_predictor
    
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
def index():
    """Landing page - shows featured pick and top 5 rankings preview (PUBLIC)"""
    # Check if user is logged in
    user_logged_in = 'user_id' in session
    user_email = session.get('email', None)
    
    try:
        # Get today's basketball games
        basketball_preds = basketball_predictor.get_upcoming_predictions()
        
        # Get this week's football games  
        football_preds = football_predictor.get_upcoming_predictions()
        
        # Find the best pick (highest edge) from either sport
        featured_pick = None
        
        # Check basketball for high-edge games
        if len(basketball_preds) > 0:
            basketball_preds['abs_spread_diff'] = basketball_preds['spread_diff'].abs()
            best_bball = basketball_preds.nlargest(1, 'abs_spread_diff')
            if len(best_bball) > 0:
                pick = best_bball.iloc[0]
                if pick.get('spread_diff') is not None and not pd.isna(pick['spread_diff']) and abs(float(pick['spread_diff'])) >= 5:
                    # Check if prob is already a percentage (>1) or decimal (0-1)
                    prob_value = float(pick['prob'])
                    if prob_value <= 1:
                        prob_value = prob_value * 100  # Convert from decimal to percentage
                    
                    # Handle betting_spread NaN
                    betting_spread_value = None if pd.isna(pick.get('betting_spread')) else pick.get('betting_spread')
                    
                    featured_pick = {
                        'sport': 'basketball',
                        'home': pick['home'],
                        'away': pick['away'],
                        'predicted_winner': pick['predicted_winner'],
                        'margin': round(float(pick['margin']), 1),
                        'prob': round(prob_value, 1),
                        'betting_spread': betting_spread_value,
                        'edge': round(abs(float(pick['spread_diff'])), 1),
                        'edge_class': pick.get('edge_class', ''),
                        'neutral_site': pick.get('neutral_site', False)
                    }
        
        # Check football if no basketball pick
        if not featured_pick and len(football_preds) > 0:
            football_preds['abs_spread_diff'] = football_preds['spread_diff'].abs()
            best_football = football_preds.nlargest(1, 'abs_spread_diff')
            if len(best_football) > 0:
                pick = best_football.iloc[0]
                if pick.get('spread_diff') is not None and not pd.isna(pick['spread_diff']) and abs(float(pick['spread_diff'])) >= 3:
                    # Check if prob is already a percentage (>1) or decimal (0-1)
                    prob_value = float(pick['prob'])
                    if prob_value <= 1:
                        prob_value = prob_value * 100  # Convert from decimal to percentage
                    
                    # Handle betting_spread NaN
                    betting_spread_value = None if pd.isna(pick.get('betting_spread')) else pick.get('betting_spread')
                    
                    featured_pick = {
                        'sport': 'football',
                        'home': pick['home'],
                        'away': pick['away'],
                        'predicted_winner': pick['predicted_winner'],
                        'margin': round(float(pick['margin']), 1),
                        'prob': round(prob_value, 1),
                        'betting_spread': betting_spread_value,
                        'edge': round(abs(float(pick['spread_diff'])), 1),
                        'edge_class': pick.get('edge_class', ''),
                        'neutral_site': pick.get('neutral_site', False)
                    }
        
        if not featured_pick:
            print("No featured pick found (no games with sufficient edge)")
        
        # Get top 5 rankings for preview
        football_top5 = football_predictor.FBS_rankings.head(5).to_dict('records')
        basketball_top5 = basketball_predictor.D1_rankings.head(5).to_dict('records')
        
        return render_template('index.html', 
                             featured_pick=featured_pick,
                             football_top5=football_top5,
                             basketball_top5=basketball_top5,
                             has_games=len(basketball_preds) > 0 or len(football_preds) > 0,
                             user_logged_in=user_logged_in,
                             user_email=user_email)
    except Exception as e:
        print(f"Error loading index: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', 
                             featured_pick=None, 
                             football_top5=[],
                             basketball_top5=[],
                             has_games=False,
                             user_logged_in=user_logged_in,
                             user_email=user_email)

@app.route("/payment-pending")
def payment_pending():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('payment_pending.html', email=session.get('email'))

@app.route("/football")
@login_required
def football():
    """Football predictions page (LOGIN REQUIRED)"""
    try:
        week = request.args.get("week", str(football_predictor.next_week))
        conference = request.args.get("conference", "All")
        
        # Convert week to int for predictor
        week_param = None
        if week != "All" and week != "bowl":
            try:
                week_param = int(week)
            except:
                week_param = football_predictor.next_week
        elif week == "bowl":
            week_param = "bowl"
        
        predictions_df = football_predictor.get_upcoming_predictions(
            week=week_param,
            conference=conference if conference != "All" else None
        )
        
        # COMPLETE FIX: Clean up ALL NaN values
        if len(predictions_df) > 0:
            # Check first row to see if prob is decimal or percentage
            sample_prob = predictions_df['prob'].iloc[0]
            if sample_prob <= 1:
                predictions_df['prob'] = predictions_df['prob'] * 100  # Convert to percentage
            
            predictions_df['margin'] = predictions_df['margin'].round(1)
            
            # Replace NaN betting_spread with None
            predictions_df['betting_spread'] = predictions_df['betting_spread'].apply(
                lambda x: None if pd.isna(x) else x
            )
            
            # Replace NaN spread_diff (edge) with None
            predictions_df['spread_diff'] = predictions_df['spread_diff'].apply(
                lambda x: None if pd.isna(x) else x
            )
        
        predictions = predictions_df.to_dict('records') if len(predictions_df) > 0 else []
        
        # Get list of conferences
        try:
            conferences = sorted(set(
                list(football_predictor.completed['homeConference'].unique()) +
                list(football_predictor.completed['awayConference'].unique())
            ))
        except:
            conferences = football_conferences if football_conferences else []
        
        return render_template('football.html',
                             predictions=predictions,
                             selected_week=week,
                             conferences=conferences,
                             selected_conference=conference)
    except Exception as e:
        print(f"Error loading football: {e}")
        import traceback
        traceback.print_exc()
        return render_template('football.html', 
                             predictions=[], 
                             conferences=football_conferences,
                             selected_week=str(football_predictor.next_week) if FOOTBALL_AVAILABLE else "1",
                             selected_conference="All")

@app.route("/basketball")
@login_required
def basketball():
    """Basketball predictions page (LOGIN REQUIRED)"""
    try:
        conference = request.args.get("conference", "All")
        
        predictions_df = basketball_predictor.get_upcoming_predictions(
            conference=conference if conference != "All" else None
        )
        
        #Clean up ALL NaN values
        if len(predictions_df) > 0:
            # Check first row to see if prob is decimal or percentage
            sample_prob = predictions_df['prob'].iloc[0]
            if sample_prob <= 1:
                predictions_df['prob'] = predictions_df['prob'] * 100  # Convert to percentage
            
            predictions_df['margin'] = predictions_df['margin'].round(1)
            
            # Replace NaN betting_spread with None
            predictions_df['betting_spread'] = predictions_df['betting_spread'].apply(
                lambda x: None if pd.isna(x) else x
            )
            
            # Replace NaN spread_diff (edge) with None
            predictions_df['spread_diff'] = predictions_df['spread_diff'].apply(
                lambda x: None if pd.isna(x) else x
            )
        
        predictions = predictions_df.to_dict('records') if len(predictions_df) > 0 else []
        
        # Get list of conferences
        try:
            conferences = sorted(set(
                list(basketball_predictor.completed['homeConference'].unique()) +
                list(basketball_predictor.completed['awayConference'].unique())
            ))
        except:
            conferences = basketball_conferences if basketball_conferences else []
        
        return render_template('basketball.html',
                             predictions=predictions,
                             conferences=conferences,
                             selected_conference=conference)
    except Exception as e:
        print(f"Error loading basketball: {e}")
        import traceback
        traceback.print_exc()
        return render_template('basketball.html', 
                             predictions=[], 
                             conferences=basketball_conferences,
                             selected_conference="All")

@app.route("/rankings")
@login_required
def rankings():
    """Rankings page (LOGIN REQUIRED)"""
    try:
        football_rankings = football_predictor.FBS_rankings.head(25).to_dict('records')
        basketball_rankings = basketball_predictor.D1_rankings.head(25).to_dict('records')
        
        return render_template('rankings.html',
                             football_rankings=football_rankings,
                             basketball_rankings=basketball_rankings)
    except Exception as e:
        print(f"Error loading rankings: {e}")
        return render_template('rankings.html', 
                             football_rankings=[], 
                             basketball_rankings=[])
    
@app.route("/football/custom")
@login_required
def football_custom():
    """Custom football matchup predictor (LOGIN REQUIRED)"""
    try:
        teams = sorted(football_predictor.ratings.index.tolist())
        return render_template('football_custom.html', teams=teams)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('football_custom.html', teams=[])

@app.route("/basketball/custom")
@login_required
def basketball_custom():
    """Custom basketball matchup predictor (LOGIN REQUIRED)"""
    try:
        teams = sorted(basketball_predictor.ratings.index.tolist())
        return render_template('basketball_custom.html', teams=teams)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('basketball_custom.html', teams=[])

@app.route("/api/predict/football", methods=["POST"])
@login_required
def predict_football():
    """API endpoint for custom football predictions"""
    try:
        data = request.json
        home = data.get("home")
        away = data.get("away")
        neutral = data.get("neutral", False)
        
        if not home or not away:
            return jsonify({"success": False, "error": "Missing teams"}), 400
        
        margin, prob = football_predictor.predict_game(home, away, neutral_site=neutral)
        winner = home if margin > 0 else away
        
        # Ensure prob is percentage
        if prob <= 1:
            prob = prob * 100
        
        #Flip probability if away team won (prob is always for home team)
        if winner == away:
            prob = 100 - prob
        
        return jsonify({
            "success": True,
            "winner": winner,
            "margin": round(abs(margin), 2),
            "probability": round(prob, 1)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/predict/basketball", methods=["POST"])
@login_required
def predict_basketball():
    try:
        data = request.json
        home = data.get("home")
        away = data.get("away")
        neutral = data.get("neutral", False)
        
        if not home or not away:
            return jsonify({"success": False, "error": "Missing teams"}), 400
        
        margin, prob = basketball_predictor.predict_game(home, away, neutral_site=neutral)
        winner = home if margin > 0 else away
        
        # Ensure prob is percentage
        if prob <= 1:
            prob = prob * 100
        
        # Flip probability if away team won
        if winner == away:
            prob = 100 - prob
        
        return jsonify({
            "success": True,
            "winner": winner,
            "margin": round(abs(margin), 2),
            "probability": round(prob, 1)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/terms")
def terms():
    user_logged_in = 'user_id' in session
    user_email = session.get('email', None)
    return render_template("terms.html",
                           user_logged_in=user_logged_in,
                           user_email=user_email)
if __name__ == "__main__":
    print("=" * 80)
    print("SPORTS ANALYTICS HUB - COMPLETELY FIXED VERSION")
    print("=" * 80)
    print(f"Football Predictor: {'✓ Available' if FOOTBALL_AVAILABLE else '✗ Not Available'}")
    print(f"Basketball Predictor: {'✓ Available' if BASKETBALL_AVAILABLE else '✗ Not Available'}")
    print("=" * 80)
    print("Starting server")
    print("=" * 80)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)