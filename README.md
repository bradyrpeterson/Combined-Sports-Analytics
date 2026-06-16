# 🏆 College Sports Analytics Platform

A machine learning web application that predicts outcomes for college football and basketball games using linear regression and advanced performance metrics.

**Live Link:** petersonpredicts.com

---

## 📊 Overview

This project combines two sports prediction models into a unified platform. Both models use linear regression to calculate team power ratings from game results, then incorporate efficiency statistics to forecast outcomes and win probabilities.

### Features

**Football Predictor:**
- FBS team power ratings based on game results
- Weekly predictions for all upcoming games
- DraftKings betting line comparison to identify value opportunities
- Efficiency metrics: yards per play, third down conversion, turnover margin

**Basketball Predictor:**
- D1 team ratings updated throughout the season
- Daily game predictions (EST timezone)
- Offensive and defensive efficiency analysis
- Win probability calculations

---

## 🚀 Live Application

Visit the app: **https://statalysts.com/**

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn (Linear Regression)
- **Data Processing:** Pandas, NumPy
- **APIs:** CollegeFootballData API, CollegeBasketballData API
- **Deployment:** Render
- **Frontend:** HTML, CSS, JavaScript

---

## 📈 How It Works

### The Model

Both predictors use the same core approach:

1. **Build Design Matrix**: Each completed game creates a row with +1 for the home team and -1 for the away team
2. **Fit Linear Regression**: Find team ratings that minimize prediction error across all games
3. **Calculate Margin**: `predicted_margin = home_rating - away_rating + home_advantage`
4. **Add Efficiency Stats**: Incorporate offensive/defensive metrics and turnover data
5. **Convert to Probability**: Use a sigmoid function to calculate win probability


## 💻 Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sports-analytics.git
cd sports-analytics

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "API_KEY=your_cfbd_api_key" > .env

# Run the application
python app.py
```

Visit `http://localhost:5000` in your browser.

### Get API Keys
- Football: [CollegeFootballData.com](https://collegefootballdata.com/key)
- Basketball: [CollegeBasketballData.com](https://api.collegebasketballdata.com)

---

## 📝 Data Sources

- **Game Results & Stats:** [CollegeFootballData API](https://collegefootballdata.com)
- **Basketball Data:** [CollegeBasketballData API](https://api.collegebasketballdata.com)
- **Betting Lines:** DraftKings (via CFBD API)

---

## 👤 Author

**Brady Peterson**  
[GitHub](https://github.com/bradyrpeterson) | [LinkedIn](https://www.linkedin.com/in/brady-peterson-b5ab02308/)

---
