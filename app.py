import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import sqlite3
from scipy.stats import poisson
import base64

# Page configuration
st.set_page_config(
    page_title="FootballPro Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""

    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .css-12oz5g7 {
        background-color: #262730;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
    .stButton > button {
        background-color: #00ff41;
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #00cc33;
        color: black;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1e1e, #2d3748);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #00ff41;
        margin: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(145deg, #2d3748, #1a202c);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #00ff41;
        margin: 1rem 0;
        text-align: center;
    }
    .subscription-card {
        background: linear-gradient(145deg, #1a202c, #2d3748);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #4a5568;
        margin: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .subscription-card:hover {
        border-color: #00ff41;
        transform: translateY(-5px);
    }
    .popular-plan {
        border: 2px solid #00ff41 !important;
        position: relative;
    }
    .popular-badge {
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: #00ff41;
        color: black;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #00ff41 !important;
    }
    .stDataFrame {
        color: white;
    }

""", unsafe_allow_html=True)

# Database setup
def init_db():
    conn = sqlite3.connect('footballpro.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT, 
                  subscription_tier TEXT DEFAULT 'free', predictions_used INTEGER DEFAULT 0,
                  last_reset DATE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Teams table
    c.execute('''CREATE TABLE IF NOT EXISTS teams
                 (id INTEGER PRIMARY KEY, name TEXT, league TEXT, 
                  goals_scored_1h REAL, goals_conceded_1h REAL, 
                  recent_form_gs REAL, recent_form_gc REAL)''')
    
    # Predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, user_email TEXT, home_team TEXT, away_team TEXT,
                  prediction_type TEXT, prediction_value REAL, confidence REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Enhanced Poisson Model Class
class EnhancedPoissonModel:
    def __init__(self):
        self.form_weight = 0.3  # 30% weight to recent form
        self.home_advantage = 1.15  # 15% boost for home teams
        
    def calculate_lambda(self, team_gs, team_gc, opponent_gs, opponent_gc, 
                        team_form_gs, team_form_gc, is_home=True):
        # Adjust for form (30% recent, 70% season average)
        adj_gs = (team_gs * (1 - self.form_weight)) + (team_form_gs * self.form_weight)
        adj_gc = (team_gc * (1 - self.form_weight)) + (team_form_gc * self.form_weight)
        
        # Calculate expected goals using enhanced formula
        attack_strength = adj_gs
        defense_weakness = opponent_gc
        
        lambda_team = (attack_strength + defense_weakness) / 2
        
        # Apply home advantage
        if is_home:
            lambda_team *= self.home_advantage
            
        return max(0.1, lambda_team)  # Minimum lambda to avoid zero
    
    def poisson_probability(self, lambda_val, k):
        return poisson.pmv(k, lambda_val)
    
    def calculate_match_probabilities(self, home_team_data, away_team_data):
        # Calculate lambdas for both teams
        home_lambda = self.calculate_lambda(
            home_team_data['goals_scored_1h'], home_team_data['goals_conceded_1h'],
            away_team_data['goals_scored_1h'], away_team_data['goals_conceded_1h'],
            home_team_data['recent_form_gs'], home_team_data['recent_form_gc'],
            is_home=True
        )
        
        away_lambda = self.calculate_lambda(
            away_team_data['goals_scored_1h'], away_team_data['goals_conceded_1h'],
            home_team_data['goals_scored_1h'], home_team_data['goals_conceded_1h'],
            away_team_data['recent_form_gs'], away_team_data['recent_form_gc'],
            is_home=False
        )
        
        total_lambda = home_lambda + away_lambda
        
        # Calculate probabilities for different outcomes
        probabilities = {}
        
        # Over/Under probabilities
        probabilities['over_0_5'] = 1 - self.poisson_probability(total_lambda, 0)
        probabilities['over_1_5'] = 1 - (self.poisson_probability(total_lambda, 0) + 
                                       self.poisson_probability(total_lambda, 1))
        probabilities['over_2_5'] = 1 - sum([self.poisson_probability(total_lambda, i) for i in range(3)])
        
        probabilities['under_0_5'] = self.poisson_probability(total_lambda, 0)
        probabilities['under_1_5'] = (self.poisson_probability(total_lambda, 0) + 
                                    self.poisson_probability(total_lambda, 1))
        
        # Exact score probabilities
        exact_scores = {}
        for i in range(6):
            exact_scores[f'{i}_goals'] = self.poisson_probability(total_lambda, i)
        
        return {
            'home_lambda': home_lambda,
            'away_lambda': away_lambda,
            'total_lambda': total_lambda,
            'probabilities': probabilities,
            'exact_scores': exact_scores
        }

# Sample team data
def load_sample_teams():
    conn = sqlite3.connect('footballpro.db')
    
    # Check if teams already exist
    existing = pd.read_sql("SELECT COUNT(*) as count FROM teams", conn).iloc[0]['count']
    
    if existing == 0:
        teams_data = [
            ("Arsenal", "Premier League", 0.95, 0.27, 1.1, 0.2),
            ("Chelsea", "Premier League", 0.86, 0.71, 0.9, 0.8),
            ("Manchester City", "Premier League", 1.2, 0.15, 1.3, 0.1),
            ("Liverpool", "Premier League", 1.1, 0.3, 1.2, 0.25),
            ("Tottenham", "Premier League", 0.86, 0.73, 0.8, 0.9),
            ("Manchester United", "Premier League", 0.73, 0.58, 0.7, 0.6),
            ("Newcastle", "Premier League", 0.65, 0.45, 0.7, 0.4),
            ("Brighton", "Premier League", 0.58, 0.52, 0.6, 0.5),
            ("Fulham", "Premier League", 0.73, 0.53, 0.8, 0.5),
            ("Leicester City", "Premier League", 0.44, 0.94, 0.4, 1.0),
            ("Barcelona", "La Liga", 1.15, 0.25, 1.2, 0.2),
            ("Real Madrid", "La Liga", 1.08, 0.31, 1.1, 0.3),
            ("Atletico Madrid", "La Liga", 0.72, 0.38, 0.7, 0.4),
            ("Sevilla", "La Liga", 0.64, 0.47, 0.6, 0.5),
            ("Valencia", "La Liga", 0.51, 0.68, 0.5, 0.7),
            ("Inter Milan", "Serie A", 0.89, 0.34, 0.9, 0.3),
            ("AC Milan", "Serie A", 0.83, 0.41, 0.8, 0.4),
            ("Juventus", "Serie A", 0.71, 0.48, 0.7, 0.5),
            ("Napoli", "Serie A", 0.95, 0.38, 1.0, 0.35),
            ("Roma", "Serie A", 0.67, 0.55, 0.65, 0.6)
        ]
        
        c = conn.cursor()
        c.executemany("INSERT INTO teams (name, league, goals_scored_1h, goals_conceded_1h, recent_form_gs, recent_form_gc) VALUES (?, ?, ?, ?, ?, ?)", teams_data)
        conn.commit()
    
    teams_df = pd.read_sql("SELECT * FROM teams", conn)
    conn.close()
    return teams_df

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password):
    conn = sqlite3.connect('footballpro.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", 
                 (email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def verify_user(email, password):
    conn = sqlite3.connect('footballpro.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

def get_user_info(email):
    conn = sqlite3.connect('footballpro.db')
    user_df = pd.read_sql("SELECT * FROM users WHERE email = ?", conn, params=(email,))
    conn.close()
    return user_df.iloc[0] if not user_df.empty else None

# PayPal Integration (HTML buttons)
def get_paypal_button(plan_name, price):
    # You'll replace these with your actual PayPal button codes
    buttons = {
        "Daily Tips": f"""
        
            
            
            
            
        
        """,
        "5 Tips Package": f"""
        
            
            
            
            
        
        """,
        "Pro Monthly": f"""
        
            
            
            
            
        
        """,
        "Premium": f"""
        
            
            
            
            
        
        """
    }
    return buttons.get(plan_name, "")

# Main App
def main():
    # Initialize session state
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Load teams data
    teams_df = load_sample_teams()
    model = EnhancedPoissonModel()

    # Sidebar navigation
    st.sidebar.title("⚽ FootballPro Analytics")
    
    if st.session_state.user_email:
        user_info = get_user_info(st.session_state.user_email)
        st.sidebar.success(f"Welcome, {st.session_state.user_email}")
        st.sidebar.info(f"Plan: {user_info['subscription_tier'].title()}")
        
        if st.sidebar.button("Logout"):
            st.session_state.user_email = None
            st.rerun()
            
        page = st.sidebar.selectbox("Navigate", ["Home", "Predictions", "My Account"])
    else:
        page = st.sidebar.selectbox("Navigate", ["Home", "Pricing", "Login", "Register"])

    # Main content
    if page == "Home":
        st.title("⚽ FootballPro Analytics")
        st.subheader("Advanced First-Half Football Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            
                🎯 Enhanced Accuracy
                30% form weighting + 15% home advantage for superior predictions
            
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            
                ⚡ Real-Time Analysis
                Live team statistics and advanced Poisson distribution modeling
            
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            
                💰 Value Betting
                Identify profitable opportunities in first-half markets
            
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Demo prediction
        st.subheader("🔥 Live Demo - Arsenal vs Chelsea")
        
        arsenal = teams_df[teams_df['name'] == 'Arsenal'].iloc[0]
        chelsea = teams_df[teams_df['name'] == 'Chelsea'].iloc[0]
        
        result = model.calculate_match_probabilities(arsenal, chelsea)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            
                Over 0.5 Goals
                {result['probabilities']['over_0_5']:.1%}
                Confidence: High
            
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            
                Under 1.5 Goals
                {result['probabilities']['under_1_5']:.1%}
                Confidence: Medium
            
            """, unsafe_allow_html=True)

    elif page == "Pricing":
        st.title("💰 Subscription Plans")
        
        col1, col2, col3, col4 = st.columns(4)
        
        plans = [
            ("Daily Tips", "€4.99", "5 predictions per day", "Basic analysis", False),
            ("5 Tips Package", "€9.99", "5 premium predictions", "Valid for 7 days", False),
            ("Pro Monthly", "€19.99", "50 predictions per day", "Advanced analysis + alerts", True),
            ("Premium", "€49.99", "Unlimited predictions", "API access + priority support", False)
        ]
        
        for i, (name, price, feature1, feature2, popular) in enumerate(plans):
            with [col1, col2, col3, col4][i]:
                popular_class = "popular-plan" if popular else ""
                st.markdown(f"""
                
                    {f'MOST POPULAR' if popular else ''}
                    {name}
                    {price}
                    {feature1}
                    {feature2}
                
                """, unsafe_allow_html=True)
                
                # PayPal button
                st.components.v1.html(get_paypal_button(name, price), height=100)

    elif page == "Login":
        st.title("🔑 Login")
        
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if verify_user(email, password):
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    elif page == "Register":
        st.title("📝 Create Account")
        
        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if password != confirm_password:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif create_user(email, password):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Email already exists")

    elif page == "Predictions" and st.session_state.user_email:
        st.title("🔮 Match Predictions")
        
        user_info = get_user_info(st.session_state.user_email)
        
        # Subscription limits
        limits = {
            'free': 0,
            'daily': 5,
            'package': 5,
            'pro': 50,
            'premium': -1  # Unlimited
        }
        
        user_limit = limits.get(user_info['subscription_tier'], 0)
        
        if user_limit == 0:
            st.warning("Please subscribe to access predictions!")
            return
        
        st.subheader("Select Teams")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Home Team**")
            home_team = st.selectbox("", teams_df['name'].tolist(), key="home")
        
        with col2:
            st.markdown("**Away Team**")
            away_team = st.selectbox("", teams_df['name'].tolist(), key="away")
        
        if home_team != away_team:
            home_data = teams_df[teams_df['name'] == home_team].iloc[0]
            away_data = teams_df[teams_df['name'] == away_team].iloc[0]
            
            result = model.calculate_match_probabilities(home_data, away_data)
            
            # Display prediction
            st.markdown("---")
            st.subheader(f"🆚 {home_team} vs {away_team}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                
                    Over 0.5 Goals
                    {result['probabilities']['over_0_5']:.1%}
                
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                
                    Over 1.5 Goals
                    {result['probabilities']['over_1_5']:.1%}
                
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                
                    Under 1.5 Goals
                    {result['probabilities']['under_1_5']:.1%}
                
                """, unsafe_allow_html=True)
            
            # Visualization
            fig = go.Figure()
            
            goals = list(range(6))
            probabilities = [result['exact_scores'][f'{i}_goals'] for i in goals]
            
            fig.add_trace(go.Bar(
                x=goals,
                y=probabilities,
                marker_color='#00ff41',
                name='Probability'
            ))
            
            fig.update_layout(
                title=f"First Half Goals Distribution - {home_team} vs {away_team}",
                xaxis_title="Total Goals",
                yaxis_title="Probability",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    elif page == "My Account" and st.session_state.user_email:
        st.title("👤 My Account")
        
        user_info = get_user_info(st.session_state.user_email)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            
                📧 Email
                {user_info['email']}
            
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            
                💳 Subscription
                {user_info['subscription_tier'].title()}
            
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
