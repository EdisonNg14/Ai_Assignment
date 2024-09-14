import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS with a background image
background_css = """
    <style>
    .main {
        background-image: url('360_F_88981880_YjJManMJ6hJmKr5CZteFJAkEzXIh8mxW.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }

    /* Additional styling for text, buttons, etc. */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown {
        color: white !important;
    }

    /* Styling buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
"""

# Inject CSS into the app
st.markdown(background_css, unsafe_allow_html=True)

# Add the rest of your Streamlit content
# Example of adding a title and explanation
st.title("🎮 Welcome to the Game Recommendation System")
st.markdown("""
    Welcome to the *Game Recommendation System*! This app helps you find your next favorite video game.
    Explore different features through the navigation sidebar.
""")

# Rest of your app code goes here...
