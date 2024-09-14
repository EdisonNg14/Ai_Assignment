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
        background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3Dgame%2Bbackground&psig=AOvVaw3_XR93klPh6Vp6Ad2KZE4Z&ust=1726416341845000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCJjf-5vowogDFQAAAAAdAAAAABAE');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
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
