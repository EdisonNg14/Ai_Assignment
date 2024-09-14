import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title for the Streamlit app with custom styling
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

df = load_data()

# Vectorize the content using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
content_matrix = vectorizer.fit_transform(df['content'])

# Function to recommend games based on cosine similarity
def content_based_recommendations(game_name, num_recommendations=5):
    try:
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(content_matrix, content_matrix)

        # Get the index of the input game
        idx = df[df['Title'].str.lower() == game_name.lower()].index[0]

        # Get similarity scores for all games
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort games based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get indices of the most similar games
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

        # Return the most similar games
        return df.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Sidebar content (list of pages, filters, etc.)
st.sidebar.title("Game Recommendation System")
st.sidebar.subheader("Pages:")
st.sidebar.markdown("""
- Home
- Search for Game
- About
""")

# Sidebar for additional filters within the Search page
st.sidebar.subheader("Filters")
num_recommendations = st.sidebar.slider('Number of recommendations', min_value=1, max_value=10, value=5)

# Main page
st.subheader("Search for a Game")

# Add a selectbox for game selection
game_list = df['Title'].unique()
game_input = st.selectbox("Choose a game from the list:", game_list)

# Game information display
if game_input:
    game_info = df[df['Title'] == game_input].iloc[0]
    st.markdown(f"### Selected Game: **{game_info['Title']}**")
    st.write(f"**Genres:** {game_info['Genres']}")
    st.write(f"**Platforms:** {game_info['Platforms']}")
    st.write(f"**Publisher:** {game_info['Publisher']}")
    st.write(f"**User Score:** {game_info['User Score']}")
    st.write(f"**Release Date:** {game_info['Release Date']}")

# Button to generate recommendations
if st.button('Get Recommendations'):
    recommendations = content_based_recommendations(game_input, num_recommendations)
    
    if not recommendations.empty:
        st.markdown(f"### Games similar to **{game_input}**:")
        st.table(recommendations)
    else:
        st.write("No matching game found. Please try another.")

# About section
st.markdown("<h3>About this App</h3>", unsafe_allow_html=True)
st.write("""
    The Game Recommendation System uses content-based filtering to recommend games.
    It combines features like genres, platforms, and publishers to find games similar to the one you choose.
""")

# Footer
st.markdown("<h5 style='text-align: center;'>Powered by Streamlit</h5>", unsafe_allow_html=True)
