import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title for the Streamlit app
st.title('Game Recommendation System')

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    return df

# Load and cache the data
df = load_data()

# Combine key features into a single 'content' field for each game
df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']

# Function to recommend games based on cosine similarity of content features
def content_based_recommendations(game_name, num_recommendations=5):
    try:
        # Vectorize the content using TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        content_matrix = vectorizer.fit_transform(df['content'])

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
        return df.iloc[sim_indices][['Title', 'Genres', 'User Score']]
    
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Main page functionality
st.subheader("Search for a Game")
game_input = st.text_input("Enter the game name:")

# Button to generate recommendations
if st.button('Get Recommendations'):
    if game_input:
        recommendations = content_based_recommendations(game_input)
        
        if not recommendations.empty:
            st.write(f"Top game recommendations similar to **{game_input}**:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please check the name and try again.")
    else:
        st.write("Please enter a game name to get recommendations.")
