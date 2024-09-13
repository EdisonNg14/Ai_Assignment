import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Title for the Streamlit app
st.title('Game Recommendation System')

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df_game_name = pd.DataFrame({'Game': df['Title']}).reset_index(drop=True)
    return df, df_game_name

df, df_game_name = load_data()

# Data Cleaning Process
df = df.dropna(subset=['Product Rating', 'Platforms', 'Genres', 'Publisher', 'Release Date', 'User Score', 'Developer', 'User Ratings Count'])
df = df[df['Platforms'].notna() & ~df['Platforms'].isin(['Meta Quest'])]
df = df[df['Genres'] != 'Misc']
df = df[df['Publisher'].notna() & (df['Publisher'] != 'Unknown')]
df['Release Date'] = df['Release Date'].astype(str)
df['User Score'] = df['User Score'].astype(float)
df = df.drop(columns=['Developer'])

# One-hot encoding for categorical data
df.set_index('Title', inplace=True)
df = pd.get_dummies(df, drop_first=True)

# MinMaxScaler for numerical data
scaler = MinMaxScaler()
numeric_cols = df.select_dtypes(include=['float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Nearest Neighbors model setup
model = NearestNeighbors(metric='euclidean')
model.fit(df)

# Cosine similarity calculation
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

# Function to recommend games based on Nearest Neighbors model
def GameRecommended(gamename: str, recommended_games: int = 6):
    try:
        distances, neighbors = model.kneighbors(df.loc[[gamename]], n_neighbors=recommended_games + 1)
        similar_games = [df_game_name.loc[neighbor][0] for neighbor in neighbors[0][1:]]
        similar_distances = [f"{round(100 - distance, 2)}%" for distance in distances[0][1:]]
        return pd.DataFrame({"Game": similar_games, "Similarity": similar_distances})
    except KeyError:
        return pd.DataFrame(columns=["Game", "Similarity"])

# Function to recommend games based on cosine similarity
def CosineGameRecommended(gamename: str, recommended_games: int = 5):
    try:
        sim_scores = cosine_sim_df.loc[gamename].sort_values(ascending=False)
        similar_games = sim_scores.index[1:recommended_games + 1]
        scores = sim_scores[1:recommended_games + 1]
        return pd.DataFrame({"Game": similar_games, "Cosine Similarity": scores})
    except KeyError:
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

# Main page
st.subheader("Search for a Game")
game_input = st.text_input("Enter the game name:", "")

# Sidebar options
model_type = st.sidebar.selectbox("Choose recommendation model", ["Euclidean Distance", "Cosine Similarity"])

# Button to generate recommendations
if st.button('Get Recommendations'):
    # Convert user input to lowercase for case-insensitive matching
    game_input_lower = game_input.lower()
    
    # Search for the game in the dataset
    matching_games = df_game_name['Game'].str.lower().str.contains(game_input_lower)

    if matching_games.any():
        selected_game = df_game_name[matching_games].iloc[0]['Game']  # Get the first matching game
        st.write(f"Recommendations for the game: {selected_game}")
        
        # Depending on the model type selected, get the recommendations
        if model_type == "Euclidean Distance":
            recommendations = GameRecommended(selected_game)
        elif model_type == "Cosine Similarity":
            recommendations = CosineGameRecommended(selected_game)
        
        # Display the recommendations
        if not recommendations.empty:
            st.table(recommendations)
        else:
            st.write("No recommendations found.")
    else:
        st.write("No matching game found. Please try again.")
