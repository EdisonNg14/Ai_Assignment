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

# Load and cache the data
df, df_game_name = load_data()

# Data Cleaning Process
def clean_data(df):
    # Drop rows with missing critical columns
    df.dropna(subset=['Product Rating', 'Platforms', 'Genres', 'Publisher', 'Release Date', 'User Score', 'User Ratings Count'], inplace=True)
    
    # Drop rows with specific conditions
    df = df[df['Platforms'] != 'Meta Quest']
    df = df[df['Genres'] != 'Misc']
    df = df[df['Publisher'] != 'Unknown']
    
    # Convert columns to appropriate types
    df['Release Date'] = df['Release Date'].astype(str)
    df['User Score'] = df['User Score'].astype(float)
    df.drop(columns=['Developer'], inplace=True)
    
    return df

# Clean the data
df = clean_data(df)

# One-hot encoding for categorical data
df.set_index('Title', inplace=True)
one_hot_label = pd.get_dummies(df.select_dtypes(include='object'))
df = pd.concat([df, one_hot_label], axis=1)
df.drop(columns=df.select_dtypes(include='object').columns, inplace=True)

# MinMaxScaler for numerical data
scaler = MinMaxScaler()
df[df.select_dtypes(include='float64').columns] = scaler.fit_transform(df.select_dtypes(include='float64'))

# Nearest Neighbors model setup
model = NearestNeighbors(metric='euclidean')
model.fit(df)

# Cosine similarity calculation
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

# Function to recommend games based on Nearest Neighbors model
def game_recommended(gamename: str, recommended_games: int = 6):
    if gamename not in df.index:
        return pd.DataFrame(columns=["Game", "Similarity"])  # Return empty DataFrame if game not found
    
    distances, neighbors = model.kneighbors(df.loc[[gamename]], n_neighbors=recommended_games)
    similar_games = [df.index[neighbor] for neighbor in neighbors[0]]
    similar_distances = [f"{round(100 - distance, 2)}%" for distance in distances[0]]
    return pd.DataFrame(data={"Game": similar_games[1:], "Similarity": similar_distances[1:]})

# Function to recommend games based on cosine similarity
def cosine_game_recommended(gamename: str, recommended_games: int = 5):
    if gamename not in cosine_sim_df.index:
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    # If game is not found, return an empty DataFrame
    if gamename not in cosine_sim_df.index:
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    # Get similarity scores and indices
    sim_scores = cosine_sim_df.loc[gamename]
    top_indices = sim_scores.nlargest(recommended_games + 1).index
    top_scores = sim_scores[top_indices]
    
    # Remove the game itself from the results
    if gamename in top_indices:
        top_indices = top_indices[top_indices != gamename]
        top_scores = top_scores.drop(gamename)
    
    similar_games = top_indices
    cosine_scores = top_scores
    
    return pd.DataFrame(data={"Game": similar_games, "Cosine Similarity": cosine_scores})

# Main page functionality
st.subheader("Search for a Game")
game_input = st.text_input("Enter the game name:")

model_type = st.selectbox("Choose recommendation model", ["Euclidean Distance", "Cosine Similarity"])

if st.button('Get Recommendations'):
    # Search for the game in the dataset
    matching_games = df_game_name['Game'].str.contains(game_input, case=False, na=False)
    
    if matching_games.any():
        selected_game = df_game_name[matching_games].iloc[0]['Game']  # Get the first matching game
        st.write(f"Recommendations for the game: {selected_game}")
        
        # Depending on the model type selected, get the recommendations
        if model_type == "Euclidean Distance":
            recommendations = game_recommended(selected_game)
        elif model_type == "Cosine Similarity":
            recommendations = cosine_game_recommended(selected_game)

        # Display the recommendations
        if recommendations.empty:
            st.write("No recommendations available.")
        else:
            st.table(recommendations)
    else:
        st.write("No matching game found. Please try again.")
