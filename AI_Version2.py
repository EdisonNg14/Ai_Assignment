# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv('all_video_games(cleaned).csv')

df = load_data()

# Log the shape of the dataset to confirm it is loaded
st.write(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Save game names in a new dataframe
df_game_name = pd.DataFrame({'Game': df['Title']}).reset_index(drop=True)

# Check if the game names are loaded correctly
st.write("First few game titles:", df_game_name.head())

# Use the name column as index
df.set_index('Title', inplace=True)

# Select all columns with datatype object
column_object = df.dtypes[df.dtypes == 'object'].keys()

# Convert category data to one-hot encoding
one_hot_label = pd.get_dummies(df[column_object])

# Delete columns with datatype object
df.drop(column_object, axis=1, inplace=True)

# Unify one-hot encoding data with the rest of the data
df = pd.concat([df, one_hot_label], axis=1)

# Select all numeric columns
column_numeric = list(df.dtypes[df.dtypes == 'float64'].keys())

# MinMaxScaler initiation
scaler = MinMaxScaler()

# Numerical column data standardization
scaled = scaler.fit_transform(df[column_numeric])

# Apply scaled data
for i, column in enumerate(column_numeric):
    df[column] = scaled[:, i]

# Log to check that the data scaling is applied
st.write("Data after scaling:", df.head())

# Model initiation for Euclidean-based recommendations
model = NearestNeighbors(metric='euclidean')
model.fit(df)

# Function to get game recommendations using Euclidean distance
def GameRecommended(gamename: str, recommended_games: int = 6):
    try:
        distances, neighbors = model.kneighbors(df.loc[gamename], n_neighbors=recommended_games)
        similar_game = []
        for game in df_game_name.loc[neighbors[0]].values:
            similar_game.append(game[0])
        similar_distance = []
        for distance in distances[0]:
            similar_distance.append(f"{round(100 - distance, 2)}%")
        return pd.DataFrame(data={"Game": similar_game[1:], "Similarity": similar_distance[1:]})
    except Exception as e:
        st.write(f"Error in GameRecommended: {e}")
        return pd.DataFrame()

# Calculate cosine similarity of the dataframe
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_game_name['Game'], columns=df_game_name['Game'])

# Function to get game recommendations using cosine similarity
def CosineGameRecommended(gamename: str, recommended_games: int = 5):
    try:
        similarity_scores = cosine_sim_df[gamename].sort_values(ascending=False)
        similar_game = similarity_scores.iloc[1:recommended_games + 1].index  # Exclude the game itself
        cosine_score = similarity_scores.iloc[1:recommended_games + 1].values
        return pd.DataFrame(data={"Game": similar_game, "Cosine Similarity": cosine_score}).sort_values(by='Cosine Similarity', ascending=False)
    except Exception as e:
        st.write(f"Error in CosineGameRecommended: {e}")
        return pd.DataFrame()

# Streamlit app layout
st.title('Game Recommendation System')

# Search box for game name input
game_input = st.text_input("Enter the name of a game you like", "")

# Log the user input
st.write("User entered game name:", game_input)

if game_input:
    # Convert both input and game names to lowercase for case-insensitive matching
    game_input = game_input.lower()
    df_game_name['Game'] = df_game_name['Game'].str.lower()
    
    # Check if the game exists in the dataset
    if game_input in df_game_name['Game'].values:
        st.subheader(f'Game Recommendations based on "{game_input}"')
        
        # Euclidean-based recommendations
        st.write("**Recommendations using Euclidean Distance:**")
        euclidean_recommendations = GameRecommended(game_input)
        st.dataframe(euclidean_recommendations)
        
        # Cosine similarity-based recommendations
        st.write("**Recommendations using Cosine Similarity:**")
        cosine_recommendations = CosineGameRecommended(game_input)
        st.dataframe(cosine_recommendations)
    else:
        st.error("Game not found in the dataset. Please try another game.")
else:
    st.write("Please enter a game name to get recommendations.")
