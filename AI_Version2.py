# Streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Title for the Streamlit app
st.title('Game Recommendation System')

# Load the data
@st.cache_resource
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df_game_name = pd.DataFrame({'Game': df['Title']}).reset_index(drop=True)
    return df, df_game_name

# Load and cache the data
df, df_game_name = load_data()

# Data Cleaning Process (same as your original code)
for index in df[df['Product Rating'].isna()].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['Platforms'].isna()].index:
    df.drop(index, axis=0, inplace=True)
platform_counter = Counter(df['Platforms'])
platform_less_than_350 = ['Meta Quest']
df = df[~df['Platforms'].isin(platform_less_than_350)]
for index in df[df['Genres'].isna()].index:
    df.drop(index, axis=0, inplace=True)
df = df[df['Genres'] != 'Misc']
for index in df[df['Publisher'].isna()].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['Publisher'] == 'Unknown'].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['Release Date'].isna()].index:
    df.drop(index, axis=0, inplace=True)
df['Release Date'] = df['Release Date'].astype('str')
for index in df[df['User Score'].isna()].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['Developer'].isna()].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['User Ratings Count'].isna()].index:
    df.drop(index, axis=0, inplace=True)
for index in df[df['Product Rating'].isna()].index:
    df.drop(index, axis=0, inplace=True)
df['User Score'] = df['User Score'].astype('float')
df.drop('Developer', axis=1, inplace=True)

# One-hot encoding for categorical data
df.set_index('Title', inplace=True)
column_object = df.dtypes[df.dtypes == 'object'].keys()
one_hot_label = pd.get_dummies(df[column_object])
df.drop(column_object, axis=1, inplace=True)
df = pd.concat([df, one_hot_label], axis=1)

# MinMaxScaler for numerical data
column_numeric = list(df.dtypes[df.dtypes == 'float64'].keys())
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[column_numeric])
i = 0
for column in column_numeric:
    df[column] = scaled[:, i]
    i += 1

# Nearest Neighbors model setup
model = NearestNeighbors(metric='euclidean')
model.fit(df)

# Cosine similarity calculation
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

# Function to recommend games based on Nearest Neighbors model
def GameRecommended(gamename:str, recommended_games:int=6):
    distances, neighbors = model.kneighbors(df.loc[gamename], n_neighbors=recommended_games)
    similar_game = [df_game_name.loc[neighbor][0] for neighbor in neighbors[0]]
    similar_distance = [f"{round(100 - distance, 2)}%" for distance in distances[0]]
    return pd.DataFrame(data={"Game": similar_game[1:], "Similarity": similar_distance[1:]})

# Function to recommend games based on cosine similarity
def CosineGameRecommended(gamename:str, recommended_games:int=5):
    arr, ind = np.unique(cosine_sim_df.loc[gamename], return_index=True)
    similar_game = [df_game_name.loc[index][0] for index in ind[-(recommended_games+1):-1]]
    cosine_score = [arr[index] for index in range(-(recommended_games+1), -1)]
    return pd.DataFrame(data={"Game": similar_game, "Cosine Similarity": cosine_score}).sort_values(by='Cosine Similarity', ascending=False)

# Sidebar options
model_type = st.sidebar.selectbox("Choose recommendation model", ["Euclidean Distance", "Cosine Similarity"])

# Search box for game selection on the main page
st.subheader("Search for a Game")
game_input = st.text_input("Enter the game name:", "")

# Convert user input to lowercase for case-insensitive matching
game_input = game_input.lower()

# Button to generate recommendations
if st.button('Get Recommendations'):
    # Search for the game in the dataset
    matching_games = df_game_name['Game'].apply(lambda x: x.lower()).str.contains(game_input)

    if matching_games.any():
        selected_game = df_game_name[matching_games].iloc[0]['Game']  # Get the first matching game
        st.write(f"Recommendations for the game: {selected_game}")
        
        # Depending on the model type selected, get the recommendations
        if model_type == "Euclidean Distance":
            recommendations = GameRecommended(selected_game)
        elif model_type == "Cosine Similarity":
            recommendations = CosineGameRecommended(selected_game)

        # Display the recommendations
        st.table(recommendations)
    else:
        st.write("No matching game found. Please try again.")
