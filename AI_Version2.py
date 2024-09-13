# Streamlit and other necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Title for the Streamlit app with a better UI layout
st.set_page_config(page_title="Game Recommendation System", layout="wide")
st.title('🎮 Game Recommendation System')

# Load the dataset
@st.cache_resource
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df_game_name = pd.DataFrame({'Game': df['Title']}).reset_index(drop=True)
    return df, df_game_name

# Load the dataset
df, df_game_name = load_data()

# Data Cleaning Process
def clean_data(df):
    for col in ['Product Rating', 'Platforms', 'Genres', 'Publisher', 'Release Date', 'User Score', 'Developer', 'User Ratings Count']:
        df.dropna(subset=[col], inplace=True)

    df = df[df['Platforms'] != 'Meta Quest']
    df = df[df['Genres'] != 'Misc']
    df = df[df['Publisher'] != 'Unknown']
    df['Release Date'] = df['Release Date'].astype('str')
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
    for i, column in enumerate(column_numeric):
        df[column] = scaled[:, i]
    
    return df

# Clean the dataset
df = clean_data(df)

# Model initialization
model = NearestNeighbors(metric='euclidean')
model.fit(df)

# Cosine similarity matrix calculation
cosine_sim = cosine_similarity(df)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

# Functions for recommendations
def GameRecommended(gamename:str, recommended_games:int=6):
    distances, neighbors = model.kneighbors(df.loc[gamename], n_neighbors=recommended_games)
    similar_game = [df_game_name.loc[neighbor][0] for neighbor in neighbors[0]]
    similar_distance = [f"{round(100 - distance, 2)}%" for distance in distances[0]]
    return pd.DataFrame(data={"Game": similar_game[1:], "Similarity": similar_distance[1:]})

def CosineGameRecommended(gamename:str, recommended_games:int=5):
    arr, ind = np.unique(cosine_sim_df.loc[gamename], return_index=True)
    similar_game = [df_game_name.loc[index][0] for index in ind[-(recommended_games+1):-1]]
    cosine_score = [arr[index] for index in range(-(recommended_games+1), -1)]
    return pd.DataFrame(data={"Game": similar_game, "Cosine Similarity": cosine_score}).sort_values(by='Cosine Similarity', ascending=False)

# UI layout with tabs
tab1, tab2 = st.tabs(["🔍 Search and Recommend", "📊 Visualize Data"])

with tab1:
    st.header("Game Search & Recommendations")
    
    # Search box for game selection on the main page
    game_input = st.text_input("Enter the game name:", "")
    game_input = game_input.lower()

    # Recommendation settings (side by side layout)
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Choose recommendation model", ["Euclidean Distance", "Cosine Similarity"])
    with col2:
        n_recommendations = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    # Button to generate recommendations
    if st.button('Get Recommendations'):
        matching_games = df_game_name['Game'].apply(lambda x: x.lower()).str.contains(game_input)

        if matching_games.any():
            selected_game = df_game_name[matching_games].iloc[0]['Game']
            st.write(f"Recommendations for the game: {selected_game}")
            
            if model_type == "Euclidean Distance":
                recommendations = GameRecommended(selected_game, n_recommendations)
            elif model_type == "Cosine Similarity":
                recommendations = CosineGameRecommended(selected_game, n_recommendations)
            
            # Display recommendations
            st.table(recommendations)
        else:
            st.warning("No matching game found. Please try again.")

with tab2:
    st.header("Data Visualizations")
    
    # Show dataset overview
    st.subheader("Game Data Overview")
    st.dataframe(df_game_name.head(10))
    
    # Show distribution of genres or platforms
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Platform Distribution")
        platform_counts = df['Platforms'].value_counts().nlargest(10)
        st.bar_chart(platform_counts)
    
    with col2:
        st.subheader("Genre Distribution")
        genre_counts = df['Genres'].value_counts().nlargest(10)
        st.bar_chart(genre_counts)
