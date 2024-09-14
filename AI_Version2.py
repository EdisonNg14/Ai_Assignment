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

        sim_scores = sim_scores[1:11]

        # Get indices of the most similar games
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

        # Return the most similar games
        return df.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])


# Initialize session state for the current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar for navigation (similar to ChatGPT-style buttons)
st.sidebar.title("Navigation")

# Sidebar buttons with session state handling
if st.sidebar.button("Home"):
    st.session_state.current_page = "Home"
if st.sidebar.button("Search for Game"):
    st.session_state.current_page = "Search for Game"
if st.sidebar.button("About"):
    st.session_state.current_page = "About"

# Page: Home
if st.session_state.current_page == "Home":
    st.markdown("<h2>Welcome to the Game Recommendation System</h2>", unsafe_allow_html=True)
    st.write("""
        This app helps you find similar games based on the one you like. 
        Use the sidebar to navigate through the app. 
        You can search for a game and get recommendations for similar games, or learn more about how the system works in the 'About' section.
    """)

# Page: Search for Game
elif st.session_state.current_page == "Search for Game":
    st.subheader("Search for a Game")

    #additional filters within the Search page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

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

# Page: About
elif st.session_state.current_page == "About":
    st.subheader("About this App")
    st.write("""
        The Game Recommendation System uses a content-based filtering approach to find similar games.
        It takes into account genres, platforms, and publishers, and recommends games that are most similar to the one you search for.
        
        The system utilizes:
        - **TF-IDF Vectorizer** to convert textual information into numerical vectors.
        - **Cosine Similarity** to compute the similarity between games.
        
        Explore different games and find new ones that match your preferences!
    """)

# Footer
st.markdown("<h5 style='text-align: center;'>Powered by Streamlit</h5>", unsafe_allow_html=True)
