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

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #333333;
        color: white;
    }
    
    /* Sidebar titles and text */
    .css-1d391kg h1, .css-1d391kg p {
        color: white;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Text styling */
    h1, h2, h3, h4 {
        font-family: 'Arial', sans-serif;
        text-align: center;
        color: #2C3E50;
    }
    p {
        font-family: 'Arial', sans-serif;
        color: #2C3E50;
        line-height: 1.6;
    }
    
    /* Game recommendation card */
    .game-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
        margin-bottom: 20px;
    }
    .game-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
    }
    .game-title {
        font-weight: bold;
        font-size: 20px;
    }
    .game-details {
        color: #555;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #2C3E50;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>Powered by Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the data for content-based recommendations
@st.cache_data
def load_content_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

# Load the data for correlation finder
@st.cache_data
def load_correlation_data():
    path = 'all_video_games(cleaned).csv'
    df = pd.read_csv(path)
    path_user = 'User_Dataset.csv'
    userset = pd.read_csv(path_user)
    data = pd.merge(df, userset, on='Title').dropna()  
    return data

df_content = load_content_data()
df_corr = load_correlation_data()

# Function to recommend games based on cosine similarity
def content_based_recommendations(game_name, num_recommendations=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(df_content['content'])

    try:
        cosine_sim = cosine_similarity(content_matrix, content_matrix)
        idx = df_content[df_content['Title'].str.lower() == game_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        return df_content.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Function to recommend games based on file upload and filters
def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df

# Create a custom sidebar menu
st.sidebar.title("Navigation")
pages = {
    "Home": "🏠",
    "Content-Based Recommendations": "🔍",
    "Top 10 Recommendation based on User Preferences": "📈",
    "Game Correlation Finder": "🔗",
    "About": "ℹ️"
}
for page, icon in pages.items():
    if st.sidebar.button(f"{icon} {page}"):
        st.session_state.page = page

# Set the default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Page Navigation
page = st.session_state.page

# Home Page
if page == "Home":
    st.title("🎮 Welcome to the Game Recommendation System")
    st.markdown("""Welcome to the *Game Recommendation System*! This app provides various ways to find your next favorite video game.""")
    
# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Find Games Similar to Your Favorite</h2>", unsafe_allow_html=True)
    
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    # Filters within the main page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    # Game information display
    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: {game_info['Title']}")
        st.write(f"Genres: {game_info['Genres']}")
        st.write(f"Platforms: {game_info['Platforms']}")
        st.write(f"Publisher: {game_info['Publisher']}")
        st.write(f"User Score: {game_info['User Score']}")
        st.write(f"Release Date: {game_info['Release Date']}")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to {game_input}:")
            for index, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="game-card">
                    <span class="game-title">{row['Title']}</span>
                    <p class="game-details">Genres: {row['Genres']}<br>User Score: {row['User Score']}<br>Platforms: {row['Platforms']}<br>Release Date: {row['Release Date']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No matching game found. Please try another.")

# More pages: You can add the rest of your pages similarly...

# Footer
st.markdown('<div class="footer"><p>Powered by Streamlit</p></div>', unsafe_allow_html=True)
