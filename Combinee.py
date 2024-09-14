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

# Custom CSS for background color and button styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

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

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About", "Content-Based Recommendations", "File Upload and Filters", "Game Correlation Finder"])

# Page: Home
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Welcome to the Game Recommendation System!</h1>", unsafe_allow_html=True)
    st.markdown("""
    This app is designed to help you discover new games based on your preferences and the games you already enjoy.
    Use the navigation sidebar to explore the following features:
    
    - **Content-Based Recommendations**: Find games similar to your favorites.
    - **File Upload and Filters**: Get personalized recommendations based on the data you upload.
    - **Game Correlation Finder**: Explore the relationship between different games based on user ratings.
    
    Enjoy your gaming journey! 🎮
    """)

# Page: About
elif page == "About":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📖 About This App</h1>", unsafe_allow_html=True)
    st.markdown("""
    This app uses advanced machine learning techniques like **TF-IDF** and **cosine similarity** to recommend video games based on their genres, platforms, publishers, and user scores.
    
    - **Content-Based Filtering**: We analyze the characteristics of the games you like and recommend others that share similar features.
    - **User Score Correlation**: By analyzing user ratings, the app can show how different games are correlated with each other.

    ### Technologies Used:
    - **Streamlit**: For building the interactive web application.
    - **Pandas**: For data manipulation and analysis.
    - **Scikit-learn**: For machine learning algorithms like TF-IDF and cosine similarity.
    
    This app is part of a game recommendation project, designed to give personalized suggestions and insights into the gaming world.
    """)

# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Find Games Similar to Your Favorite</h2>", unsafe_allow_html=True)
    st.write("This app helps you find games similar to the ones you like. Enter the game title below to get recommendations.")

    # Add a selectbox for game selection
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    # Filters within the main page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    # Game information display
    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: **{game_info['Title']}**")
        st.write(f"**Genres:** {game_info['Genres']}")
        st.write(f"**Platforms:** {game_info['Platforms']}")
        st.write(f"**Publisher:** {game_info['Publisher']}")
        st.write(f"**User Score:** {game_info['User Score']}")
        st.write(f"**Release Date:** {game_info['Release Date']}")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to **{game_input}**:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please try another.")

# Page 2: File Upload and Filters
elif page == "File Upload and Filters":
    st.title("📂 Upload Your Game Data")
    st.markdown("""
    Follow these steps to get personalized game recommendations:
    1. Upload your game data in CSV format.
    2. Enter your preferred genre and minimum user score.
    3. Click "Get Recommendations" to view the top suggestions.
    """)

    # Upload section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the dataset
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Data processing
            df_uploaded['Genres'] = df_uploaded['Genres'].astype(str).fillna('')
            df_uploaded['User Score'] = pd.to_numeric(df_uploaded['User Score'], errors='coerce')

            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(df_uploaded.head())

            # Filter options section
            st.subheader("🎯 Filter Options")

            genres = st.text_input(
                "Preferred Genre (e.g., Action, Adventure):",
                value="Action",
                placeholder="Enter genres (e.g., Action, Adventure)",
                help="Enter the genre(s) you like. Use commas for multiple genres."
            ).strip()

            min_user_score_str = st.text_input(
                "Minimum Acceptable User Score (0.0 to 10.0):",
                value="0.0",
                placeholder="Enter a score between 0.0 and 10.0",
                help="Enter a number between 0.0 and 10.0 for the minimum score."
            )

            try:
                min_user_score = float(min_user_score_str)
                if min_user_score < 0.0 or min_user_score > 10.0:
                    st.error("Score must be between 0.0 and 10.0.")
                    min_user_score = 0.0
            except ValueError:
                st.error("Please enter a valid numeric score.")
                min_user_score = 0.0

            # Button to get recommendations
            if st.button("Get Recommendations"):
               
