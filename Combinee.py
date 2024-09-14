import streamlit as st

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar and button styling
st.markdown("""
    <style>
    .css-1d391kg {  /* Adjust based on your Streamlit version for the sidebar */
        background-color: #f0f2f6;
        border-right: 1px solid #ddd;
    }
    .css-1d391kg .stRadio>div>div>div>div>div {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .css-1d391kg .stRadio>div>div>div>div>div:hover {
        background-color: #e1e4e8;
    }
    .css-1d391kg .stRadio>div>div>div>div>div.stSelected {
        background-color: #4CAF50;
        color: white;
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

# Navigation Sidebar
st.sidebar.title("Navigation")

# Define the pages with icons
pages = {
    "Home": "🏠",
    "Content-Based Recommendations": "🔍",
    "Top 10 Recommendation based on User Preference": "📈",
    "Game Correlation Finder": "🔗",
    "About": "ℹ️"
}

# Create a radio button for page selection with icons
page = st.sidebar.radio(
    "Go to",
    options=list(pages.keys()),
    format_func=lambda page: f"{pages[page]} {page}"
)

# Page content based on selection
if page == "Home":
    st.title("🎮 Welcome to the Game Recommendation System")
    st.markdown("""Welcome to the *Game Recommendation System*! This app provides various ways to find your next favorite video game.

    ### Features:
    - *Content-Based Recommendations:* Find games similar to the ones you already enjoy.
    - *Top 10 Recommendations:* Get personalized recommendations based on your uploaded game data.
    - *Game Correlation Finder:* Discover how games are related based on user ratings.

    Use the navigation sidebar to explore different features of the app.
    """)

elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Find Games Similar to Your Favorite</h2>", unsafe_allow_html=True)
    st.write("This app helps you find games similar to the ones you like. Enter the game title below to get recommendations.")

    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: {game_info['Title']}")
        st.write(f"Genres: {game_info['Genres']}")
        st.write(f"Platforms: {game_info['Platforms']}")
        st.write(f"Publisher: {game_info['Publisher']}")
        st.write(f"User Score: {game_info['User Score']}")
        st.write(f"Release Date: {game_info['Release Date']}")

    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to {game_input}:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please try another.")

elif page == "Top 10 Recommendation based on User Preference":
    st.title("📂 Upload Your Game Data")
    st.markdown("""Follow these steps to get personalized game recommendations:
    1. Upload your game data in CSV format.
    2. Enter your preferred genre and minimum user score.
    3. Click "Get Recommendations" to view the top suggestions.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            df_uploaded['Genres'] = df_uploaded['Genres'].astype(str).fillna('')
            df_uploaded['User Score'] = pd.to_numeric(df_uploaded['User Score'], errors='coerce')

            st.write("### Dataset Preview")
            st.dataframe(df_uploaded.head())

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

            if st.button("Get Recommendations"):
                with st.spinner("Processing your request..."):
                    try:
                        recommended_games = recommend_games(df_uploaded, {'Genres': genres, 'Minimum User Score': min_user_score})
                        if not recommended_games.empty:
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games")
                            st.dataframe(top_10_games)

                            csv = top_10_games.to_csv(index=False)
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=csv,
                                file_name='recommended_games.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No games match your preferences. Try adjusting the genre or score.")
                    except Exception as e:
                        st.error(f"An error occurred while processing recommendations: {e}")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

elif page == "Game Correlation Finder":
    st.title('🎮 Game Correlation Finder')
    st.markdown("Find out how games are related based on user ratings!")

    @st.cache_data
    def load_data():
        path = 'all_video_games(cleaned).csv'
        df = pd.read_csv(path)
        path_user = 'User_Dataset.csv'
        userset = pd.read_csv(path_user)
        data = pd.merge(df, userset, on='Title').dropna()  
        return data

    data = load_data()

    score_matrix = data.pivot_table(index='user_id', columns='Title', values='user_score', fill_value=0)
    game_titles = score_matrix.columns.sort_values().tolist()

    col1, col2 = st.columns([1, 3])

    with col1:
        game_title = st.selectbox("Select a game title", game_titles, help="Choose a game to see its correlation with others.")

    st.markdown("---")

    if game_title:
        game_user_score = score_matrix[game_title]
        similar_to_game = score_matrix.corrwith(game_user_score)
        corr_drive = pd.DataFrame(similar_to_game, columns=['Correlation']).dropna()

        with col2:
            st.subheader(f"🎯 Correlations for '{game_title}'")
            st.dataframe(corr_drive.sort_values('Correlation', ascending=False).head(10))

        user_scores_count = data.groupby('Title')['user_score'].count().rename('total num_of_user_score')
        merged_corr_drive = corr_drive.join(user_scores_count, how='left')

        avg_user_score = data.groupby('Title')['user_score'].mean().rename('avg_user_score')
        detailed_corr_info = merged_corr_drive.join(avg_user_score, how='left')

        additional_info = data[['Title', 'Developer', 'Genres']].drop_duplicates().set_index('Title')
        detailed_corr_info = detailed_corr_info.join(additional_info, how='left')

        st.subheader("Games you may like (with more than 10 number of user scores
