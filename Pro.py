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
        background-color: #black;
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

# Load the data for content-based and knowledge-based recommendations
@st.cache_data
def load_data():
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

df_content = load_data()
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

# New function: Knowledge-based filtering
def knowledge_based_recommendations(df, genre, platform, developer, min_score):
    # Apply filters based on user inputs
    genre_filter = df['Genres'].str.contains(genre, case=False, na=False) if genre else True
    platform_filter = df['Platforms'].str.contains(platform, case=False, na=False) if platform else True
    developer_filter = df['Publisher'].str.contains(developer, case=False, na=False) if developer else True
    score_filter = df['User Score'] >= min_score

    # Combine all filters
    filtered_df = df[genre_filter & platform_filter & developer_filter & score_filter]
    return filtered_df

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Content-Based Recommendations", "Knowledge-Based Filtering", "File Upload and Filters", "Game Correlation Finder", "About"])

# Page: Home
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Game Recommendation System! Navigate through the app using the sidebar to explore different recommendation methods.")

# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Content-Based Recommendations</h1>", unsafe_allow_html=True)
    st.write("This method helps you find games similar to the ones you like based on their content features.")

    # Add a selectbox for game selection
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    # Filters within the main page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    # Game information display
    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: *{game_info['Title']}*")
        st.write(f"*Genres:* {game_info['Genres']}")
        st.write(f"*Platforms:* {game_info['Platforms']}")
        st.write(f"*Publisher:* {game_info['Publisher']}")
        st.write(f"*User Score:* {game_info['User Score']}")
        st.write(f"*Release Date:* {game_info['Release Date']}")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to *{game_input}*:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please try another.")

# Page 2: Knowledge-Based Filtering
elif page == "Knowledge-Based Filtering":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Knowledge-Based Filtering</h1>", unsafe_allow_html=True)
    st.write("This method allows you to specify your game preferences for tailored recommendations.")

    # Knowledge-based filtering inputs
    genre_input = st.text_input("Preferred Genre (e.g., Action, Adventure):", "").strip()
    platform_input = st.text_input("Preferred Platform (e.g., PC, PS4):", "").strip()
    developer_input = st.text_input("Preferred Developer/Publisher (e.g., Ubisoft):", "").strip()
    min_score_input = st.slider("Minimum User Score:", min_value=0.0, max_value=10.0, value=7.0)

    # Button to get knowledge-based recommendations
    if st.button("Get Knowledge-Based Recommendations"):
        kb_recommendations = knowledge_based_recommendations(df_content, genre_input, platform_input, developer_input, min_score_input)
        if not kb_recommendations.empty:
            st.write("### Recommended Games Based on Your Preferences")
            st.table(kb_recommendations[['Title', 'Genres', 'User Score', 'Platforms', 'Publisher', 'Release Date']])
        else:
            st.warning("No games match your specified preferences. Try adjusting the filters.")

# Page 3: File Upload and Filters
elif page == "File Upload and Filters":
    st.title("📂 Upload Your Game Data")
    st.write("Upload your game dataset and use filters to get personalized game recommendations.")

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
                    raise ValueError("Score must be between 0.0 and 10.0.")
            except ValueError:
                st.error("Invalid score input. Please enter a number between 0.0 and 10.0.")
                min_user_score = 0.0

            # Apply filters to get recommendations
            preferences = {'Genres': genres, 'Minimum User Score': min_user_score}
            recommendations = recommend_games(df_uploaded, preferences)

            # Show the filtered recommendations
            st.subheader("🎮 Recommended Games")
            if not recommendations.empty:
                st.dataframe(recommendations[['Title', 'Genres', 'User Score', 'Release Date']])
            else:
                st.warning("No games found matching your preferences. Try adjusting the filters.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Page 4: Game Correlation Finder
elif page == "Game Correlation Finder":
    st.title("🔍 Game Correlation Finder")
    st.write("Find games similar to your favorite game based on user ratings.")
    
    # Select a game
    game_list = df_corr['Title'].unique()
    selected_game = st.selectbox("Choose a game:", game_list)

    if st.button("Find Similar Games"):
        corr_matrix = df_corr.corr()
        game_corr = corr_matrix[selected_game].sort_values(ascending=False)
        
        # Display top 10 similar games
        st.write(f"Games similar to {selected_game}:")
        st.write(game_corr.head(10))

# Page 5: About
elif page == "About":
    st.title("About")
    st.write("This Game Recommendation System uses a variety of algorithms to recommend games based on different criteria. Explore the methods using the sidebar.")

