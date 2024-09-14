import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

# Vectorize the content using TfidfVectorizer
def content_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(df['content'])

# Function for content-based recommendations
def content_based_recommendations(df, content_matrix, game_name, num_recommendations=5):
    try:
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(content_matrix, content_matrix)

        # Get the index of the input game
        idx = df[df['Title'].str.lower() == game_name.lower()].index[0]

        # Get similarity scores for all games
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort games based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get indices of the most similar games
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

        # Return the most similar games
        return df.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Function for user preference-based recommendations
def recommend_games(df, preferences):
    # Filter by genre
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    
    # Filter by user score
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    
    # Apply filters
    return df[genre_filter & score_filter]

# Navigation and main function
def main():
    st.title("Game Recommendation System")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Content-based Recommendations", "CSV-based Recommendations"])

    if page == "Content-based Recommendations":
        st.markdown("<h2>Welcome to the Content-based Recommendation System</h2>", unsafe_allow_html=True)
        
        df = load_data()  # Load data for content-based recommendations
        content_matrix = content_vectorizer(df)  # Vectorize the content

        # Search Box on the Main Page
        game_input = st.text_input("Search for a game:", "")

        # Additional filters
        if game_input:
            st.subheader("Filters")
            num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

            # Display game info if found
            game_info = df[df['Title'].str.lower() == game_input.lower()]
            
            if not game_info.empty:
                game_info = game_info.iloc[0]
                st.markdown(f"### Selected Game: **{game_info['Title']}**")
                st.write(f"**Genres:** {game_info['Genres']}")
                st.write(f"**Platforms:** {game_info['Platforms']}")
                st.write(f"**Publisher:** {game_info['Publisher']}")
                st.write(f"**User Score:** {game_info['User Score']}")
                st.write(f"**Release Date:** {game_info['Release Date']}")
            else:
                st.write("Game not found. Please try another name.")
            
            # Button to get recommendations
            if st.button('Get Recommendations'):
                recommendations = content_based_recommendations(df, content_matrix, game_input, num_recommendations)
                
                if not recommendations.empty:
                    st.markdown(f"### Games similar to **{game_input}**:")
                    st.table(recommendations)
                else:
                    st.write("No recommendations available for this game.")

    elif page == "CSV-based Recommendations":
        st.markdown("<h2>Welcome to the CSV-based Recommendation System</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        Follow these steps:
        1. **Upload** a CSV file containing your game data.
        2. **Enter** your preferred genre and minimum acceptable user score.
        3. **Click** the "Get Recommendations" button to see your results.
        """)

        # Upload the dataset via Streamlit's file uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Load the uploaded dataset
                df = pd.read_csv(uploaded_file)
                
                # Ensure 'Genres' column is treated as a string
                df['Genres'] = df['Genres'].astype(str).fillna('')
                
                # Convert 'User Score' to numeric, handling non-numeric values
                df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

                # Display a preview of the dataset
                st.write("### Dataset Preview:")
                st.dataframe(df.head())

                st.sidebar.header("Filter Options")
                
                # Function to get user preferences via Streamlit inputs
                def get_user_preferences():
                    genres = st.sidebar.text_input(
                        "Enter your preferred genre (e.g., Action, Adventure):",
                        value="Action",
                        help="Type the genre you are interested in. For multiple genres, separate them with commas."
                    ).strip()
                    
                    # Minimum user score input with validation
                    min_user_score_str = st.sidebar.text_input(
                        "Enter your minimum acceptable user score (0.0 to 10.0):",
                        value="0.0",
                        help="Specify the minimum user score you are willing to accept."
                    )
                    
                    try:
                        min_user_score = float(min_user_score_str)
                        if min_user_score < 0.0 or min_user_score > 10.0:
                            st.sidebar.error("Score must be between 0.0 and 10.0.")
                            min_user_score = 0.0
                    except ValueError:
                        st.sidebar.error("Please enter a valid numeric score.")
                        min_user_score = 0.0
                    
                    return {
                        'Genres': genres,
                        'Minimum User Score': min_user_score
                    }

                # Get User Preferences
                user_preferences = get_user_preferences()

                if st.sidebar.button("Get Recommendations"):
                    # Recommend Games
                    if user_preferences['Genres']:  # Check if genres input is provided
                        try:
                            recommended_games = recommend_games(df, user_preferences)
                            
                            if not recommended_games.empty:
                                top_10_games = recommended_games.head(10)
                                st.write("### Top 10 Recommended Games based on your preferences:")
                                st.dataframe(top_10_games)

                                # Download recommendations as CSV
                                csv = top_10_games.to_csv(index=False)
                                st.download_button(
                                    label="Download Recommendations as CSV",
                                    data=csv,
                                    file_name='recommended_games.csv',
                                    mime='text/csv'
                                )
                            else:
                                st.write("No games match your preferences. Try adjusting the genre or score.")
                        except Exception as e:
                            st.write(f"An error occurred while processing recommendations: {e}")
                    else:
                        st.write("Please enter a genre to filter by.")
            except Exception as e:
                st.write(f"An error occurred while loading the file: {e}")
        else:
            st.write("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()

