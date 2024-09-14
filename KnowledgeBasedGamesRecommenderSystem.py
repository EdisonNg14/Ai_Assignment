import streamlit as st
import pandas as pd

def main():
    # Set page config
    st.set_page_config(
        page_title="Game Recommendation System",
        page_icon=":video_game:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #black;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
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
    
    # Title and description
    st.title("🎮 Game Recommendation System")
    st.markdown("""
    Welcome to the **Game Recommendation System**! This app helps you find new games based on your preferences. 
    Follow these steps to get personalized game recommendations:
    1. **Upload** your game data in CSV format.
    2. **Enter** your preferred genre and minimum user score.
    3. **Click** "Get Recommendations" to view the top suggestions.
    """)
    
    # Upload section
    st.header("Upload Your Game Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Data processing
            df['Genres'] = df['Genres'].astype(str).fillna('')
            df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

            # Display dataset preview
            st.write("### Dataset Preview:")
            st.dataframe(df.head())

            # Filter options section
            st.header("Filter Options")

            genres = st.text_input(
                "Preferred Genre (e.g., Action, Adventure):",
                value="Action",
                help="Enter the genre(s) you like. Use commas for multiple genres."
            ).strip()

            min_user_score_str = st.text_input(
                "Minimum Acceptable User Score (0.0 to 10.0):",
                value="0.0",
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
                with st.spinner("Processing your request..."):
                    try:
                        recommended_games = recommend_games(df, {'Genres': genres, 'Minimum User Score': min_user_score})

                        if not recommended_games.empty:
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games:")
                            st.dataframe(top_10_games)

                            # Download button
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
                        st.error(f"An error occurred while processing recommendations: {e}")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df

if __name__ == "__main__":
    main()
