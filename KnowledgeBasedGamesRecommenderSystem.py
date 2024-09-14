import streamlit as st
import pandas as pd

# Upload the dataset via Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded dataset
    try:
        df = pd.read_csv(uploaded_file)
        
        # Ensure 'Genres' column is treated as a string
        df['Genres'] = df['Genres'].astype(str).fillna('')

        # Convert 'User Score' to numeric, handling non-numeric values
        df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

        # Function to get user preferences via Streamlit inputs
        def get_user_preferences():
            genres = st.sidebar.text_input("Enter your preferred genre:", value="Action").strip()  # Streamlit input for genre
            min_user_score = st.sidebar.number_input("Enter your minimum acceptable user score:", min_value=0.0, value=0.0)  # Numeric input for score
            return {
                'Genres': genres,
                'Minimum User Score': min_user_score
            }

        # Function to recommend games based on user preferences
        def recommend_games(df, preferences):
            # Filter by genre
            genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
            
            # Filter by user score
            score_filter = df['User Score'] >= preferences['Minimum User Score']
            
            # Apply filters
            filtered_df = df[genre_filter & score_filter]
            
            return filtered_df

        # Main Streamlit app logic
        st.title("Game Recommendation System")

        # Get User Preferences
        user_preferences = get_user_preferences()

        # Recommend Games
        if user_preferences['Genres']:  # Check if genres input is provided
            try:
                recommended_games = recommend_games(df, user_preferences)
                
                if not recommended_games.empty:
                    # Show top 10 recommendations
                    top_10_games = recommended_games.head(10)
                    st.write("Top 10 Recommended Games based on your preferences:")
                    st.dataframe(top_10_games)  # Display the top 10 recommendations in a table
                else:
                    st.write("No games match your preferences.")
            except Exception as e:
                st.write(f"An error occurred while processing recommendations: {e}")
    except Exception as e:
        st.write(f"An error occurred while loading the file: {e}")
else:
    st.write("Please upload a CSV file to get started.")
