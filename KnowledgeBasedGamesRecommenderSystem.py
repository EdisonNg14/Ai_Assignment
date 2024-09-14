import streamlit as st
import pandas as pd

def main():
    st.title("Game Recommendation System")
    
    st.markdown("""
    Welcome to the Game Recommendation System! Please follow these steps:
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
                # Example genre input
                genres = st.sidebar.text_input(
                    "Enter your preferred genre (e.g., Action, Adventure):",
                    value="Action",
                    help="Type the genre you are interested in. For multiple genres, separate them with commas."
                ).strip()
                
                # Minimum user score input with validation
                min_user_score_str = st.sidebar.text_input(
                    "Enter your minimum acceptable user score (0.0 to 10.0):",
                    value="0.0",
                    help="Specify the minimum user score you are willing to accept. Enter a number between 0.0 and 10.0."
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

            # Function to recommend games based on user preferences
            def recommend_games(df, preferences):
                # Filter by genre
                genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
                
                # Filter by user score
                score_filter = df['User Score'] >= preferences['Minimum User Score']
                
                # Apply filters
                filtered_df = df[genre_filter & score_filter]
                
                return filtered_df

            # Get User Preferences
            user_preferences = get_user_preferences()

            if st.sidebar.button("Get Recommendations"):
                # Recommend Games
                if user_preferences['Genres']:  # Check if genres input is provided
                    try:
                        recommended_games = recommend_games(df, user_preferences)
                        
                        if not recommended_games.empty:
                            # Show top 10 recommendations
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games based on your preferences:")
                            st.dataframe(top_10_games)  # Display the top 10 recommendations in a table

                            # Provide an option to download the recommendations
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
