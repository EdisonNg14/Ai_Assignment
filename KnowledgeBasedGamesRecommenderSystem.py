import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Game Recommendation System", page_icon=":video_game:", layout="wide")
    
    st.title("🎮 Game Recommendation System")
    
    st.markdown("""
    Welcome to the Game Recommendation System! Follow these steps to find your next favorite game:
    1. **Upload** a CSV file with game data.
    2. **Specify** your preferred genre and minimum acceptable user score.
    3. **Click** "Get Recommendations" to see the results.
    """)

    # File upload section
    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the uploaded dataset
            df = pd.read_csv(uploaded_file)
            
            # Ensure 'Genres' column is treated as a string
            df['Genres'] = df['Genres'].astype(str).fillna('')
            
            # Convert 'User Score' to numeric, handling non-numeric values
            df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

            # Display a preview of the dataset in the main area
            st.write("### Dataset Preview:")
            st.dataframe(df.head())

            # Sidebar for user preferences
            st.sidebar.header("Filter Options")
            
            # Get user preferences
            genres = st.sidebar.text_input(
                "Enter your preferred genre (e.g., Action, Adventure):",
                value="Action",
                help="Type the genre you are interested in. For multiple genres, separate them with commas."
            ).strip()
            
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

            # Function to recommend games based on user preferences
            def recommend_games(df, preferences):
                genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
                score_filter = df['User Score'] >= preferences['Minimum User Score']
                filtered_df = df[genre_filter & score_filter]
                return filtered_df

            # Recommend games when button is clicked
            if st.sidebar.button("Get Recommendations"):
                st.spinner("Processing your request...")
                if genres:  # Check if genres input is provided
                    try:
                        recommended_games = recommend_games(df, {'Genres': genres, 'Minimum User Score': min_user_score})
                        
                        if not recommended_games.empty:
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games:")
                            st.dataframe(top_10_games)  # Display the top 10 recommendations in a table

                            # Option to download recommendations
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
