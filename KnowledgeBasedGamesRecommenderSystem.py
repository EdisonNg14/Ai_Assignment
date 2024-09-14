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
    
    # Custom CSS for centering and styling
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f5;  /* Set background color for the whole page */
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
    
    # Center the UI without container
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    
    # Title and description
    st.title("🎮 Game Recommendation System")
    st.markdown("""
    <p style='text-align: center;'>
    Welcome to the <strong>Game Recommendation System</strong>! This app helps you find new games based on your preferences.<br>
    Follow these steps to get personalized game recommendations:
    </p>
    <ol style='text-align: center;'>
        <li>Upload your game data in CSV format.</li>
        <li>Enter your preferred genre and minimum user score.</li>
        <li>Click "Get Recommendations" to view the top suggestions.</li>
    </ol>
    """, unsafe_allow_html=True)

    # Divider line
    st.markdown("<hr>", unsafe_allow_html=True)

    # Upload section
    st.subheader("📂 Upload Your Game Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Data processing
            df['Genres'] = df['Genres'].astype(str).fillna('')
            df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')

            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(df.head())

            # Divider line
            st.markdown("<hr>", unsafe_allow_html=True)

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
                with st.spinner("Processing your request..."):
                    try:
                        recommended_games = recommend_games(df, {'Genres': genres, 'Minimum User Score': min_user_score})

                        if not recommended_games.empty:
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games")
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
                            st.warning("No games match your preferences. Try adjusting the genre or score.")
                    except Exception as e:
                        st.error(f"An error occurred while processing recommendations: {e}")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")
    
    # Close the centering div
    st.markdown('</div>', unsafe_allow_html=True)

def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df

if __name__ == "__main__":
    main()
