import pandas as pd
import streamlit as st

# Title with emojis for a friendly, engaging look
st.title('🎮 Game Correlation Finder')
st.markdown("Find out how games are related based on user ratings!")

@st.cache_data
def load_data():
    path = 'NewDataSet.csv'
    df = pd.read_csv(path)
    path_user = 'User_Dataset.csv'
    userset = pd.read_csv(path_user)
    data = pd.merge(df, userset, on='Title').dropna()  
    return data

data = load_data()

score_matrix = data.pivot_table(index='user_id', columns='Title', values='user_score', fill_value=0)

game_titles = score_matrix.columns.sort_values().tolist()

# Split layout into two columns for better organization
col1, col2 = st.columns([1, 3])

with col1:
    # Game title selection without 'expanded' or 'label_visibility' parameters
    game_title = st.selectbox("Select a game title", game_titles, help="Choose a game to see its correlation with others.")

# Divider line for better visual separation
st.markdown("---")

if game_title:
    game_user_score = score_matrix[game_title]
    similar_to_game = score_matrix.corrwith(game_user_score)
    corr_drive = pd.DataFrame(similar_to_game, columns=['Correlation']).dropna()

    # Display the top 10 correlated games in the second column
    with col2:
        st.subheader(f"🎯 Correlations for '{game_title}'")
        st.dataframe(corr_drive.sort_values('Correlation', ascending=False).head(10))

    # Display number of user scores for each game
    user_scores_count = data.groupby('Title')['user_score'].count().rename('total num_of_user_score')
    merged_corr_drive = corr_drive.join(user_scores_count, how='left')

    # Add developer and publisher columns (assuming they're in the dataset)
    additional_info = data[['Title', 'Developer', 'Genres']].drop_duplicates().set_index('Title')
    detailed_corr_info = merged_corr_drive.join(additional_info, how='left')

    # Show detailed high-score correlations with more information
    st.subheader("Detailed High Score Correlations (with > 10 scores):")
    high_score_corr = detailed_corr_info[detailed_corr_info['total num_of_user_score'] > 10].sort_values('Correlation', ascending=False).head()
    
    # Display the dataframe with additional details
    st.dataframe(high_score_corr[['Correlation', 'total num_of_user_score', 'Developer', 'Genres']])

else:
    st.warning("Please select a game title from the dropdown to see the correlations.")

# Only use dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
