# Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns  

# Set the page title
st.title('Movie Recommendation System')

# Load the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dataset.csv', sep='\t', names=column_names)
movie_titles = pd.read_csv('movieIdTitles.csv')

# Merge the dataset with movie titles
df = pd.merge(df, movie_titles, on='item_id')

# Drop duplicates
df.drop_duplicates(subset=['user_id', 'title'], inplace=True)

# Calculate mean ratings and number of ratings for each movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['numOfRatings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# Add columns for movie recommendations
ratings['First Movie Recommendation'] = np.nan
ratings['Second Movie Recommendation'] = np.nan
ratings['Third Movie Recommendation'] = np.nan
ratings['Fourth Movie Recommendation'] = np.nan

# Sidebar for selecting a movie
selected_movie = st.sidebar.selectbox('Select a Movie:', ratings.index)

# Display the selected movie's details
st.write('### Movie Details')
st.write(movie_titles[movie_titles['title'] == selected_movie])

# Display recommendations for the selected movie
st.write('### Recommendations for', selected_movie)
recommended_movies = ratings.loc[selected_movie, ['First Movie Recommendation', 'Second Movie Recommendation', 'Third Movie Recommendation', 'Fourth Movie Recommendation']]
st.write(recommended_movies)

# Interactive visualization - Distribution of ratings
st.write('### Distribution of Ratings')
selected_movie_ratings = df[df['title'] == selected_movie]['rating']
st.bar_chart(selected_movie_ratings.value_counts())

# Interactive visualization - Distribution of number of ratings
st.write('### Distribution of Number of Ratings')
st.bar_chart(ratings['numOfRatings'])

# Interactive visualization - Joint plot of ratings and number of ratings
st.write('### Joint Plot of Ratings and Number of Ratings')
st.write('This may take a moment...')
st.write(sns.jointplot(x='rating', y='numOfRatings', data=ratings, alpha=0.5).fig)
