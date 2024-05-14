# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the datasets
dataset_path = 'dataset.csv'
movie_titles_path = 'movieIdTitles.csv'

# Read the datasets
df = pd.read_csv(dataset_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movie_titles = pd.read_csv(movie_titles_path)

# Display the first few rows of each dataset to understand their structure
print(df.head())
print(movie_titles.head())


#===================================================================================================================================

def main():

    # Set the page title
    st.title('Interactive Movie Recommendation System')
    st.markdown("""
                Today digital age, the recommender systems play a pivotal role in enhancing user engagement 
             and satisfaction on online platforms, particularly those centered around content consumption. 
             These systems leverage sophisticated algorithms and user data to suggest items that align with 
             individual preferences, thereby creating personalized experiences. This project focuses on the 
             development of a movie recommender system, aiming to guide users towards films that resonate with 
             their tastes and interests
                """
            )

    # Load the data
    dataset_path = 'dataset.csv'
    movie_titles_path = 'movieIdTitles.csv'

    # Read the datasets
    df = pd.read_csv(dataset_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movie_titles = pd.read_csv(movie_titles_path)

    # Merge the dataset with movie titles
    df = pd.merge(df, movie_titles, on='item_id')

    # Drop duplicates
    df.drop_duplicates(subset=['user_id', 'title'], inplace=True)

    # Calculate mean ratings and number of ratings for each movie
    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['numOfRatings'] = pd.DataFrame(df.groupby('title')['rating'].count())

    # Create the recommendation system
    moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

    # Function to get movie recommendations
    def get_recommendations(movie_name, num_recommendations=5):
        movie_user_ratings = moviemat[movie_name]
        similar_to_movie = moviemat.corrwith(movie_user_ratings)
        corr_to_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
        corr_to_movie.dropna(inplace=True)
        corr_to_movie = corr_to_movie.join(ratings['numOfRatings'])
        recommendations = corr_to_movie[corr_to_movie['numOfRatings'] > 100].sort_values('Correlation', ascending=False).head(num_recommendations)
        return recommendations.index

    # Sidebar for selecting a movie
    selected_movie = st.sidebar.text_input('Enter a movie name:')

    if selected_movie:
        # Display the selected movie's details
        st.write('### Movie Details')
        st.write(movie_titles[movie_titles['title'] == selected_movie])

        # Get recommendations for the selected movie
        recommended_movies = get_recommendations(selected_movie)
        st.write(f'### Recommendations for {selected_movie}')
        st.write(recommended_movies)
        st.markdown("")
        st.markdown("")

        # Interactive visualizations for each recommended movie
        for movie in recommended_movies:
            st.write(f'## {movie}')

            # Distribution of ratings
            st.write('### Distribution of Ratings')
            movie_ratings = df[df['title'] == movie]['rating']
            st.bar_chart(movie_ratings.value_counts())
            

            # Number of ratings
            num_ratings = ratings.loc[movie]['numOfRatings']
            st.bar_chart(pd.DataFrame({'numOfRatings': [num_ratings]}, index=[movie]))
            

            # Joint plot of ratings and number of ratings
            st.write('### Joint Plot of Ratings and Number of Ratings')
            sns.jointplot(x='rating', y='numOfRatings', data=ratings)
            st.pyplot(plt)
            st.markdown("")
            st.markdown("")

if __name__ == "__main__":
    main()