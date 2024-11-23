import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
ratings_data = {
    'user': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'movie': ['A', 'B', 'A', 'C', 'A', 'B', 'B', 'D', 'C', 'D'],
    'rating': [5, 3, 4, 2, 2, 5, 4, 5, 4, 1]
}

# Convert data to DataFrame
ratings_df = pd.DataFrame(ratings_data)

# Create a user-movie matrix
user_movie_matrix = ratings_df.pivot(index='user', columns='movie', values='rating').fillna(0)

# Compute user similarity using cosine similarity
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Recommendation function
def recommend_movies(user, num_recommendations=2):
    # Find similar users
    similar_users = user_similarity_df[user].sort_values(ascending=False).drop(user)
    
    # Get weighted average of ratings from similar users
    user_ratings = user_movie_matrix.loc[user]
    recommendations = pd.Series(dtype='float64')
    for other_user, similarity in similar_users.items():
        other_user_ratings = user_movie_matrix.loc[other_user]
        weighted_ratings = other_user_ratings * similarity
        recommendations = recommendations.add(weighted_ratings, fill_value=0)
    
    # Filter out movies the user has already rated
    recommendations = recommendations[user_ratings == 0]
    
    # Return top recommendations
    return recommendations.sort_values(ascending=False).head(num_recommendations)

# Recommend movies for User 1
print("Recommended movies for User 1:")
print(recommend_movies(1))
