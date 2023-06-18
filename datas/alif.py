
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2)

# Create a user-item matrix
train_matrix = train_data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Calculate item-item similarity matrix using cosine similarity
item_similarities = cosine_similarity(train_matrix.T)

# Function to get similar items based on item similarity matrix
def get_similar_items(item_id, similarity_matrix, num_items):
    similar_items = list(enumerate(similarity_matrix[item_id]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    similar_items = similar_items[:num_items]
    return [item[0] for item in similar_items]

# Function to make recommendations for a user
def recommend_items(user_id, train_matrix, similarity_matrix, num_recommendations):
    user_items = train_matrix.loc[user_id]
    user_items = user_items[user_items > 0].index
    
    similar_items = []
    for item_id in user_items:
        similar_items.extend(get_similar_items(item_id, similarity_matrix, num_recommendations))
    
    unique_items = list(set(similar_items) - set(user_items))
    item_scores = [(item_id, similarity_matrix[user_items[-1]][item_id]) for item_id in unique_items]
    
    item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
    recommendations = [item[0] for item in item_scores[:num_recommendations]]
    return recommendations

# Test the recommendation system
user_id = 'user123'
num_recommendations = 5

recommended_items = recommend_items(user_id, train_matrix, item_similarities, num_recommendations)
print("Recommended items for user", user_id)
for item_id in recommended_items:
    print("Item:", item_id)

