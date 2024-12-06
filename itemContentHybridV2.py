import pandas as pd
import gzip
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import datetime

# Step 1: Load and Parse the Data
def parse(path):
    with gzip.open(path, 'rb') as g:
        for line in g:
            yield json.loads(line)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# Load data
df = getDF('Luxury_Beauty_5.json.gz')

# Step 2: Data Preparation
# Keep only the relevant columns and content features
df = df[['reviewerID', 'asin', 'overall', 'reviewTime', 'style', 'reviewText']]

# Handle duplicates by specifying aggregation functions for each column
df_grouped = df.groupby(['reviewerID', 'asin'], as_index=False).agg({
    'overall': 'mean',
    'reviewTime': 'first',  # Handle review time
    'style': lambda x: ','.join(x.dropna().astype(str)),  # Concatenate styles
    'reviewText': lambda x: ' '.join(x.dropna().astype(str))  # Concatenate review texts
})


# Now, proceed with df_grouped instead of df
df = df_grouped

# Continue with the rest of your processing...
# Group by each user (reviewerID)
user_groups = df.groupby('reviewerID')

# Split the data into training and testing sets
train_data = []
test_data = []

np.random.seed(42)  # For reproducibility
for _, group in user_groups:
    ratings = group.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(ratings) * 0.8)  # 80% train, 20% test
    if len(ratings[:split_idx]) > 0 and len(ratings[split_idx:]) > 0:
        train_data.append(ratings[:split_idx])
        test_data.append(ratings[split_idx:])

# Concatenate all user data to form the final training and testing datasets
train_df = pd.concat(train_data).reset_index(drop=True)
test_df = pd.concat(test_data).reset_index(drop=True)

global_mean = train_df['overall'].mean()

user_means = train_df.groupby('reviewerID')['overall'].mean().to_dict()

# Compute individual item means
item_means = train_df.groupby('asin')['overall'].mean().to_dict()

# Step 3: Implementing Item-Item Collaborative Filtering
# Create a pivot table for training data (users x items matrix)
train_pivot = train_df.pivot(index='reviewerID', columns='asin', values='overall').fillna(0)

# Calculate item-item similarity matrix using cosine similarity
item_similarity = cosine_similarity(train_pivot.T)
item_similarity_df = pd.DataFrame(item_similarity, index=train_pivot.columns, columns=train_pivot.columns)

# Step 4: Incorporate Content-Based Filtering
# Aggregate content features per item
content_df = df.groupby('asin').agg({
    'reviewText': lambda x: ' '.join(x.dropna().astype(str)),
    'style': lambda x: ','.join([str(s) for s in x.dropna().astype(str)]),
    'reviewTime': 'first'  # Using the first review time as a feature
}).reset_index()

# Process reviewText using TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=500)  # Limit features to manage computational load
content_df['reviewText'] = content_df['reviewText'].fillna('')
tfidf_matrix = tfidf.fit_transform(content_df['reviewText'])

# Process style using One-Hot Encoding
# First, extract unique style attributes
def extract_styles(style_str):
    if not style_str:
        return []
    styles = style_str.split(',')
    styles = [s.strip() for s in styles if s.strip()]
    return styles

content_df['style_list'] = content_df['style'].apply(extract_styles)

# Flatten the list of styles and fit the encoder
all_styles = set(style for sublist in content_df['style_list'] for style in sublist)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)  # Updated parameter
encoder.fit(np.array(list(all_styles)).reshape(-1,1))

# Transform the style lists
def encode_styles(styles):
    if not styles:
        return csr_matrix((1, len(encoder.categories_[0])), dtype=int)
    return encoder.transform(np.array(styles).reshape(-1,1)).sum(axis=0)

style_encoded = content_df['style_list'].apply(encode_styles)
style_sparse = csr_matrix(np.vstack([s.A for s in style_encoded]))


# Process reviewTime by extracting year and month
def extract_date_features(date_str):
    try:
        # Assuming format like "01 5, 2018"
        date = datetime.datetime.strptime(date_str, "%m %d, %Y")
        return [date.year, date.month]
    except:
        return [0, 0]

date_features = content_df['reviewTime'].apply(extract_date_features).tolist()
date_df = pd.DataFrame(date_features, columns=['review_year', 'review_month'])

# Normalize date features
date_df['review_year'] = date_df['review_year'].fillna(0)
date_df['review_month'] = date_df['review_month'].fillna(0)
date_matrix = csr_matrix(date_df.values)

# Combine all content features
content_features = hstack([tfidf_matrix, style_sparse, date_matrix])

# Calculate content-based similarity matrix using cosine similarity
content_similarity = cosine_similarity(content_features)
content_similarity_df = pd.DataFrame(content_similarity, index=content_df['asin'], columns=content_df['asin'])

# Step 5: Combine Collaborative and Content-Based Similarities
# Define weights for CF and CBF
alpha = 0  # Weight for Collaborative Filtering
beta = 1   # Weight for Content-Based Filtering

# Ensure that the indices match
common_asins = train_pivot.columns.intersection(content_similarity_df.index)
train_pivot = train_pivot[common_asins]
item_similarity_df = item_similarity_df.loc[common_asins, common_asins]
content_similarity_df = content_similarity_df.loc[common_asins, common_asins]

# Normalize similarity matrices
item_similarity_norm = item_similarity_df / item_similarity_df.max().max()
content_similarity_norm = content_similarity_df / content_similarity_df.max().max()

# Combine similarities
hybrid_similarity_df = alpha * item_similarity_norm + beta * content_similarity_norm

# Step 6: Modify the Rating Prediction Function to Use Hybrid Similarity
def predict_rating_hybrid(user, item, k=5):
    if item not in hybrid_similarity_df.index:
        return np.nan  # If the item is unknown, we cannot make a prediction
    
    # Find k most similar items that the user has rated
    similar_items = hybrid_similarity_df[item].sort_values(ascending=False)[1:k+1]
    rated_items = train_pivot.loc[user].loc[similar_items.index]
    rated_items = rated_items[rated_items > 0]  # Keep only items that have been rated
    
    if rated_items.empty:
        return np.nan  # If there are no similar rated items, return NaN
    
    # Compute the weighted sum of ratings
    similarity_sum = similar_items.loc[rated_items.index].sum()
    if similarity_sum == 0:
        return np.nan  # Avoid division by zero
    
    return np.dot(rated_items, similar_items.loc[rated_items.index]) / similarity_sum

# Make predictions for the testing set using hybrid similarity
test_df['predicted_rating'] = test_df.apply(lambda x: predict_rating_hybrid(x['reviewerID'], x['asin']), axis=1)

# Step 7: Calculate MAE and RMSE
# Drop any rows where prediction is NaN (due to cold start problem or missing items)
test_df = test_df.dropna(subset=['predicted_rating'])

# Calculate MAE and RMSE
mae = mean_absolute_error(test_df['overall'], test_df['predicted_rating'])
rmse = np.sqrt(mean_squared_error(test_df['overall'], test_df['predicted_rating']))

# Step 8: Precompute Recommendations Using Hybrid Similarity
def precompute_recommendations_hybrid(train_pivot, hybrid_similarity_df, k=10):
    # Convert training pivot to a sparse matrix
    train_sparse = csr_matrix(train_pivot.values)
    
    # Convert hybrid similarity to a sparse matrix
    hybrid_similarity_sparse = csr_matrix(hybrid_similarity_df.values)
    
    # Predict ratings for all users and all items
    all_predictions = train_sparse.dot(hybrid_similarity_sparse).toarray()
    
    # Normalize by the sum of similarities
    sum_sim = hybrid_similarity_sparse.sum(axis=0).A1  # Sum across rows (matching items)
    # Avoid division by zero
    sum_sim[sum_sim == 0] = 1e-9
    all_predictions = all_predictions / sum_sim[np.newaxis, :]
    
    # Mask out already-rated items
    rated_mask = (train_sparse > 0).toarray()  # Mask of rated items
    all_predictions[rated_mask] = -np.inf  # Ensure rated items are not recommended
    
    # Generate top-k recommendations for each user
    recommendations = {}
    for user_idx, user_id in enumerate(train_pivot.index):
        user_predictions = all_predictions[user_idx]
        # Ensure k does not exceed the number of valid recommendations
        valid_items_mask = user_predictions != -np.inf  # Identify items with valid predictions
        available_items = valid_items_mask.sum()
        top_k = min(k, available_items)
        
        if top_k > 0:
            top_k_items_idx = np.argpartition(user_predictions, -top_k)[-top_k:]
            top_k_items_idx = top_k_items_idx[np.argsort(user_predictions[top_k_items_idx])[::-1]]  # Sort indices
            top_k_items = train_pivot.columns[top_k_items_idx].tolist()  # Map indices back to item IDs
            recommendations[user_id] = top_k_items
        else:
            recommendations[user_id] = []  # No recommendations available for this user
    
    return recommendations


# Precompute recommendations using hybrid similarity
recommendations = precompute_recommendations_hybrid(train_pivot, hybrid_similarity_df, k=10)

# Step 9: Evaluate Recommendations and Print Testing & Recommended Items for Some Users
def evaluate_recommendations_hybrid(recommendations, test_df, k=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    # Set a limit to print for the first few users (for debugging and readability)
    num_users_to_print = 5  # Modify this number to print more or fewer users' recommendations

    # Get the list of users in the test set
    test_users = test_df['reviewerID'].unique()

    for user, rec_list in recommendations.items():
        # Only evaluate users who are in the test set
        if user not in test_users:
            continue

        # Get the actual items the user purchased in the test set
        test_items = test_df[test_df['reviewerID'] == user]['asin'].values

        # Print relevant test items and recommended items for debugging
        if num_users_to_print > 0:
            print(f"User {user} - Testing Items: {test_items}")
            print(f"User {user} - Recommended Items: {rec_list}")
            print("-" * 50)
            num_users_to_print -= 1

        # Calculate Precision
        hits = len(set(rec_list) & set(test_items))
        precision = hits / k if k > 0 else 0
        precision_list.append(precision)

        # Calculate Recall
        recall = hits / len(test_items) if len(test_items) > 0 else 0
        recall_list.append(recall)

        # Calculate NDCG
        dcg = sum([1 / np.log2(idx + 2) for idx, item in enumerate(rec_list) if item in test_items])
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(test_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    # Compute average metrics
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0

    # Compute F-measure
    f_measure = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return avg_precision, avg_recall, f_measure, avg_ndcg

# Evaluate recommendations
avg_precision, avg_recall, f_measure, avg_ndcg = evaluate_recommendations_hybrid(recommendations, test_df, k=10)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Print evaluation metrics
print(f"Precision: {avg_precision}")
print(f"Recall: {avg_recall}")
print(f"F-measure: {f_measure}")
print(f"NDCG: {avg_ndcg}")
