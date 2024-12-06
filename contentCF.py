import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ndcg_score
from scipy.sparse import csr_matrix
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import gzip  # Importing gzip to handle .gz files

warnings.filterwarnings('ignore')

# 1. Load the JSON data into a pandas DataFrame
def load_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    return df

# 2. Split the data into training and testing datasets
def train_test_split_per_user(df, test_size=0.2, random_state=42):
    train_list = []
    test_list = []
    grouped = df.groupby('reviewerID')
    for user, group in grouped:
        if len(group) < 5:
            # If a user has less than 5 reviews, put all in train to avoid very small test sets
            train_list.append(group)
            continue
        train, test = train_test_split(group, test_size=test_size, random_state=random_state)
        train_list.append(train)
        test_list.append(test)
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df

# 3. Build the content-based model
def build_content_based_model(train_df):
    # Combine reviewText and style into a single text field
    train_df['content'] = train_df['reviewText'].fillna('') + ' ' + train_df['style'].fillna('').apply(lambda x: ' '.join([f"{k} {v}" for k, v in x.items()]) if isinstance(x, dict) else '')
    
    # Create a product profile by aggregating all contents per product
    product_profiles = train_df.groupby('asin')['content'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Vectorize the product profiles using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(product_profiles['content'])
    
    # Create a mapping from asin to index
    asin_to_index = pd.Series(product_profiles.index, index=product_profiles['asin']).to_dict()
    
    return tfidf_matrix, asin_to_index, vectorizer, product_profiles

# 4. Predict ratings for the test set
def predict_ratings(train_df, test_df, tfidf_matrix, asin_to_index):
    # Check if tfidf_matrix is sparse and convert it to dense if needed
    if isinstance(tfidf_matrix, csr_matrix):
        tfidf_matrix = tfidf_matrix.toarray()  # Convert to dense if it's sparse

    # Create user profiles by averaging the TF-IDF vectors of items they've rated
    user_profiles = {}
    user_group = train_df.groupby('reviewerID')
    for user, group in tqdm(user_group, desc="Building user profiles"):
        item_indices = group['asin'].map(asin_to_index)
        item_indices = item_indices.dropna().astype(int)
        if len(item_indices) == 0:
            user_profiles[user] = None
            continue
        user_profile = tfidf_matrix[item_indices].mean(axis=0).flatten()  # Use dense array directly
        user_profiles[user] = user_profile

    # Predict ratings
    y_true = []
    y_pred = []
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Predicting ratings"):
        user = row['reviewerID']
        item = row['asin']
        true_rating = row['overall']
        y_true.append(true_rating)
        
        if item not in asin_to_index or user_profiles.get(user) is None:
            pred_rating = train_df['overall'].mean()
        else:
            item_idx = asin_to_index[item]
            item_vector = tfidf_matrix[item_idx]  # Already dense now
            user_profile = user_profiles[user]
            # Compute cosine similarity
            similarity = np.dot(user_profile, item_vector.T)
            # Normalize similarity
            norm_user = np.linalg.norm(user_profile)
            norm_item = np.linalg.norm(item_vector)
            if norm_user == 0 or norm_item == 0:
                similarity = 0
            else:
                similarity = similarity / (norm_user * norm_item)
            # Assume rating prediction based on similarity
            pred_rating = train_df['overall'].mean() + similarity * (5 - train_df['overall'].mean())
            pred_rating = np.clip(pred_rating, 1, 5)
        y_pred.append(pred_rating)
    
    return y_true, y_pred

# 5. Evaluate predictions
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Compute prediction errors
    errors = np.array(y_true) - np.array(y_pred)
    
    # Plot a histogram of prediction errors
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Error (True Rating - Predicted Rating)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 6. Recommend ten items per user
def generate_recommendations(train_df, test_df, tfidf_matrix, asin_to_index, product_profiles):
    recommendations = {}
    # Create a set of test items per user
    test_items_per_user = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    # Create a set of train items per user
    train_items_per_user = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    
    # Precompute the cosine similarity matrix
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    norms = np.array(tfidf_matrix.power(2).sum(axis=1)).flatten()
    norm_matrix = np.sqrt(norms[:, None] * norms[None, :])
    similarity_matrix = similarity_matrix / norm_matrix
    similarity_matrix[np.isnan(similarity_matrix)] = 0  # Handle division by zero

    for user in tqdm(test_items_per_user.keys(), desc="Generating recommendations"):
        train_items = train_items_per_user.get(user, set())
        test_items = test_items_per_user.get(user, set())
        if not test_items:
            recommendations[user] = []
            continue
        # Compute user profile as the average of train item vectors
        train_indices = [asin_to_index[item] for item in train_items if item in asin_to_index]
        if not train_indices:
            user_profile = np.zeros(tfidf_matrix.shape[1])
        else:
            user_profile = tfidf_matrix[train_indices].mean(axis=0).flatten()  # Ensure it is a dense array
        # Compute scores for test items
        scores = {}
        for item in test_items:
            if item not in asin_to_index:
                scores[item] = 0
                continue
            item_idx = asin_to_index[item]
            item_vector = tfidf_matrix[item_idx].toarray().flatten()  # Convert to dense and flatten
            similarity = np.dot(user_profile, item_vector.T)  # Compute cosine similarity
            norm_user = np.linalg.norm(user_profile)
            norm_item = np.linalg.norm(item_vector)
            if norm_user == 0 or norm_item == 0:
                similarity = 0
            else:
                similarity = similarity / (norm_user * norm_item)
            scores[item] = similarity
        # Recommend top 10 items
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        recommendations[user] = [item for item, score in top_items]
    return recommendations, test_items_per_user

# 7. Evaluate recommendations
def evaluate_recommendations(recommendations, test_items_per_user, k=10):
    precisions = []
    recalls = []
    f_measures = []
    ndcgs = []
    
    for user, recommended in recommendations.items():
        relevant = test_items_per_user.get(user, set())
        if not relevant:
            continue
        recommended_set = set(recommended)
        relevant_set = set(relevant)
        tp = len(recommended_set & relevant_set)
        precision = tp / k
        recall = tp / len(relevant_set)
        if precision + recall == 0:
            f_measure = 0
        else:
            f_measure = 2 * precision * recall / (precision + recall)
        
        # For NDCG, create binary relevance
        y_true = [1 if item in relevant_set else 0 for item in recommended]
        y_score = [1] * len(recommended)  # All recommended items are assumed equally relevant
        
        # Now we compute NDCG for the user's recommended items
        if len(y_true) > 1:  # NDCG requires more than 1 document
            ndcg = ndcg_score([y_true], [y_score])
        else:
            ndcg = 0  # If there's only one document, NDCG is undefined or 0
        
        precisions.append(precision)
        recalls.append(recall)
        f_measures.append(f_measure)
        ndcgs.append(ndcg)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f_measure = np.mean(f_measures) if f_measures else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"F-measure@{k}: {avg_f_measure:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")

# Main function to execute all steps
def main():
    # Path to the JSON file
    file_path = 'Luxury_Beauty_5.json.gz'  # Updated file path
    
    print("Loading data...")
    df = load_data(file_path)
    print(f"Total reviews: {len(df)}")
    
    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split_per_user(df)
    print(f"Training reviews: {len(train_df)}")
    print(f"Testing reviews: {len(test_df)}")
    
    print("Building content-based model...")
    tfidf_matrix, asin_to_index, vectorizer, product_profiles = build_content_based_model(train_df)
    
    print("Predicting ratings...")
    y_true, y_pred = predict_ratings(train_df, test_df, tfidf_matrix, asin_to_index)
    
    print("Evaluating rating predictions...")
    evaluate_predictions(y_true, y_pred)
    
    print("Generating recommendations...")
    recommendations, test_items_per_user = generate_recommendations(train_df, test_df, tfidf_matrix, asin_to_index, product_profiles)
    
    print("Evaluating recommendations...")
    evaluate_recommendations(recommendations, test_items_per_user, k=10)

if __name__ == "__main__":
    main()
