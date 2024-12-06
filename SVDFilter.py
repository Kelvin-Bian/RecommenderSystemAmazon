import json
import gzip  # Import gzip to handle .gz files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import accuracy
from collections import defaultdict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df = df[['reviewerID', 'asin', 'overall']]
    return df

# Function to split data into train and test per user
def train_test_split_per_user(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    train_list = []
    test_list = []
    for user, group in df.groupby('reviewerID'):
        if len(group) < 5:
            # If user has less than 5 reviews, put all in train
            train_list.append(group)
            continue
        train, test = train_test_split(group, test_size=test_size, random_state=random_state)
        train_list.append(train)
        test_list.append(test)
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df

# Function to build and train SVD model
def train_svd(train_df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[['reviewerID', 'asin', 'overall']], reader)
    trainset = data.build_full_trainset()
    algo = SVD(random_state=42)
    algo.fit(trainset)
    return algo

# Function to predict ratings for the test set
def predict_ratings(algo, test_df):
    predictions = []
    for _, row in test_df.iterrows():
        pred = algo.predict(row['reviewerID'], row['asin']).est
        predictions.append(pred)
    test_df = test_df.copy()
    test_df['predicted'] = predictions
    return test_df

# Function to calculate MAE, RMSE, and return residuals for histogram
def evaluate_predictions(test_df):
    residuals = test_df['overall'] - test_df['predicted']
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    return mae, rmse, residuals

# Function to plot a histogram of residuals
def plot_residual_histogram(residuals):
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Histogram of Residuals", fontsize=16)
    plt.xlabel("Residuals (Actual - Predicted)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Function to get top N recommendations per user from test set
def get_recommendations(algo, train_df, test_df, N=10):
    recommendations = defaultdict(list)
    
    # Create a set of training items per user to exclude them from recommendations
    train_user_items = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    all_items = set(train_df['asin'].unique()).union(set(test_df['asin'].unique()))
    
    # Iterate over test users to generate recommendations
    for user in tqdm(test_df['reviewerID'].unique(), desc="Generating Recommendations"):
        train_items = train_user_items.get(user, set())
        
        # Items that can be recommended (not in train items)
        candidate_items = all_items - train_items
        
        # Predict ratings for all candidate items
        preds = []
        for item in candidate_items:
            est = algo.predict(user, item).est
            preds.append((item, est))
        
        # Sort items by estimated rating and select top N
        preds.sort(key=lambda x: x[1], reverse=True)
        top_N = [item for item, _ in preds[:N]]
        recommendations[user] = top_N
    
    return recommendations

# Function to calculate Precision, Recall, F1, NDCG
def evaluate_recommendations(recommendations, test_df, N=10):
    # Prepare test items per user
    test_items = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    
    precision_list = []
    recall_list = []
    f1_list = []
    ndcg_list = []
    
    for user, recommended in recommendations.items():
        relevant = test_items.get(user, set())
        if not relevant:
            continue
        recommended_set = set(recommended)
        relevant_recommended = recommended_set & relevant
        precision = len(relevant_recommended) / N
        recall = len(relevant_recommended) / len(relevant)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate DCG
        dcg = 0.0
        for idx, item in enumerate(recommended):
            if item in relevant:
                dcg += 1 / math.log2(idx + 2)  # idx starts at 0
        # Calculate IDCG
        idcg = 0.0
        for idx in range(min(len(relevant), N)):
            idcg += 1 / math.log2(idx + 2)
        ndcg = dcg / idcg if idcg > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ndcg_list.append(ndcg)
    
    # Calculate average metrics
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_ndcg = np.mean(ndcg_list)
    
    return avg_precision, avg_recall, avg_f1, avg_ndcg

def main():
    # Load data
    file_path = 'Luxury_Beauty_5.json.gz'  # Updated to .json.gz
    print("Loading data...")
    df = load_data(file_path)
    print(f"Total reviews: {len(df)}")
    
    # Split data
    print("Splitting data into training and testing sets...")
    train_df, test_df = train_test_split_per_user(df, test_size=0.2)
    print(f"Training reviews: {len(train_df)}")
    print(f"Testing reviews: {len(test_df)}")
    
    # Train SVD model
    print("Training SVD model...")
    algo = train_svd(train_df)
    
    # Predict ratings
    print("Predicting ratings for test set...")
    test_pred_df = predict_ratings(algo, test_df)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    mae, rmse, residuals = evaluate_predictions(test_pred_df)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Plot histogram of residuals
    print("Plotting residuals histogram...")
    plot_residual_histogram(residuals)
    
    # Generate recommendations
    print("Generating recommendations...")
    recommendations = get_recommendations(algo, train_df, test_df, N=10)
    
    # Evaluate recommendations
    print("Evaluating recommendations...")
    precision, recall, f1, ndcg = evaluate_recommendations(recommendations, test_df, N=10)
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")
    print(f"F1-Score@10: {f1:.4f}")
    print(f"NDCG@10: {ndcg:.4f}")
    
    # Print sample user predictions and testing items
    print("Sample user predictions and testing items:")
    sample_users = test_df['reviewerID'].unique()[:5]  # Take 5 sample users
    for user in sample_users:
        print(f"\nUser: {user}")
        
        # Predicted items for the user
        predicted_items = recommendations.get(user, [])
        print(f"Predicted items (Top 10): {predicted_items}")
        
        # Actual items in the test set
        actual_items = test_df[test_df['reviewerID'] == user]['asin'].tolist()
        print(f"Actual items in test set: {actual_items}")

if __name__ == "__main__":
    main()
