import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars
import math
import matplotlib.pyplot as plt

import torch
#from sentence_transformers import SentenceTransformer, util

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

book_data_csv_file = "./Datasets/books_filtered_by_language_modified_desc.csv"
train_data_csv_file = "./Datasets/train_interactions.csv"
val_data_csv_file = "./Datasets/validate_interactions.csv"
preprocessed = False # Set to True if the CSVs above have already been preprocessed by the function below
bert_preprocessed = True  # Set to True if the BERT embeddings have already been generated and saved to the book data CSV

def preprocess_data(training_df, test_df):
    
    print("\n\nPreprocessing data...")

    # Calculate the time_to_start for each interaction
    training_df["time_to_start_seconds"] = pd.to_datetime(
        training_df["started_at"]
    ) - pd.to_datetime(training_df["date_added"])
    test_df["time_to_start_seconds"] = pd.to_datetime(
        test_df["started_at"]
    ) - pd.to_datetime(test_df["date_added"])

    training_df["time_to_start_seconds"] = training_df[
        "time_to_start_seconds"
    ].dt.total_seconds()
    test_df["time_to_start_seconds"] = test_df[
        "time_to_start_seconds"
    ].dt.total_seconds()

    # Remove time_to_start_seconds that are less than or equal to 0
    training_df = training_df[training_df["time_to_start_seconds"] > 0]
    test_df = test_df[test_df["time_to_start_seconds"] > 0]

    # Read the book data csv
    book_data = pd.read_csv(book_data_csv_file)
    
    if not bert_preprocessed:
        print("Generating BERT embeddings...")
        
        #model = SentenceTransformer("all-MiniLM-L6-v2")

        #model.max_seq_length = 512
        
        all_embeddings = []
        descriptions = []
        with tqdm(total=len(book_data), desc="Processing descriptions") as pbar:
            for description in book_data['modified_description'].fillna(''):
                descriptions.append(description)
                pbar.update(1)
        
        for description in tqdm(descriptions, desc="Generating BERT embeddings"):
            with torch.no_grad():
                current_embedding = model.encode([description])
                all_embeddings.extend(current_embedding)
    
        # Add embeddings as a new column in the DataFrame
        book_data['bert_embeddings'] = all_embeddings

        # save the book data to a csv
        book_data.to_csv("./Datasets/books_filtered_by_language_modified_desc_bert.csv", index=False)
        
    if bert_preprocessed:
        book_data = pd.read_csv("./Datasets/books_filtered_by_language_modified_desc_bert.csv")
        

    # Merge the book data with the training and test data
    training_df = training_df.merge(book_data, how="left", on="book_id")
    test_df = test_df.merge(book_data, how="left", on="book_id")

    # Define a function to calculate mean similarity for a user-book pair
    def mean_similarity(book_id, user_read_books):
        book_index = book_data.index[book_data['book_id'] == book_id].tolist()
        if not book_index:
            return 0
        embedding_str = book_data.loc[book_index[0], 'bert_embeddings']
        book_embedding = np.fromstring(embedding_str.replace('[', '').replace(']', ''), dtype=float, sep=' ')
        
        user_book_indices = book_data.index[book_data['book_id'].isin(user_read_books)].tolist()
        if not user_book_indices:
            return 0
        
        # user_book_embeddings = np.array(book_data.loc[user_book_indices, 'bert_embeddings'].tolist())
        user_book_embeddings_strs = book_data.loc[user_book_indices, 'bert_embeddings'].tolist()
        user_book_embeddings = np.array([np.fromstring(x.replace('[', '').replace(']', ''), dtype=float, sep=' ') for x in user_book_embeddings_strs])
        
        cosine_similarities = cosine_similarity([book_embedding], user_book_embeddings)
        return np.mean(cosine_similarities)

    # for description in e(descriptions, desc="Generating BERT embeddings"):
    # Calculate mean BERT similarity for each user-book pair in training and test sets
    for df in [training_df, test_df]:
        # Get a list of books each user has read
        user_books = df.groupby('user_id')['book_id'].apply(list)
        # Apply mean_similarity function to each row
        tqdm.pandas()
        df['bert_similarity'] = df.progress_apply(lambda row: mean_similarity(row['book_id'], user_books.get(row['user_id'], [])), axis=1)
        tqdm.pandas().close()

    # Convert the language code into an integer
    lang_map = {
        "eng": 0,
        "en-US": 1,
        "en-GB": 2,
    }

    test_df["language_code"] = test_df["language_code"].map(lang_map)
    training_df["language_code"] = training_df["language_code"].map(lang_map)

    # # For each user, calculate the average time_to_start_seconds for all other books they have read
    user_avg_time_to_start = training_df.groupby("user_id")["time_to_start_seconds"].mean()
    user_avg_time_to_start = user_avg_time_to_start.rename("user_avg_time_to_start")
    training_df = training_df.merge(user_avg_time_to_start, how="left", on="user_id")
    
    user_avg_time_to_start = test_df.groupby("user_id")["time_to_start_seconds"].mean()
    user_avg_time_to_start = user_avg_time_to_start.rename("user_avg_time_to_start")
    test_df = test_df.merge(user_avg_time_to_start, how="left", on="user_id")
    
    # # Remove unnecessary columns - ones that are not useful for training the model
    training_df = training_df.drop(
        columns=["isbn", "isbn13", "date_added", "read_at", "started_at", "title", "book_id", "user_id", "modified_description", "bert_embeddings", "description"]
    )
    test_df = test_df.drop(
        columns=["isbn", "isbn13", "date_added", "read_at", "started_at", "title", "book_id", "user_id", "modified_description", "bert_embeddings", "description"]
    )

    training_df = training_df.dropna()
    test_df = test_df.dropna()
    
    return training_df, test_df


def calculate_MAE(diffs):
    """
    This function will calculate the mean absolute error (MAE) given a list of differences between
    predicted and actual values.

    """
    return sum(abs(diff) for diff in diffs) / len(diffs)


def calculate_RMSE(diffs):
    """
    This function will calculate the root mean squared error (RMSE) given a list of differences between
    predicted and actual values.

    """
    return (sum(diff**2 for diff in diffs) / len(diffs)) ** 0.5


def true_baseline(training_df, test_df):
    """
    This function will calculate the mean time_to_start for all the books in the training set.
    It will then use this mean as the prediction for all the books in the test set.
    """
    # Convert time_to_start from Timedelta to total seconds for simplicity
    mean_time_to_start_seconds = training_df["time_to_start_seconds"].mean()

    # Create a Series of predictions with the same index as the test_df
    predictions = pd.Series(
        [mean_time_to_start_seconds] * len(test_df), index=test_df.index
    )

    # Calculate differences
    actual = test_df["time_to_start_seconds"]

    # Calculate differences, convert from seconds to days, and print the MAE and RMSE
    diffs = predictions - actual
    diffs = diffs / (24 * 60 * 60)
    print("True Baseline MAE:", calculate_MAE(diffs))
    print("True Baseline RMSE:", calculate_RMSE(diffs))


def random_forest_baseline(training_df, test_df, cols_to_drop=None):
    
    # Drop the columns that we are choosing not to use for this model
    if cols_to_drop:
        training_df = training_df.drop(columns=cols_to_drop)
        test_df = test_df.drop(columns=cols_to_drop)
        
    # Print the columns that are being used for the model
    print("\n\nColumns being used for the model:\n", training_df.columns)
    print("\n\nSample data:")
    print(training_df.head(10))
    
    # Prepare the features (X) and target (y) for training
    X_train = training_df.drop(columns=["time_to_start_seconds"])
    y_train = training_df["time_to_start_seconds"]

    X_test = test_df.drop(columns=["time_to_start_seconds"])
    y_test = test_df["time_to_start_seconds"]

    # Initialize and train the RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate differences, convert from seconds to days, and print the MAE and RMSE
    diffs = predictions - y_test
    diffs = diffs / (24 * 60 * 60)
    print("Random Forest MAE:", calculate_MAE(diffs))
    print("Random Forest RMSE:", calculate_RMSE(diffs))


def main():
   
    print("starting")
    training_df = pd.read_csv(train_data_csv_file)
    test_df = pd.read_csv(val_data_csv_file)

    if not preprocessed:
        print("preprocessing")
        training_df, test_df = preprocess_data(training_df, test_df)
        training_df.to_csv("./Datasets/train_interactions_preprocessed.csv", index=False)
        test_df.to_csv("./Datasets/validate_interactions_preprocessed.csv", index=False)

    # Calculate the true baseline
    true_baseline(training_df, test_df)

    # Calculate the random forest baseline
    print("Testing with all columns")
    random_forest_baseline(training_df, test_df, cols_to_drop=None)
    
    print("Testing without bert_similarity column")
    random_forest_baseline(training_df, test_df, cols_to_drop=["bert_similarity"])
    
    print("Testing without user_avg_time_to_start column")
    random_forest_baseline(training_df, test_df, cols_to_drop=["user_avg_time_to_start"])
    
    print("Testing without num_pages column")
    random_forest_baseline(training_df, test_df, cols_to_drop=["num_pages"])
    
    print("done")


if __name__ == "__main__":
    main()