
import spacy
import random

import csv
import numpy as np
import pandas as pd
from collections import Counter



# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

mod_book_csv_path = "./Datasets/books_filtered_by_language_modified_desc.csv"


def preprocess_data(train_path, val_path, test_path, books_path):

    print("\n\nPreprocessing data...")

    training_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    

    # Calculate the time_to_start for each interaction
    training_df["time_to_start_seconds"] = pd.to_datetime(training_df["started_at"]) - pd.to_datetime(training_df["date_added"])
    val_df["time_to_start_seconds"] = pd.to_datetime(val_df["started_at"]) - pd.to_datetime(val_df["date_added"])
    test_df["time_to_start_seconds"] = pd.to_datetime(test_df["started_at"]) - pd.to_datetime(test_df["date_added"])

    training_df["time_to_start_seconds"] = training_df["time_to_start_seconds"].dt.total_seconds()
    val_df["time_to_start_seconds"] = val_df["time_to_start_seconds"].dt.total_seconds()
    test_df["time_to_start_seconds"] = test_df["time_to_start_seconds"].dt.total_seconds()

    # Remove time_to_start_seconds that are less than or equal to 0
    training_df = training_df[training_df["time_to_start_seconds"] > 0]
    val_df = val_df[val_df["time_to_start_seconds"] > 0]
    test_df = test_df[test_df["time_to_start_seconds"] > 0]

    # Read the book data csv
    #book_data = pd.read_csv(mod_book_csv_path)

    # drop rows that have null in modified description column
    books_df = pd.read_csv(books_path)
    books_df = books_df[books_df['modified_description'].notna()]

    # Tokenization
    tok = spacy.load("en_core_web_sm")
    def tokenize (text):
        words = text.split()
        return [word.strip() for word in words if word.strip()]

    # Count number of occurences of each word
    counts = Counter()
    for text in list(books_df['modified_description']):
      counts.update(tokenize(text))
      # counts.update(text)

    # Deleting infrequent words
    print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:",len(counts.keys()))

    # Creating vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    #vocab_size = len(words)  # needed later in pytorch training loop

    def encode_sentence(text, vocab2index, N=200):
        tokenized = tokenize(text)
        encoded = [vocab2index.get(word, vocab2index["UNK"]) for word in tokenized]
        length = min(N, len(encoded))
        encoded_str = ','.join(map(str, encoded[:length]))  # Convert encoded array to comma-separated string
        return encoded_str, length

    # Apply encoding function to each element in 'modified_description'
    #encoded_data = books_df['modified_description'].apply(lambda x: encode_sentence(x, vocab2index))
    encoded_data = books_df['modified_description'].apply(lambda x: encode_sentence(x, vocab2index)[0])


    # Separate the arrays and lengths into separate columns
    books_df['encoded'] = encoded_data.apply(lambda x: x[0])  # Extract arrays
    books_df['encoded_length'] = encoded_data.apply(lambda x: x[1])  # Extract lengths

    #book_data['encoded'] = book_data['modified_description'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))

    # Merge the book data with the training and test data
    training_df = training_df.merge(books_df, how="left", on="book_id")
    val_df = val_df.merge(books_df, how="left", on="book_id")
    test_df = test_df.merge(books_df, how="left", on="book_id")


     # Convert the language code into an integer
    lang_map = {
        "eng": 0,
        "en-US": 1,
        "en-GB": 2,
    }

    test_df["language_code"] = test_df["language_code"].map(lang_map)
    val_df["language_code"] = val_df["language_code"].map(lang_map)
    training_df["language_code"] = training_df["language_code"].map(lang_map)

    # For each user, calculate the average time_to_start_seconds for all other books they have read
    user_avg_time_to_start = training_df.groupby("user_id")["time_to_start_seconds"].mean()
    user_avg_time_to_start = user_avg_time_to_start.rename("user_avg_time_to_start")
    training_df = training_df.merge(user_avg_time_to_start, how="left", on="user_id")

    user_avg_time_to_start = val_df.groupby("user_id")["time_to_start_seconds"].mean()
    user_avg_time_to_start = user_avg_time_to_start.rename("user_avg_time_to_start")
    val_df = val_df.merge(user_avg_time_to_start, how="left", on="user_id")

    user_avg_time_to_start = test_df.groupby("user_id")["time_to_start_seconds"].mean()
    user_avg_time_to_start = user_avg_time_to_start.rename("user_avg_time_to_start")
    test_df = test_df.merge(user_avg_time_to_start, how="left", on="user_id")

    # Remove unnecessary columns - ones that are not useful for training the model
    training_df = training_df.drop(
        columns=["isbn", "isbn13", "date_added", "read_at", "started_at", "title", "description"]
    )
    val_df = val_df.drop(
        columns=["isbn", "isbn13", "date_added", "read_at", "started_at", "title", "description"]
    )
    test_df = test_df.drop(
        columns=["isbn", "isbn13", "date_added", "read_at", "started_at", "title", "description"]
    )

    training_df = training_df.dropna()
    val_df = val_df.dropna()
    test_df = test_df.dropna()

    with open('./counts.csv','w') as f:
        writer=csv.writer(f)
        for key, value in counts.items():
            writer.writerow([key] + [value])


    return training_df, val_df, test_df, words, counts


