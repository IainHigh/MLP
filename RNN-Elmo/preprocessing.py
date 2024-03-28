import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm  # Import tqdm for progress bars
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import torch

mod_book_csv_path = "/exports/eddie/scratch/s2062378/MLP_Dataset/books_filtered_by_language_modified_desc.csv"

def preprocess_data(training_df, val_df, test_df, books_df):

    print("\n\nPreprocessing data...")

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

    # drop rows that have null in modified description column
    books_df = books_df[books_df['modified_description'].notna()]

    # Only use the first 1000 rows for testing
    # books_df = books_df.head(1000)

    all_embeddings = []
    tqdm.pandas(desc="Calculating ELMO embeddings")
    # books_df['elmo_embeddings'] = books_df['modified_description'].progress_apply(lambda desc: calculate_elmo_embedding(desc, elmo))
    elmo = ElmoEmbedder(cuda_device=0)
    for description in tqdm(books_df['modified_description'].fillna(''), desc="Generating ELMO embeddings"):
        with torch.no_grad():
            tokens = description.split()
            vectors = elmo.embed_sentence(tokens)
            embedding = np.mean(np.mean(vectors, axis=0), axis=0)
            all_embeddings.append(embedding)
    books_df['elmo_embeddings'] = all_embeddings
            
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
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
        #enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in text])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    # Apply encoding function to each element in 'modified_description'
    tqdm.pandas(desc="Encoding sentences")
    encoded_data = books_df['modified_description'].progress_apply(lambda x: encode_sentence(x, vocab2index))

    # Separate the arrays and lengths into separate columns
    books_df.loc[:, 'encoded'] = encoded_data.apply(lambda x: x[0])
    books_df.loc[:, 'encoded_length'] = encoded_data.apply(lambda x: float(x[1]))

    #book_data['encoded'] = book_data['modified_description'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))

    # Merge the book data with the training and test data
    training_df = training_df.merge(books_df, how="left", on="book_id")
    val_df = val_df.merge(books_df, how="left", on="book_id")
    test_df = test_df.merge(books_df, how="left", on="book_id")

    def mean_similarity(book_id, user_read_books):
        book_index = books_df.index[books_df['book_id'] == book_id].tolist()
        if not book_index:
            return 0
        book_embedding = books_df.loc[book_index[0], 'elmo_embeddings']
        
        user_book_indices = books_df.index[books_df['book_id'].isin(user_read_books)].tolist()
        if not user_book_indices:
            return 0
        
        user_book_embeddings = np.array(books_df.loc[user_book_indices, 'elmo_embeddings'].tolist())     
        
        if np.isnan(book_embedding).any() or np.isnan(user_book_embeddings).any() or not user_book_embeddings.any():
            return 0
        
        cosine_similarities = cosine_similarity([book_embedding], user_book_embeddings)
        return np.mean(cosine_similarities)

    # Calculate the elmo similarity for each interaction
    for df in [training_df, val_df, test_df]:
        # Get a list of books each user has read
        user_books = df.groupby('user_id')['book_id'].apply(list)
        # Apply mean_similarity function to each row
        desc = "Calculating ELMO similarity for training set" if df is training_df else "Calculating ELMO similarity for test set"
        tqdm.pandas(desc=desc)
        df['elmo_similarity'] = df.progress_apply(lambda row: mean_similarity(row['book_id'], user_books.get(row['user_id'], [])), axis=1)

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

    return training_df, val_df, test_df, words