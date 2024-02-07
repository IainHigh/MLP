"""
This script takes in a user_id and a book_isbn and returns a dataframe of the previous books read by the user and the tf-idf similarity of the new book description to the previous books read by the user.
"""


import pandas as pd  # Dataframes
import re  # Regular expressions
import nltk
from nltk.tokenize import word_tokenize  # Split text into words
from nltk.corpus import stopwords  # Lists of unimportant words
from collections import (
    Counter,
    defaultdict,
)  # Count word frequency & provide more versatile dicts
from pandas.core.common import flatten  # Collapse lists of lists
from nltk.stem.wordnet import WordNetLemmatizer  # Reduce terms to their root
from nltk import pos_tag  # Tag words with parts of speech
import seaborn as sns  # Visualisations
import matplotlib.pyplot as plt  # Visualisations
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # Convert text to TF-IDF representations
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # Check similarities between vectors
from textwrap import wrap  # format long text
from tqdm import tqdm


def crate_dtm(tfidf_vectorizer, book_dataframe):
    # dtm = document term matrix
    # Calculate the tf-idf matrix for each modified description
    descriptions = book_dataframe["modified_description"]

    return tfidf_vectorizer.fit_transform(descriptions)


def read_book_data(dataset_loc):
    return pd.read_csv(dataset_loc)


def get_book_description(book_dataframe):
    print("Enter description of the book: ")
    return input()

    book = None
    while (book is None) or (book.empty):
        print("Enter a book isbn number: ")
        isbn = input()
        book = book_dataframe.loc[book_dataframe["isbn"] == isbn]
    return book["description"].values[0]


def get_user_dataframe(csv_loc, chunksize=20**5):
    user_df = pd.DataFrame()
    while user_df.empty:
        print("Enter a user id: ")
        user_id = input()

        # Process the file in chunks
        with pd.read_csv(csv_loc, chunksize=chunksize) as reader:
            for chunk in tqdm(reader):
                user_df = pd.concat([user_df, chunk[chunk["user_id"] == user_id]])
    return user_df


def get_wordnet_tags(tokens):
    """Gets wordnet tags from a set of tokens

    Input:
        - a list of string tokens
    Output:
        - a list of (word, tag) tuples
    """
    tag_map = defaultdict(lambda: "n")  # by default, assume nouns
    tag_map["J"] = "a"  # adjectives
    tag_map["V"] = "v"  # verbs
    tag_map["R"] = "r"  # adverbs

    # Tag tokens with pos_tagger
    tagged_tokens = pos_tag(tokens)

    # Convert each tag to a version wordnet can understand
    tagged_tokens = [(token[0], tag_map[token[1][0]]) for token in tagged_tokens]

    return tagged_tokens


def convert_text_to_vector(tfidf_vectorizer, text):
    """Converts a text string into a TFIDF vector

    Input:
        - text (str): a book description
    Output:
        - vector (scipy sparse matrix): a tf-idf vector for the description
    """
    # Clean text
    text = text.lower()
    text = re.sub("[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatize and remove stopwords
    lemma = WordNetLemmatizer()
    text = text.split(" ")
    text = get_wordnet_tags(text)
    text = [lemma.lemmatize(word=word[0], pos=word[1]) for word in text]
    text = [
        word
        for word in text
        if word not in stopwords.words("english") and len(word) > 3
    ]
    text = " ".join(text)

    # Convert the description to a TF-IDF vector
    vector = tfidf_vectorizer.transform([text])

    return vector


def calculate_similarity(description_dtm, search_description_vector, book_dataframe):
    # Use cosine similarity to find the most similar vectors to the test

    similarities = cosine_similarity(
        search_description_vector, description_dtm
    ).flatten()

    book_dataframe["similarity"] = similarities
    return book_dataframe


def get_user_books(user_dataframe, book_dataframe):
    # Get the books read by the user
    user_books = user_dataframe["book_id"].unique()
    user_books = book_dataframe[book_dataframe["book_id"].isin(user_books)]

    return user_books


def main():
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=0.01)

    book_dataframe = read_book_data("Datasets/goodreads_books_modified.csv")
    description_dtm = crate_dtm(tfidf_vectorizer, book_dataframe)

    search_description = get_book_description(book_dataframe)
    search_description_vector = convert_text_to_vector(
        tfidf_vectorizer, search_description
    )

    book_dataframe = calculate_similarity(
        description_dtm, search_description_vector, book_dataframe
    )

    user_dataframe = get_user_dataframe("Datasets/goodreads_interactions.csv")
    user_books = get_user_books(user_dataframe, book_dataframe)

    # Write the results to a csv
    user_books.to_csv("Datasets/user_books.csv")

    # Print the average similarity of the books read by the user
    print(user_books["similarity"].mean())


if __name__ == "__main__":
    main()
