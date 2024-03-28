# This python script will plot the corellation between the similarity and the time_to_start for the most recent book each user has read.
# x-axis: time_to_start / average_time_to_start (0-inf)
# y-axis: similarity (0-1)

# Pseuducode:
# For each user:
#     Get dataframe of all books they've finished reading.
#     Sort dataframe by date finished reading.
#     Get users most recent book.
#     Get the similarity of the most recent book with all other books.
#     Get the time_to_start of the most recent book.
#     Save the time_to_start and similarity to a list.
# Repeat for all users.
# Calculate the line of best fit for the time_to_start and similarity.
# Show Graph.

import pandas as pd
import tqdm
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)
import matplotlib.pyplot as plt  # Visualisations
from tqdm import tqdm
import numpy as np

# Define the path to the data
interactions_csv = "Datasets/goodreads_interactions.csv"

book_df = pd.read_csv("goodreads_books_with_modified.csv")

# Define the chunksize for reading the data
chunksize = 10**6

max_points = 1000

tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=0.01)

# Drop the book if either of the following are NaN: modified_description, book_id
book_df = book_df.dropna(subset=["modified_description", "book_id"])

# Convert the book_df book_id to an integer
# If the book_id ends in "X", remove the "X" and convert to an integer
book_df["book_id"] = book_df["book_id"].apply(lambda x: x[:-1] if x[-1] == "X" else x)
book_df["book_id"] = book_df["book_id"].astype(int, errors="ignore")
book_df = book_df[book_df["book_id"].apply(lambda x: x.isnumeric())]
book_df["book_id"] = book_df["book_id"].astype(int, errors="raise")

# dtm = document term matrix
# Calculate the tf-idf matrix for each modified description
descriptions = book_df["modified_description"]
description_dtm = tfidf_vectorizer.fit_transform(descriptions)

point_list = []

counter = 0
progress_bar = tqdm(total=max_points)
with pd.read_csv(interactions_csv, chunksize=chunksize) as reader:
    for chunk in reader:
        for user_id, user_df in chunk.groupby("user_id"):

            # Combine the user dataframe with the book dataframe
            user_books = pd.merge(user_df, book_df, on="book_id")

            if len(user_books) <= 2:
                continue

            counter += 1
            progress_bar.update(1)

            # Convert the date_added, read_at, and started_at to datetime
            user_books["date_added"] = pd.to_datetime(
                user_books["date_added"], format="%Y-%m-%d %H:%M:%S"
            )
            user_books["read_at"] = pd.to_datetime(
                user_books["read_at"], format="%Y-%m-%d %H:%M:%S"
            )
            user_books["started_at"] = pd.to_datetime(
                user_books["started_at"], format="%Y-%m-%d %H:%M:%S"
            )

            most_recent_book = user_books.sort_values(by="read_at").iloc[0]

            # Get the description of the most recent book
            search_description = most_recent_book["modified_description"]
            search_description_vector = tfidf_vectorizer.transform([search_description])

            book_df["similarity"] = cosine_similarity(
                search_description_vector, description_dtm
            ).flatten()

            user_books["similarity"] = pd.merge(user_df, book_df, on="book_id")[
                "similarity"
            ]

            # Remove the most recent book from the dataframe
            user_books = user_books[
                user_books["book_id"] != most_recent_book["book_id"]
            ]

            # Calculate the average book similarity
            similarity = user_books["similarity"].mean()

            # Calculate the time to start of the most recent book
            time_to_start = (
                most_recent_book["started_at"] - most_recent_book["date_added"]
            )

            # Convert from timedelta to days
            time_to_start = time_to_start.total_seconds() / (60 * 60 * 24)

            if time_to_start < 0:
                continue

            # Save the time_to_start and similarity to a list
            point_list.append((time_to_start, similarity))

            if counter >= max_points:
                break
        if counter >= max_points:
            break

    points_array = np.array(point_list)

# Now, assuming your points_array is structured correctly with time_to_start as x and similarity as y:
x = points_array[:, 0]  # Time to start values
y = points_array[:, 1]  # Similarity values

# Calculate the line of best fit
slope, intercept = np.polyfit(x, y, 1)

# Generate y-values for the line of best fit based on x-values
line = slope * x + intercept

# Plot the original scatter plot
plt.scatter(x, y, color="blue", marker=".")

# Plot the line of best fit
plt.plot(x, line, color="red", label="Line of Best Fit")

plt.xlabel("Time to Start (days)")
plt.ylabel("Similarity")
plt.show()
