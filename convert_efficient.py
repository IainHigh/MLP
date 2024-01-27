import json
import pandas as pd
from tqdm import tqdm
import re

batch_size = 20000  # Adjust based on your system's memory capability
date_pattern = re.compile(r"\w{3} (\w{3} \d{2} \d{2}:\d{2}:\d{2}) \-\d{4} (\d{4})")


def convert_interactions(input_json_file, output_csv_file):
    print("\n<=== Converting interactions ===>")
    # Define the column names that we want to extract
    columns = ["user_id", "book_id", "date_added", "read_at", "started_at"]

    # Create a new file and write headers if it doesn't exist
    with open(output_csv_file, "w") as f:
        f.write(",".join(columns) + "\n")

    # Function to convert date strings to datetime objects
    def convert_to_datetime(date_str):
        try:
            # Use regex to find the relevant parts of the date string
            match = date_pattern.search(date_str)
            if match:
                # Construct the date string with the relevant parts
                date_main = f"{match.group(1)} {match.group(2)}"
                return pd.to_datetime(date_main, format="%b %d %H:%M:%S %Y")
            else:
                raise ValueError("Invalid date format")
        except ValueError as e:
            print(f"Error converting date '{date_str}': {e}")
            return None

    data_accumulator = []  # Initialize a list to accumulate data

    with open(input_json_file, "r") as f:
        for line in tqdm(f, unit=" lines", desc="Processed"):
            # Parse each line as JSON
            entry = json.loads(line.strip())

            # Check if all date fields are non-null
            if all(entry.get(col) for col in ["date_added", "read_at", "started_at"]):
                # Convert date fields and store in the accumulator
                processed_entry = {
                    key: (
                        convert_to_datetime(entry[key])
                        if key in ["date_added", "read_at", "started_at"]
                        else entry[key]
                    )
                    for key in columns
                }

                # Remove None values from the dictionary
                processed_entry = {
                    key: value
                    for key, value in processed_entry.items()
                    if value is not None
                }

                data_accumulator.append(processed_entry)

                # Write in batches
                if len(data_accumulator) >= batch_size:
                    batch_df = pd.DataFrame(data_accumulator)
                    batch_df.to_csv(
                        output_csv_file,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    data_accumulator = []  # Reset the accumulator

    # Write any remaining data
    if data_accumulator:
        batch_df = pd.DataFrame(data_accumulator)
        batch_df.to_csv(output_csv_file, mode="a", header=False, index=False)


def convert_books(input_json_file, output_csv_file):
    print("\n<=== Converting books ===>")
    columns = [
        "isbn",
        "language_code",
        "description",
        "isbn13",
        "book_id",
        "title",
        "num_pages",
    ]

    with open(output_csv_file, "w") as f:
        f.write(",".join(columns) + "\n")

    data_accumulator = []  # Initialize a list to accumulate data

    with open(input_json_file, "r") as f:
        for line in tqdm(f, unit=" lines", desc="Processed"):
            # Parse each line as JSON
            entry = json.loads(line.strip())

            # Check if all fields are non-null
            if all(entry.get(col) for col in columns):
                data_accumulator.append(entry)

                # Write in batches
                if len(data_accumulator) >= batch_size:
                    batch_df = pd.DataFrame(data_accumulator)
                    batch_df.to_csv(
                        output_csv_file,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    data_accumulator = []  # Reset the accumulator

    # Write any remaining data
    if data_accumulator:
        batch_df = pd.DataFrame(data_accumulator)
        batch_df.to_csv(output_csv_file, mode="a", header=False, index=False)


def display_first_n_rows(csv_file, n):
    with open(csv_file, "r") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= n:
                break


def main():
    convert_interactions(
        "Datasets/goodreads_interactions_dedup.json",
        "Datasets/goodreads_interactions.csv",
    )
    convert_books("Datasets/goodreads_books.json", "Datasets/goodreads_books.csv")
    display_first_n_rows("Datasets/goodreads_interactions.csv", 5)
    display_first_n_rows("Datasets/goodreads_books.csv", 5)


if __name__ == "__main__":
    main()
