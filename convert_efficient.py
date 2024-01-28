import json
import pandas as pd
from tqdm import tqdm
import re

batch_size = 20000
date_pattern = re.compile(r"\w{3} (\w{3} \d{2} \d{2}:\d{2}:\d{2}) \-\d{4} (\d{4})")


def convert_to_datetime(date_str):
    try:
        match = date_pattern.search(date_str)
        if match:
            date_main = f"{match.group(1)} {match.group(2)}"
            return pd.to_datetime(date_main, format="%b %d %H:%M:%S %Y")
    except ValueError:
        return None


def convert_json_to_csv(input_json_file, output_csv_file, columns, date_fields=[]):
    print(f"\n<=== Converting {input_json_file} ===>")
    with open(output_csv_file, "w") as f:
        f.write(",".join(columns) + "\n")

    data_accumulator = []
    with open(input_json_file, "r") as f:
        for line in tqdm(f, unit=" lines", desc="Processed"):
            entry = json.loads(line.strip())
            if all(entry.get(col) for col in columns):
                entry = {
                    col: convert_to_datetime(entry[col])
                    if col in date_fields
                    else entry[col]
                    for col in columns
                }

                # Check that all date fields are not None
                if not all(entry.get(col) for col in date_fields):
                    continue

                data_accumulator.append(entry)
                if len(data_accumulator) >= batch_size:
                    pd.DataFrame(data_accumulator).to_csv(
                        output_csv_file, mode="a", header=False, index=False
                    )
                    data_accumulator = []
    if data_accumulator:
        pd.DataFrame(data_accumulator).to_csv(
            output_csv_file, mode="a", header=False, index=False
        )


def display_first_n_rows(csv_file, n):
    with open(csv_file, "r") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= n:
                break


def main():
    interactions_columns = ["user_id", "book_id", "date_added", "read_at", "started_at"]
    book_columns = [
        "isbn",
        "language_code",
        "description",
        "isbn13",
        "book_id",
        "title",
        "num_pages",
    ]

    convert_json_to_csv(
        "Datasets/goodreads_interactions_dedup.json",
        "Datasets/goodreads_interactions.csv",
        interactions_columns,
        date_fields=["date_added", "read_at", "started_at"],
    )
    convert_json_to_csv(
        "Datasets/goodreads_books.json", "Datasets/goodreads_books.csv", book_columns
    )
    display_first_n_rows("Datasets/goodreads_interactions.csv", 5)
    display_first_n_rows("Datasets/goodreads_books.csv", 5)


if __name__ == "__main__":
    main()
