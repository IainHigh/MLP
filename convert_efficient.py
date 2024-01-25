import json
import pandas as pd
from tqdm import tqdm
import re

dir_to_interactions_json = "goodreads_interactions_dedup.json"

# Define the column names
columns = ["user_id", "book_id", "date_added", "read_at", "started_at"]
batch_size = 20000  # Adjust based on your system's memory capability
date_pattern = re.compile(r"\w{3} (\w{3} \d{2} \d{2}:\d{2}:\d{2}) \-\d{4} (\d{4})")

# Create a new file and write headers if it doesn't exist
with open("goodreads_interactions.csv", "w") as f:
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

with open(dir_to_interactions_json, "r") as f:
    for line in tqdm(f, unit=" lines", desc="Processed"):
        # Parse each line as JSON
        try:
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
                data_accumulator.append(processed_entry)

                # Write in batches
                if len(data_accumulator) >= batch_size:
                    batch_df = pd.DataFrame(data_accumulator)
                    batch_df.to_csv(
                        "goodreads_interactions.csv",
                        mode="a",
                        header=False,
                        index=False,
                    )
                    data_accumulator = []  # Reset the accumulator

        except json.JSONDecodeError:
            print(f"Error decoding JSON for line: {line}")
            continue

# Write any remaining data
if data_accumulator:
    batch_df = pd.DataFrame(data_accumulator)
    batch_df.to_csv("goodreads_interactions.csv", mode="a", header=False, index=False)
