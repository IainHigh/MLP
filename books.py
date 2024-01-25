from tqdm import tqdm

with open("goodreads_books.json", "r") as f:
    for line in tqdm(f, unit=" lines", desc="Processed"):
        print(line)
