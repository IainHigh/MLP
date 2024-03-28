import pandas as pd  # Dataframes
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from tqdm import tqdm


def process_descriptions(tokens):

    # Create a dictionary to map tags to ones that the lemmatizer will understand.
    tag_map = defaultdict(lambda: "n")  # by default, assume nouns
    tag_map["J"] = "a"  # adjectives
    tag_map["V"] = "v"  # verbs
    tag_map["R"] = "r"  # adverbs

    # Create a lemmatizing object
    lemma = WordNetLemmatizer()

    # Get a list of stopwords
    stops = set(stopwords.words("english"))

    """Process a list of tokens: tag, lemmatize, remove stopwords and short words."""
    # Tag tokens with pos_tagger and convert each tag for the lemmatizer
    tagged_tokens = [(token, tag_map[pos[0]]) for token, pos in pos_tag(tokens)]

    # Lemmatize tokens and filter
    lemmatized = [lemma.lemmatize(word, pos) for word, pos in tagged_tokens]
    return [
        word.lower()
        for word in lemmatized
        if word.lower() not in stops and len(word) > 2
    ]

def add_modified_description(input_file, output_file):
    # Write the titles of the columns to the output file
    with open(output_file, "w") as file:
        
        # isbn,language_code,description,isbn13,book_id,title,num_pages
        headings = [
            "isbn",
            "language_code",
            "description",
            "isbn13",
            "book_id",
            "title",
            "num_pages",
            "modified_description",
        ]
        file.write(",".join(headings) + "\n")
    
    with pd.read_csv(input_file, chunksize=10**4) as reader:        
        for books in reader:
            # Remove books with no description
            books = books.dropna(subset=["description"])
            tqdm.pandas(desc="Tokenizing descriptions")
            modified_description = books["description"].progress_apply(word_tokenize)
            tqdm.pandas(desc="Processing descriptions")
            modified_description = modified_description.progress_apply(process_descriptions)
            tqdm.pandas(desc="Joining descriptions back together")
            modified_description = modified_description.progress_apply(lambda x: " ".join(x))
            books["modified_description"] = modified_description

            books.to_csv(output_file, mode="a", index=False, header=False)

add_modified_description("Datasets/books_filtered_by_language.csv", "Datasets/books_filtered_by_language_modified_desc.csv")