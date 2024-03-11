# MLP

University of Edinburgh Machine Learning Practical Coursework

# Setup

1. Download the datasets: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions_dedup.json.gz https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz
2. After installing unzip the datasets
3. Run convert_efficient.py to convert the json file to a csv file. This should take roughly 3 hours to run.
4. Run test.ipynb to test the data loads correctly.

# RNN 
- rnn.py has the different models
- preprocessing.py has the preprocessing needed
- rnn.yml is the config file where we specify the parameters
- run_rnn.py runs the rnn pipeline

# Run RNN

- Either run run_rnn.py rnn.yml, or
- run rnn_old.py
