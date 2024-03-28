# MLP

University of Edinburgh Machine Learning Practical Coursework

# Setup

1. Download the datasets: https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_interactions_dedup.json.gz https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz
2. After installing unzip the datasets
3. Run preprocessing_scripts/convert_efficient.py to convert the json file to a csv file. This should take roughly 3 hours to run.

# Directory Structure

```
.
├── eddie_scripts # Contains the scripts that run the experiments on Eddie
    ├── eddie-script.sh # Runs random_forest/descision_forest_BERT.py on the ECDF cluster.
    ├── experiment1.sh # Once again runs the same file as above but with different parameters.
├── jupyter_notebook_experiments # Contains the jupyter notebooks that were used to evaluate the dataset
    ├── corellation_exp.py # plot the corellation between the similarity and the time_to_start for the most recent book each user has read.
    ├── rnn.ipynb # A simple RNN model in a Jupyter Notebook to debug.
    ├── test.ipynb # Basic dataset exploration - used to verify the dataset has been loaded correctly.
    ├── tf-idf.ipynb # A simple TF-IDF model in a Jupyter Notebook to debug.
    ├── tf-idf.py # Similar to above but in a python script.
├── Papers # Contains some of the original papers that were used as inspiration for the project.
    ├── Goodreads.pdf # Paper which first collated the dataset we are using.
    ├── Noah's BG Chapter.pdf
    ├── TemPEST Paper.pdf
├── preprocessing_scripts # Contains the scripts that were used to preprocess the dataset
    ├── convert_efficient.py # Converts the json file to a csv file.
    ├── convert.py # Converts the json file to a csv file. This is slower than the above script.
    ├── modiffy_description.py # Creates a modified description (All words lower case, dictionary form, no stop words, etc.)
├── random_forest # Contains the random forest models
    ├── decision_forest_BERT.py # Random forest model using BERT embeddings.
    ├── decision_forest_ELMO.py # Random forest model using ELMO embeddings.
    ├── decision_forest.py # Random forest model using TF-IDF embeddings.
├── results
    ├── All_results.txt # Contains all the results from the experiments.
    ├── decision_forest_elmo_output.o4138566
    ├── decision_forest_output.txt
    ├── RNN.o41477875
    ├── RNN.o41597200
    ├── RNN.o41689730
    ├── RNN.o41899808
├── rnn # Contains the basic RNN model (no elmo embeddings
    ├── preprocessing.py # Preprocesses the data for the RNN model.
    ├── rnn-mlp.yml # Specified environment for the RNN model (not used)
    ├── rnn.py # The RNN model
    ├── utils.py # Contains some utility functions for the RNN model.
├── RNN-Elmo # Contains the RNN model with ELMO embeddings
    ├── preprocessing.py # Preprocesses the data for the RNN model.
    ├── rnn.py # The RNN model
    ├── utils.py # Contains some utility functions for the RNN model.
```
