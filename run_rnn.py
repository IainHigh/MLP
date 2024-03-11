import yaml
import sys
import os
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt

from rnn import *
from preprocessing import preprocess_data

SEED = 9001
np.random.seed(SEED)

if __name__ == "__main__":
    with open(sys.argv[1]) as file:
        params = yaml.safe_load(file)

    mod_books_df = pd.read_csv(params['modified_book_data_path'])

    if params['preprocessing']['run']:
        processed_train_df, processed_val_df, processed_test_df, words = preprocess_data(params['preprocessing'])
    else:
        print("Skipping preprocessing")    

        processed_train_df = pd.read_csv(params['processed_train_data_path'])
        processed_test_df = pd.read_csv(params['processed_test_data_path'])
        processed_val_df = pd.read_csv(params['processed_val_data_path'])

        vocab_size = params['vocab_size']
        

    X_train = list(processed_train_df['encoded'])
    X_train_enc_len = list(processed_train_df['encoded_length'])
    y_train = list(processed_train_df['time_to_start_seconds'])

    # get validation set
    X_valid = list(processed_val_df['encoded'])
    X_valid_enc_len = list(processed_val_df['encoded_length'])
    y_valid = list(processed_val_df['time_to_start_seconds'])

    # get test set
    X_test = list(processed_test_df['encoded'])
    X_test_enc_len = list(processed_test_df['encoded_length'])
    y_test = list(processed_test_df['time_to_start_seconds'])    

    train_ds = CommonLitReadabiltyDataset(X_train, y_train, X_train_enc_len)
    valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid, X_valid_enc_len)

    batch_size = params['batch_size']
    embedding_dim = params['embedding_dim']
    epochs = params['epochs']
    learning_rate = params['learning_rate']
    vocab_size = len(words)
    hidden_dim = params['hidden_dim']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    #if params['load_pretrained_weights']:

    if params['model'] == 'LSTM':
        print('Training LSTM model')
        model = LSTM_regr(vocab_size, embedding_dim, hidden_dim)

    if params['model'] == 'LSTM_attention':
        print('Trianing LSTM + attention model')
        model = LSTM_regr_attention(vocab_size, embedding_dim, hidden_dim)
    
    train_model_regr(model, train_dl, val_dl, epochs=epochs, lr=learning_rate)