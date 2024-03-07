import os
import sys
import math
import nltk
import torch
import random
import string
import datasets
import statistics
import spacy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from pprint import pprint
from tqdm.notebook import tqdm
from abc import ABC, abstractmethod

from nltk.corpus import brown
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchtext.vocab import Vectors
# from transformers import AutoTokenizer
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForMaskedLM

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from utilities import set_seed, set_device
from preprocessing import preprocess_data

SEED = 42
set_seed(seed=SEED)
DEVICE = set_device()

# datasets paths
train_csv_path = "./Datasets/train_interactions.csv"
val_csv_path = "./Datasets/test_interactions.csv"
test_csv_path = "./Datasets/validate_interactions.csv"

book_csv_path = "./Datasets/books_filtered_by_language.csv"
mod_book_csv_path = "./Datasets/books_filtered_by_language_modified_desc.csv"


# get dataframes
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
val_df = pd.read_csv(val_csv_path)
mod_books_df = pd.read_csv(mod_book_csv_path)

# preprocess data
processed_train_df, processed_val_df, processed_test_df, words = preprocess_data(train_df, val_df, test_df, mod_books_df)

# save processed dataframes (colab)
# processed_train_df.to_csv("/content/drive/MyDrive/mlp-project/data/processed_train_df_2.csv")
# processed_val_df.to_csv("/content/drive/MyDrive/mlp-project/data/processed_val_df_2.csv")
# processed_test_df.to_csv("/content/drive/MyDrive/mlp-project/data/processed_test_df_2.csv")

# save processed dataframes 
processed_train_df.to_csv("./Datasets/processed_train_df_2.csv")
processed_val_df.to_csv("./Datasets/processed_val_df_2.csv")
processed_test_df.to_csv("./Datasets/processed_test_df_2.csv")


# get training set
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



class CommonLitReadabiltyDataset(Dataset):
    def __init__(self, X, Y, l):
        self.X = X
        self.y = Y
        self.l = l
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx], self.l[idx]



train_ds = CommonLitReadabiltyDataset(X_train, y_train, X_train_enc_len)
valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid, X_valid_enc_len)

# PyTorch training loop
def train_model_regr(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            #y = y.float()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss = validation_metrics_regr(model, val_dl)
        if i % 5 == 1:
            print("train mse %.3f val rmse %.3f" % (sum_loss/total, val_loss))


def validation_metrics_regr (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        #y = y.float()
        y_hat = model(x, l)
        loss = np.sqrt(F.mse_loss(y_hat, y.unsqueeze(-1)).item())
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total


batch_size = 64
vocab_size = len(words) 
embedding_dim = 300
hidden_dim = 200
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)


# LSTM regression model
class LSTM_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
    

model =  LSTM_regr(vocab_size, embedding_dim, hidden_dim)

train_model_regr(model, epochs=30, lr=0.005)


