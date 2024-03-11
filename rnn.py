#````````````````````````````````````````````````````````````````````````````````````````````````````
# from https://www.kaggle.com/code/houssemayed/lstm-models-for-regression-on-text-data/notebook
#````````````````````````````````````````````````````````````````````````````````````````````````````

# Imports

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

#SEED = 42
#set_seed(seed=SEED)
#DEVICE = set_device()

preprocessed = False  # Set to True if the CSVs are already preprocessed

# datasets paths
train_csv_path = "./Datasets/train_interactions.csv"
val_csv_path = "./Datasets/validate_interactions.csv"
test_csv_path = "./Datasets/test_interactions.csv"

book_csv_path = "./Datasets/books_filtered_by_language.csv"
mod_book_csv_path = "./Datasets/books_filtered_by_language_modified_desc.csv"



class CommonLitReadabiltyDataset(Dataset):
    def __init__(self, X, Y, l):
        self.X = X
        self.y = Y
        self.l = l
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx], self.l[idx]



# PyTorch training loop
def train_model_regr(model, train_dl, val_dl, epochs=10, lr=0.001):
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


#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# Load GloVe Vectors (pretrained model)
#  https://nlp.stanford.edu/projects/glove/
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# glove_embedding_vectors_text_file = "../input/embeddings-glove-crawl-torch-cached/crawl-300d-2M.vec"
# glove_embedding_vectors_pt_file = "../input/embeddings-glove-crawl-torch-cached/crawl-300d-2M.vec.pt"

# We can load the vectors using our custom functions
# def load_glove_vectors(glove_file= glove_embedding_vectors_text_file):
#     """Load the glove word vectors"""
#     word_vectors = {}
#     with open(glove_file) as f:
#         for line in f:
#             split = line.split()
#             word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
#     return word_vectors

# def get_emb_matrix(pretrained, word_counts, emb_size = 300):
#     """ Creates embedding matrix from word vectors"""
#     vocab_size = len(word_counts) + 2
#     vocab_to_idx = {}
#     vocab = ["", "UNK"]
#     W = np.zeros((vocab_size, emb_size), dtype="float32")
#     W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
#     W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
#     vocab_to_idx["UNK"] = 1
#     i = 2
#     for word in word_counts:
#         if word in word_vecs:
#             W[i] = word_vecs[word]
#         else:
#             W[i] = np.random.uniform(-0.25,0.25, emb_size)
#         vocab_to_idx[word] = i
#         vocab.append(word)
#         i += 1   
#     return W, np.array(vocab), vocab_to_idx

# word_vecs = load_glove_vectors()
# pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)
# # Or we can do it directly 
# itos, stoi, pretrained_weights, embedding_dim = torch.load(glove_embedding_vectors_pt_file)
# vocab_size = pretrained_weights.size(0)
# pretrained_weights = pretrained_weights.numpy()


#```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# LSTM Model with GloVe vectors
#``````````````````````````````````````````````````````````````````````````````````
# class LSTM_reg_glove_vecs(torch.nn.Module) :
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights) :
#         super().__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
#         self.embeddings.weight.requires_grad = False ## Freeze embeddings
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, 1)
#         self.dropout = nn.Dropout(0.4)
        
#     def forward(self, x, l):
#         x = self.embeddings(x)
#         x = self.dropout(x)
#         lstm_out, (ht, ct) = self.lstm(x)
#         return self.linear(ht[-1])
    
# model_glove = LSTM_reg_glove_vecs(vocab_size, embedding_dim, hidden_dim, pretrained_weights)

# train_model_regr(model_glove, epochs=30, lr=0.005)



#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# LSTM + attention model
# ````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# class PositionAwareAttention(nn.Module):
    
#     def __init__(self, input_size, attn_size):
#         super().__init__()
#         self.input_size  = input_size
#         self.wx = nn.Conv1d(input_size, attn_size, 1, bias=True)  # from input to attention matrix
#         self.wh = nn.Conv1d(input_size, attn_size, 1, bias=False) # from hidden to attention matrix
#         self.wt = nn.Conv1d(attn_size, 1, 1, bias=True)           # from attention matrix to score
        
#     def forward(self, x, h):
#         x = x.permute(1,2,0) # features last
#         wx = self.wx(x)
#         wh = self.wh(h.permute(1,0,2).contiguous().view(-1,self.input_size,1))
#         score = self.wt(torch.tanh(wx + wh))
#         score = F.softmax(score, dim=2)
#         out = torch.bmm(score, x.permute(0,2,1)).squeeze()
        
#         return out


# class RecNN(nn.Module):

#     def __init__(self, embs_dim, hidden_size, glove_weights, layers=1, atten_features = 24, 
#                  dropout=0., bidirectional=False):
#         super().__init__()

#         self.hidden_size = hidden_size
#         self.bidirectional = bidirectional
#         self.num_layers = layers
#         self.emb_dim = embs_dim
#         self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.emb.weight.data.copy_(torch.from_numpy(glove_weights)) # load pretrained vectors
#         self.emb.weight.requires_grad = False # make embedding non trainable
        
#         self.lstm = nn.LSTM(self.emb_dim, self.hidden_size,
#                             num_layers=layers, bidirectional=bidirectional, dropout=dropout)
        
#         self.gru = nn.GRU(self.emb_dim, self.hidden_size,
#                             num_layers=layers, bidirectional=bidirectional, dropout=dropout)
#         self.pregru = nn.Conv1d(self.emb_dim, self.emb_dim, 1, bias=True)
#         self.atten = PositionAwareAttention(hidden_size*(bidirectional+1), atten_features)
        
#         self.out = nn.Linear(2* hidden_size*(bidirectional+1), 32)
#         self.last = nn.Linear(32, 1)
    
#     def forward(self, x, l):    
#         embs = self.emb(x)
        
#         lstm, (h1, c) = self.lstm(embs)
#         gru = F.relu(self.pregru(embs.permute(1,2,0)), inplace=True).permute(2,0,1)
        
#         gru, h2 = self.gru(gru, h1)
#         lstm = lstm + gru
        
#         x_max, _ = lstm.max(dim=0, keepdim=False) 
#         x_atten = self.atten(lstm, h1+h2)
#         out = self.out(torch.cat([x_max, x_atten],dim = 1))
#         out = self.last(F.relu(out)).squeeze()
#         return out
    
#model_LSTM_attention = RecNN(embedding_dim, hidden_dim, pretrained_weights, dropout=0.4)
#train_model_regr(model_LSTM_attention, epochs=30, lr=0.005)



# def main():

#     # get dataframes
#     train_df = pd.read_csv(train_csv_path)
#     test_df = pd.read_csv(test_csv_path)
#     val_df = pd.read_csv(val_csv_path)
#     mod_books_df = pd.read_csv(mod_book_csv_path)

#     if not preprocessed:
#         # preprocess data
#         processed_train_df, processed_val_df, processed_test_df, words = preprocess_data(train_df, val_df, test_df, mod_books_df)

#         # save processed dataframes 
#         processed_train_df.to_csv("./Datasets/processed_train_df_2.csv", index=False)
#         processed_val_df.to_csv("./Datasets/processed_val_df_2.csv", index=False)
#         processed_test_df.to_csv("./Datasets/processed_test_df_2.csv", index=False)

#     # get training set
#     X_train = list(processed_train_df['encoded'])
#     X_train_enc_len = list(processed_train_df['encoded_length'])
#     y_train = list(processed_train_df['time_to_start_seconds'])

#     # get validation set
#     X_valid = list(processed_val_df['encoded'])
#     X_valid_enc_len = list(processed_val_df['encoded_length'])
#     y_valid = list(processed_val_df['time_to_start_seconds'])

#     # get test set
#     X_test = list(processed_test_df['encoded'])
#     X_test_enc_len = list(processed_test_df['encoded_length'])
#     y_test = list(processed_test_df['time_to_start_seconds'])    

#     train_ds = CommonLitReadabiltyDataset(X_train, y_train, X_train_enc_len)
#     valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid, X_valid_enc_len)

#     batch_size = 64
#     vocab_size = len(words) 
#     embedding_dim = 300
#     hidden_dim = 200
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_dl = DataLoader(valid_ds, batch_size=batch_size)

#     model =  LSTM_regr(vocab_size, embedding_dim, hidden_dim)
#     train_model_regr(model, train_dl, val_dl, epochs=30, lr=0.005)