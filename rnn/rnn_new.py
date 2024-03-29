#````````````````````````````````````````````````````````````````````````````````````````````````````
# from https://www.kaggle.com/code/houssemayed/lstm-models-for-regression-on-text-data/notebook
#````````````````````````````````````````````````````````````````````````````````````````````````````

# Imports

import torch
import random
import numpy as np
import pandas as pd
import csv


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# from pytorch_pretrained_bert import BertTokenizer
# from pytorch_pretrained_bert import BertForMaskedLM

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from collections import defaultdict
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import pos_tag

import ast
from tqdm import tqdm

from utilities import set_seed, set_device
from preprocessing import preprocess_data

SEED = 42
set_seed(seed=SEED)
DEVICE = set_device()

preprocessed = True  # Set to True if the CSVs are already preprocessed

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
        # Convert string representation of a list into an actual list
        try:
            x_as_list = ast.literal_eval(self.X[idx])
        except ValueError as e:
            # If the value is a single integer, convert it to a list
            if isinstance(self.X[idx], int):
                x_as_list = [self.X[idx]]
            else:
                print(f"Error parsing string to list at index {idx}: {self.X[idx]}")
                raise e
        # Convert list into a NumPy array
        x_as_array = np.array(x_as_list, dtype=np.float32)
        
        return torch.from_numpy(x_as_array).float(), torch.tensor(self.y[idx], dtype=torch.float32), self.l[idx]


# PyTorch training loop
def train_model_regr(model, train_dl, val_dl, epochs=10, lr=0.001):
    model.to(DEVICE)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # Check regression - Relu. TODO
    for i in range(epochs):
        for x, y, l in tqdm(train_dl, desc=f"Epoch {i+1}/{epochs}"):
            torch.cuda.empty_cache()
            x, y = x.to(DEVICE).long(), y.to(DEVICE).float()
            l = l.to(DEVICE)
            
            y_pred = model(x, l)
            optimizer.zero_grad()
            # Convert y and y_pred from seconds to days
            y = y / (60*60*24)
            y_pred = y_pred / (60*60*24)
            
            # loss = F.mse_loss(y_pred, y.unsqueeze(-1)) # TODO: Try with different loss functions - MAE, RMSE.
            # loss = torch.log(loss)
            loss = combined_rmse_mae_loss(y_pred, y.unsqueeze(-1))
            
            # If the loss is NaN, skip this batch
            if torch.isnan(loss):
                continue
            
            loss.backward()
            optimizer.step()
        
        if i % 10 == 0:
            model.eval()
            val_rmse = calculate_rmse(model, val_dl)
            train_rmse = calculate_rmse(model, train_dl)
            
            val_mae = calculate_mae(model, val_dl)
            train_mae = calculate_mae(model, train_dl)
            
            model.train()
            print("Epoch %d | Train RMSE %.3f | Train MAE %.3f | Val RMSE %.3f | Val MAE %.3f" % (i, train_rmse, train_mae, val_rmse, val_mae))
    return model


def calculate_rmse(model, valid_dl):
    total = 0
    sum_loss = 0.0
    with torch.no_grad():
        for x, y, l in valid_dl:
            x, y = x.to(DEVICE).long(), y.to(DEVICE).float()
            l = l.to(DEVICE)  # Assuming `l` is also a tensor
            y_hat = model(x, l)
            
            # Convert y and y_hat from seconds to days
            y = y / (60*60*24)
            y_hat = y_hat / (60*60*24)
            
            loss = np.sqrt(F.mse_loss(y_hat, y.unsqueeze(-1)).item())
            
            total += y.shape[0]
            sum_loss += loss.item()*y.shape[0]
            
    # Calculate the MAE as well:
    rmse = sum_loss/total
    return rmse

def calculate_mae(model, valid_dl):
    total_samples = 0
    absolute_error_sum = 0.0
    with torch.no_grad():
        for x, y, l in valid_dl:
            x, y = x.to(DEVICE).long(), y.to(DEVICE).float()
            l = l.to(DEVICE)
            y_hat = model(x, l)
            
            # Convert y and y_hat from seconds to days
            y = y / (60*60*24)
            y_hat = y_hat / (60*60*24)
            
            absolute_error_sum += torch.abs(y_hat.squeeze() - y).sum().item() # Ensure y_hat is squeezed to match y's shape
            total_samples += y.size(0)
        
    mae = absolute_error_sum / total_samples
    return mae



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



def combined_rmse_mae_loss(y_pred, y_true, alpha=0.5):
    """
    Combines RMSE and MAE into a single loss.
    
    Parameters:
    - y_pred: Predicted values
    - y_true: True values
    - alpha: Weight for RMSE and MAE. 0 <= alpha <= 1. 
             alpha = 1 means the loss is entirely RMSE, 
             alpha = 0 means the loss is entirely MAE.
    
    Returns:
    - Combined loss value
    """
    mse_loss = F.mse_loss(y_pred, y_true)
    mae_loss = F.l1_loss(y_pred, y_true)
    rmse_loss = torch.sqrt(mse_loss)
    return alpha * rmse_loss + (1 - alpha) * mae_loss



#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# Load GloVe Vectors (pretrained model)
#  https://nlp.stanford.edu/projects/glove/
#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

#We can load the vectors using our custom functions
def load_glove_vectors(glove_file):#= glove_embedding_vectors_text_file):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(word_vecs, word_counts, emb_size= 300):
    """ Creates embedding matrix from word vectors"""
    vocab_size = len(word_counts) + 2
    print(f"get_emb_matrix vocab_size: {vocab_size}")
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in word_vecs:
            W[i] = word_vecs[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx




#```````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# LSTM Model with GloVe vectors
#``````````````````````````````````````````````````````````````````````````````````
class LSTM_reg_glove_vecs(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, glove_weights) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.from_numpy(glove_weights))
        self.embeddings.weight.requires_grad = False ## Freeze embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])



#````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# LSTM + attention model
# ````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
class PositionAwareAttention(nn.Module):
    
    def __init__(self, input_size, attn_size):
        super().__init__()
        self.input_size  = input_size
        self.wx = nn.Conv1d(input_size, attn_size, 1, bias=True)  # from input to attention matrix
        self.wh = nn.Conv1d(input_size, attn_size, 1, bias=False) # from hidden to attention matrix
        self.wt = nn.Conv1d(attn_size, 1, 1, bias=True)           # from attention matrix to score
        
    def forward(self, x, h):
        x = x.permute(1,2,0) # features last
        wx = self.wx(x)
        wh = self.wh(h.permute(1,0,2).contiguous().view(-1,self.input_size,1))
        score = self.wt(torch.tanh(wx + wh))
        score = F.softmax(score, dim=2)
        out = torch.bmm(score, x.permute(0,2,1)).squeeze()
        
        return out


class RecNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, glove_weights, layers=1, atten_features = 24, 
                 dropout=0., bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = layers
        self.emb_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emb.weight.data.copy_(torch.from_numpy(glove_weights)) # load pretrained vectors
        self.emb.weight.requires_grad = False # make embedding non trainable
        
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size,
                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)
        
        self.gru = nn.GRU(self.emb_dim, self.hidden_size,
                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)
        self.pregru = nn.Conv1d(self.emb_dim, self.emb_dim, 1, bias=True)
        self.atten = PositionAwareAttention(hidden_size*(bidirectional+1), atten_features)
        
        self.out = nn.Linear(2* hidden_size*(bidirectional+1), 32)
        self.last = nn.Linear(32, 1)
    
    def forward(self, x, l):    
        embs = self.emb(x)
        
        lstm, (h1, c) = self.lstm(embs)
        gru = F.relu(self.pregru(embs.permute(1,2,0)), inplace=True).permute(2,0,1)
        
        gru, h2 = self.gru(gru, h1)
        lstm = lstm + gru
        
        x_max, _ = lstm.max(dim=0, keepdim=False) 
        x_atten = self.atten(lstm, h1+h2)

        # Ensure that x_max and x_atten have the same number of dimensions
        if len(x_max.shape) < len(x_atten.shape):
            x_max = x_max.unsqueeze(-1)
        elif len(x_max.shape) > len(x_atten.shape):
            x_atten = x_atten.unsqueeze(-1)
        # print(f"x_max shape: {x_max.shape}")
        # print(f"x_atten shape: {x_atten.shape}")
        out = self.out(torch.cat([x_max, x_atten.t()], dim = 1))
        out = self.last(F.relu(out)).squeeze()
        return out
    

processed_train_csv_path = "./Datasets/processed_train_df_2.csv"
processed_val_csv_path = "./Datasets/processed_val_df_2.csv"
processed_test_csv_path = "./Datasets/processed_test_df_2.csv"


def main():
   
    if not preprocessed:
        # preprocess data
        processed_train_df, processed_val_df, processed_test_df, words, counts = preprocess_data(train_csv_path, val_csv_path, test_csv_path, mod_book_csv_path)

        # save processed dataframes 
        processed_train_df.to_csv("./Datasets/processed_train_df_2.csv", index=False)
        processed_val_df.to_csv("./Datasets/processed_val_df_2.csv", index=False)
        processed_test_df.to_csv("./Datasets/processed_test_df_2.csv", index=False)
    else:
        print("skipping preprocessing..")
        # get dataframes
        processed_train_df = pd.read_csv(processed_train_csv_path)
        processed_test_df = pd.read_csv(processed_test_csv_path)
        processed_val_df = pd.read_csv(processed_val_csv_path)
        #mod_books_df = pd.read_csv(mod_book_csv_path)
        with open('words2.csv') as f:
            reader = csv.reader(f, delimiter=',')
            data = list(reader)
        words = data[0]

        with open('counts.csv') as f:
            reader = csv.reader(f)
            counts = {rows[0]:rows[1] for rows in reader}


    # get training set
    print("\ngetting train set")
    X_train = list(processed_train_df['encoded'])
    X_train_enc_len = list(processed_train_df['encoded_length'])
    y_train = list(processed_train_df['time_to_start_seconds'])

    # get validation set
    print("\ngetting validation set")
    X_valid = list(processed_val_df['encoded'])
    X_valid_enc_len = list(processed_val_df['encoded_length'])
    y_valid = list(processed_val_df['time_to_start_seconds'])
   

    train_ds = CommonLitReadabiltyDataset(X_train, y_train, X_train_enc_len)
    valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid, X_valid_enc_len)

    # Load GLoVe vectors (pretrained model / embeddings)
    glove_embedding_vectors_text_file = "./embeddings/glove/glove.6B.300d.txt"


    word_vecs = load_glove_vectors(glove_embedding_vectors_text_file)
    pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)
    
    batch_size = 64
    #vocab_size = len(words) 
    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 200
    num_workers = 2
    epochs = 200
    lr = 1


        # Print the arguments
    print("\n\nBatch size: %d" % batch_size)
    print("Vocab size: %d" % vocab_size)
    print("Embedding dim: %d" % embedding_dim)
    print("Hidden dim: %d" % hidden_dim)
    print("Number of workers: %d" % num_workers)
    print("Epochs: %d" % epochs)
    print("Learning rate: %.3f" % lr)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)
    

    #model =  LSTM_regr(vocab_size, embedding_dim, hidden_dim)
    #model = LSTM_reg_glove_vecs(vocab_size, embedding_dim, hidden_dim, pretrained_weights)
    model = RecNN(vocab_size, embedding_dim, hidden_dim, pretrained_weights, dropout=0.4)
    
    print("\n\nBegin training")
    model = train_model_regr(model, train_dl, val_dl, epochs=epochs, lr=lr)

    # get test set
    print("Testing on the test set")
    X_test = list(processed_test_df['encoded'])
    X_test_enc_len = list(processed_test_df['encoded_length'])
    y_test = list(processed_test_df['time_to_start_seconds']) 

    test_ds = CommonLitReadabiltyDataset(X_test, y_test, X_test_enc_len)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    rmse = calculate_rmse(model, test_dl)
    mae = calculate_mae(model, test_dl)
    print("\n\n\tTest RMSE: %.3f | Test MAE: %.3f" % (rmse, mae))


if __name__ == "__main__":
    main()