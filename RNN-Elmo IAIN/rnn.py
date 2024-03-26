import torch

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utilities import set_seed, set_device
from preprocessing import preprocess_data

import ast
from tqdm import tqdm

SEED = 42
set_seed(seed=SEED)
DEVICE = set_device()

preprocessed = False  # Set to True if the CSVs are already preprocessed

# datasets paths
train_csv_path = "/exports/eddie/scratch/s2062378/MLP_Dataset/train_interactions.csv"
val_csv_path = "/exports/eddie/scratch/s2062378/MLP_Dataset/validate_interactions.csv"
test_csv_path = "/exports/eddie/scratch/s2062378/MLP_Dataset/test_interactions.csv"
mod_book_csv_path = "/exports/eddie/scratch/s2062378/MLP_Dataset/books_filtered_by_language_modified_desc.csv"

class CommonLitReadabiltyDataset(Dataset):
    def __init__(self, X, Y, l):
        self.X = X
        self.y = Y
        self.l = l
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx], self.l[idx]

    
# LSTM regression model
class LSTM_regr(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1) # TODO: Try wth different number of out features
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

def main():
    if not preprocessed:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        val_df = pd.read_csv(val_csv_path)
        mod_books_df = pd.read_csv(mod_book_csv_path)
        
        # preprocess data
        processed_train_df, processed_val_df, processed_test_df, words = preprocess_data(train_df, val_df, test_df, mod_books_df)

        # save processed dataframes 
        processed_train_df.to_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_train_df.csv", index=False)
        processed_val_df.to_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_val_df.csv", index=False)
        processed_test_df.to_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_test_df.csv", index=False)
        # Save the words to a file
        with open("/exports/eddie/scratch/s2062378/MLP_Dataset/words.txt", "w") as file:
            for word in words:
                file.write(word + "\n")
               
    else:
        processed_train_df = pd.read_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_train_df.csv")
        processed_val_df = pd.read_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_val_df.csv")
        processed_test_df = pd.read_csv("/exports/eddie/scratch/s2062378/MLP_Dataset/processed_test_df.csv")
        with open("/exports/eddie/scratch/s2062378/MLP_Dataset/words.txt", "r") as file:
            words = file.readlines()
            words = [word.strip() for word in words]
        
    print("Preprocessing Finished")
    
    # get training set
    X_train = list(processed_train_df['encoded'])
    X_train_enc_len = list(processed_train_df['encoded_length'])
    y_train = list(processed_train_df['time_to_start_seconds'])

    # get validation set
    X_valid = list(processed_val_df['encoded'])
    X_valid_enc_len = list(processed_val_df['encoded_length'])
    y_valid = list(processed_val_df['time_to_start_seconds'])

    train_ds = CommonLitReadabiltyDataset(X_train, y_train, X_train_enc_len)
    valid_ds = CommonLitReadabiltyDataset(X_valid, y_valid, X_valid_enc_len)

    batch_size = 7000
    vocab_size = len(words)
    embedding_dim = 300
    hidden_dim = 200
    num_workers = 2
    epochs = 100
    lr = 1.5
    
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

    model =  LSTM_regr(vocab_size, embedding_dim, hidden_dim)
    print("\n\nBegin training")
    model = train_model_regr(model, train_dl, val_dl, epochs=epochs, lr=lr)
    
    # Test on the test set
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