import pandas as pd
import numpy as np
import random
import argparse
import pickle

# sklearn package for data preparation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# pytorch packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# set a fixed random state to produce the same results
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class WaterDataset(Dataset):
    def __init__(self, X, y):

        # The __init__ function is run once when instantiating the Dataset object.
        self.X = torch.tensor(X)
        self.y = torch.tensor(np.array(y).astype(float))

    def __len__(self):

        # The __len__ function returns the number of samples in our dataset.
        return len(self.y)

    def __getitem__(self, idx):

        # The __getitem__ function loads and returns a sample from the dataset
        # at the given index idx.
        return self.X[idx], self.y[idx]

class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # define 4 linear layers
        # the first input dimension is the number of features
        # the output layer is 1 
        # the other in and output parameters are hyperparamters and can be changed 
        nr_features = 9
        self.fc1 = nn.Linear(in_features=nr_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, x):

        # apply the linear layers with a relu activation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
    
        return x.squeeze()

def read_data(data):
    df = pd.read_csv(data)
    return df

def fill_na(df, vars_list):
    for var_ in vars_list:
        df[var_].fillna(value=df[var_].mean(),inplace=True)
    print(f'missing values: {df.isnull().sum()}')
    return df

def preprocessing(df):
    # upsample minority class
    df_pos = df[df['Potability'] == 1]
    df_neg = df[df['Potability'] == 0]

    water_upsample = resample(df_pos,
                            replace=True,
                            n_samples=(df['Potability'] == 0).sum(),
                            random_state=RANDOM_STATE)
    df_upsampled = pd.concat([water_upsample, df_neg])
    
    # store inputs and label
    X = df_upsampled.iloc[:,:-1]
    y = df_upsampled["Potability"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.11, random_state=RANDOM_STATE, shuffle=True)
    
    # scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    # save scaler
    with open('scaler.bin', 'wb') as f_out:
        pickle.dump(scaler, f_out)


    return X_train, X_val, X_test, y_train, y_val, y_test

def train(model, device, train_dataloader, optimizer, criterion, epoch, print_every):
    '''
    parameters:
    model - the model used for training
    device - the device we work on (cpu or gpu)
    train_dataloader - the training data wrapped in a dataloader
    optimizer - the optimizer used for optimizing the parameters
    criterion - loss function
    epoch - current epoch
    print_every - integer after how many batches results should be printed
    '''

    # create empty list to store the train losses
    train_loss = []
    # variable to count correct classified samples
    correct = 0
    # variable to count true positive, false positive and false negative samples
    TP = 0
    FP = 0
    FN = 0
    # create an empty list to store the predictions
    predictions = []

    # set model to training mode, i.e. the model is used for training
    # this effects layers like BatchNorm() and Dropout
    # in our simple example we don't use these layers
    # for the sake of completeness 'model.train()' is included
    model.train()

    # loop over batches
    for batch_idx, (x, y) in enumerate(train_dataloader):

        # set data to device
        x, y = x.to(device), y.to(device)

        # set optimizer to zero
        optimizer.zero_grad()

        # apply model
        y_hat = model(x.float())

        # calculate loss
        loss = criterion(y_hat, y.float())
        train_loss.append(loss.item())

        # backpropagation
        loss.backward()

        # update the weights
        optimizer.step()

        # print the loss every x batches
        if batch_idx % print_every == 0:
            percent = 100. * batch_idx / len(train_dataloader)
            print(f'Train Epoch {epoch} \
            [{batch_idx * len(train_dataloader)}/{len(train_dataloader.dataset)} \
            ({percent:.0f}%)] \tLoss: {loss.item():.6f}')

        # calculate some metrics

        # to get the predictions, we need to apply the sigmoid layer
        # this layer maps the data to the range [0,1]
        # we set all predictions > 0.5 to 1 and the rest to 0
        y_pred = torch.sigmoid(y_hat) > 0.5
        predictions.append(y_pred)
        correct += (y_pred == y).sum().item()
        TP += torch.logical_and(y_pred == 1, y == 1).sum()
        FP += torch.logical_and(y_pred == 1, y == 0).sum()
        FN += torch.logical_and(y_pred == 0, y == 1).sum()

    # total training loss over all batches
    train_loss = torch.mean(torch.tensor(train_loss))
    epoch_accuracy = correct/len(train_dataloader.dataset)
    # recall = TP/(TP+FN)
    epoch_recall = TP/(TP+FN)
    # precision = TP/(TP+FP)
    epoch_precision = TP/(TP+FP)

    return epoch_accuracy, train_loss, epoch_recall, epoch_precision

def valid(model, device, val_dataloader, criterion):
    '''
    parameters:
    model - the model used for training
    device - the device we work on (cpu or gpu)
    val_dataloader - the validation data wrapped in a dataloader
    criterion - loss function
    '''
    
    # create an empty list to store the loss
    val_loss = []
    # variable to count correct classified samples
    correct = 0
    # variable to count true positive, false positive and false negative samples
    TP = 0
    FP = 0
    FN = 0
    # create an empty list to store the predictions
    predictions = []

    # set model to evaluation mode, i.e. 
    # the model is only used for inference, this has effects on
    # dropout-layers, which are ignored in this mode and batchnorm-layers, which use running statistics
    model.eval()

    # disable gradient calculation 
    # this is useful for inference, when we are sure that we will not call Tensor.backward(). 
    # It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    with torch.no_grad():
        # loop over batches
        for x, y in val_dataloader:
            
            # set data to device
            x, y = x.to(device), y.to(device)

            # apply model
            y_hat = model(x.float())
            
            # append current loss
            loss = criterion(y_hat, y.float())
            val_loss.append(loss.item())

            # calculate some metrics
            
            # to get the predictions, we need to apply the sigmoid layer
            # this layer maps the data to the range [0,1]
            # we set all predictions > 0.5 to 1 and the rest to 0
            y_pred = torch.sigmoid(y_hat) > 0.5 
            predictions.append(y_pred)
            correct += (y_pred == y).sum().item()#y_pred.eq(y.view_as(y_pred)).sum().item()
            TP += torch.logical_and(y_pred == 1, y == 1).sum()
            FP += torch.logical_and(y_pred == 1, y == 0).sum()
            FN += torch.logical_and(y_pred == 0, y == 1).sum()
            
        # total validation loss over all batches
        val_loss = torch.mean(torch.tensor(val_loss))
        epoch_accuracy = correct/len(val_dataloader.dataset)
        # recall = TP/(TP+FN)
        epoch_recall = TP/(TP+FN)
        # precision = TP/(TP+FP)
        epoch_precision = TP/(TP+FP)
        
        print(f'Validation: Average loss: {val_loss.item():.4f}, \
                Accuracy: {epoch_accuracy:.4f} \
               ({100. * correct/len(val_dataloader.dataset):.0f}%)')
        
    return predictions, epoch_accuracy, val_loss, epoch_recall, epoch_precision


def main(args):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # read data
    df = read_data(args.data)
    print(df.head())

    # fill missing values
    vars_list = ['ph', 'Sulfate', 'Trihalomethanes']
    df = fill_na(df, vars_list)  

    # preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(df)

    # create datasets
    train_dataset = WaterDataset(X_train, y_train)
    valid_dataset = WaterDataset(X_val, y_val)

    # wrap into dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)
    print(f'train dataloader X: {next(iter(train_dataloader))[0].shape}')
    print(f'train dataloader y: {next(iter(train_dataloader))[1].shape}')
    print(f'val dataloader X: {next(iter(val_dataloader))[0].shape}')
    print(f'val dataloader y: {next(iter(val_dataloader))[1].shape}')

    # define the model and move it to the available device
    model = WaterNet().to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # define loss
    criterion = nn.BCEWithLogitsLoss()

    # create empty lists to store the accuracy and loss per validation epoch 
    train_epoch_accuracy = []
    train_epoch_loss = []
    train_epoch_recall = []
    train_epoch_precision = []
    val_epoch_accuracy = []
    val_epoch_loss = []
    val_epoch_recall = []
    val_epoch_precision = []
    min_loss = np.inf

    # loop over number of epochs
    print_every = 200
    for epoch in range(args.epochs):
    
        # train model for each epoch
        train_accuracy, train_loss, train_recall, train_precision = \
            train(model, device, train_dataloader, optimizer, criterion, epoch, print_every)
        # save training loss and accuracy for each epoch
        train_epoch_accuracy.append(train_accuracy)
        train_epoch_loss.append(train_loss)
        train_epoch_recall.append(train_recall)
        train_epoch_precision.append(train_precision)
    
        # validate model for each epoch
        predictions, val_accuracy, val_loss, val_recall, val_precision = \
            valid(model, device, val_dataloader, criterion)
        # save validation loss and accuracy for each epoch
        val_epoch_accuracy.append(val_accuracy)
        val_epoch_loss.append(val_loss)
        val_epoch_recall.append(val_recall)
        val_epoch_precision.append(val_precision)

        # save model if loss is decreasing
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), 'water_model_weights.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../medium/intro_pytorch/data/water_potability.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
