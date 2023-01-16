import numpy as np
import pandas as pd
import torch
from water_model import WaterDataset, WaterNet
import pickle

features = [4.668102, 193.681735, 47580.991603, 7.166639, 359.948574, 526.424171, 13.894419, 66.687695, 4.435821]

def preprocessing(features):
    with open('/home/frauke/DIY_github_actions/scaler.bin', 'rb') as f:
        scaler = pickle.load(f)

    df = pd.DataFrame([features])
    x = scaler.transform(df)
    return x

def predict(x):

    model = WaterNet()
    model.load_state_dict(torch.load('/home/frauke/DIY_github_actions/water_model_weights.pth'))
    model.eval()

    x = torch.from_numpy(x).float()
    output = model(x)
    y_pred = torch.sigmoid(output) > 0.5
    return y_pred

x = preprocessing(features)
print(f'preprocessed: {x}')
y_pred = predict(x)
print(f'prediction: {y_pred}')

