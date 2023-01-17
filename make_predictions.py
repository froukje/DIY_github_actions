import pickle

import pandas as pd
import torch

from water_model import WaterNet

features = [
    8.099124189298397,
    224.23625939355776,
    19909.541732292393,
    9.275883602694089,
    359.948574,
    418.6062130644815,
    16.868636929550973,
    66.42009251176368,
    3.0559337496641685,
]


def preprocessing(features):
    with open("/home/frauke/DIY_github_actions/scaler.bin", "rb") as f:
        scaler = pickle.load(f)

    df = pd.DataFrame([features])
    x = scaler.transform(df)
    return x


def predict(x):

    model = WaterNet()
    model.load_state_dict(
        torch.load("/home/frauke/DIY_github_actions/water_model_weights.pth")
    )
    model.eval()

    x = torch.from_numpy(x).float()
    output = model(x)
    y_pred = torch.sigmoid(output) > 0.5
    return y_pred


x = preprocessing(features)
print(f"preprocessed: {x}")
y_pred = predict(x)
print(f"prediction: {y_pred}")
