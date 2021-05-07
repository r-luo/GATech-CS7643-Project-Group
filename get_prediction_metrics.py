import glob
import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="training pipeline"
)
parser.add_argument("-a", help="trained with all tickers or not, y means all ticker, n means single ticcker")
parser.add_argument("-m", help="LSTM or GRU")
# arguments
args = parser.parse_args()
assert args.m in ["LSTM", "GRU"], "model type specified wrong"

if __name__ == "__main__":
    trained_type = None
    if args.a == "y":
        trained_type = "AllTrained"
    else:
        trained_type = "SingleTrained"
    fileNames = glob.glob(os.path.join("prediction_data", "{}_*_{}_result.txt".format(args.m, trained_type)))
    hist_loss = []

    for fileName in fileNames:
        file = open(fileName, "r")
        lines = file.readlines()
        MAE_loss = float(lines[-1].split(":")[-1])
        hist_loss.append(MAE_loss)

    prices = []
    percentages = []
    stockFiles = glob.glob(os.path.join("prediction_data", "{}_*_{}.csv".format(args.m, trained_type)))
    for stockFile in stockFiles:
        df = pd.read_csv(stockFile)
        true_prices = df["Real_Price"].values
        predicted_prices = df["Predicted_Price"].values
        diff = np.abs(predicted_prices - true_prices)
        percent = diff/true_prices
        percentages.append(np.mean(percent))
        prices.append(np.mean(true_prices))

    hist_loss = np.array(hist_loss)
    prices = np.array(prices)

    percentages = np.array(percentages)
    mean_MAE = np.mean(hist_loss)
    std_MAE = np.std(hist_loss)

    mean_true_price = np.mean(prices)
    std_true_price = np.std(prices)

    print("mean MAE: ", mean_MAE)
    print("std MAE: ", std_MAE)
    print("mean MAE percentage: ", np.mean(percentages))
    print("std MAE percentage: ", np.std(percentages))

