import torch
import os
from pathlib import Path
import model_data as md
from utilities import *


def predict(test_data, model_name):
    path = Path(__file__).absolute().parent.parent.joinpath("model")
    model = torch.load(os.path.join(path, model_name))
    model.eval()
    predictions = model(test_data)
    return predictions


if __name__ == "__main__":
    single_ticker_pipeline = md.SingleTickerPipeline(
        target="price",
        target_type="single",
        model_seq_len=30,
        max_overlap=20,
        train_periods=[
            # ("2000-01-01", "2006-12-31"),
            ("2018-06-01", "2018-12-31"),
        ],
        test_periods=[
            ("2007-01-01", "2008-12-31"),
            ("2019-01-01", "2021-04-01"),
        ],
        cross_validation_folds=5, )
    # Prepare data into folds
    stock_name = "AAPL"
    single_ticker_pipeline.prepare_data(stock_name)
    # load data
    single_ticker_pipeline.load_data(stock_name)
    #
    train_data = single_ticker_pipeline._train_out
    test_data = single_ticker_pipeline._test_out
    #
    # test_x = torch.from_numpy(test_data["x"]).type(torch.Tensor)
    # test_y = torch.from_numpy(test_data["y"]).type(torch.Tensor)
    # print(test_x.shape)
    # print(test_y.shape)
    # print(test_x)
    test_x = test_data["x"]
    test_y = test_data["y"]
    #
    model_name = "LSTM_AAPL"
    path = Path(__file__).absolute().parent.parent.joinpath("model")
    model = torch.load(os.path.join(path, model_name))
    model.eval()

    #
    for param in model.parameters():
        print(param.data)

    #
    x_train, y_train, x_valid, y_valid = split_data(test_x, test_y, batch_size=64)
    #
    for x_test_sequence, target in zip(x_train, y_train):
        # Need to clear gradients before each instance
        # training
        # print(x_test_sequence.shape)
        model.eval()
        model.zero_grad()
        with torch.no_grad():
            y_train_pred = model(x_test_sequence)
            print(y_train_pred)
        # print("y_train_pred: ", y_train_pred)
        # train_loss = criterion(y_train_pred, target)
