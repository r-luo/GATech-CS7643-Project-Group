from src.utilities import *
import src.model_data as md
import sys
import argparse

"""
usgae:
    python predict.py -t AAPL -a n -m LSTM
"""

parser = argparse.ArgumentParser(
    description="prediction pipeline"
)
parser.add_argument("-t", help="name of the ticker to predict")
parser.add_argument("-a", help="model trained with all ticker or not, y or n")
parser.add_argument("-m", help="LSTM or GRU")

args = parser.parse_args()

stock_name = args.t
allTicker = args.a
model_type = args.m

assert allTicker in ["y", "n"], "allTicker or not is not specified."
assert model_type in ["LSTM", "GRU"], "model type specified wrong"

if __name__ == "__main__":
    """
    ============================
    Load data from pipeline
    ============================
    """
    #
    single_ticker_pipeline = md.SingleTickerPipeline(
            target="price",
            target_type="single",
            model_seq_len=4,
            max_overlap=2,
            # normalization_method="quantile",
            train_periods=[
                ("2000-01-01", "2006-12-31"),
                ("2009-01-01", "2018-12-31"),
            ],
            test_periods=[
                # ("2007-01-01", "2008-12-31"),
                ("2019-01-01", "2021-04-01"),
            ],
            cross_validation_folds=5, )
    # Prepare data into folds
    single_ticker_pipeline.prepare_data(stock_name)
    # load data
    single_ticker_pipeline.load_data(stock_name)
    #
    train_data = single_ticker_pipeline._train_out
    test_data = single_ticker_pipeline._test_out


    """
    ============================
    Load the trained model
    ============================
    """
    if allTicker == "n":
        # model trained with single ticker
        if model_type == "LSTM":
            model_name = "LSTM_{}".format(stock_name)
        else:
            model_name = "GRU_{}".format(stock_name)
        # load model
        model = torch.load(os.path.join("model", model_name))
        # prediction and save
        pred_model_name = "{}_Prediction_{}".format(model_type, stock_name)
        prediction(model, test_data, stock_name, pred_model_name, True, True, False)
    else:
        if model_type == "LSTM":
            model_name = "LSTM_all_tickers"
        else:
            model_name = "GRU_all_tickers"
        #
        model = torch.load(os.path.join("model", model_name))
        # prediction
        pred_model_name = "{}_Prediction_{}_with_AllTrained".format(model_type, stock_name)
        prediction(model, test_data, stock_name, pred_model_name, True, True, False)

