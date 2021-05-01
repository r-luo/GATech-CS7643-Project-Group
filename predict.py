from src.utilities import *
import src.model_data as md
import sys

"""
============================
Load data from pipeline
============================
"""
if len(sys.argv) == 2:
    stock_name = sys.argv[1]
else:
    stock_name = sys.argv[2]
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
if len(sys.argv) == 2:
    model_name = "LSTM_{}".format(stock_name)
    model = torch.load("model/{}".format(model_name))


    """
    ============================
    Prediction
    ============================
    """

    pred_model_name = "LSTM_Prediction_{}".format(stock_name)
    prediction(model, test_data, stock_name, pred_model_name, True, True, False)
else:
    model_name = "LSTM_all_tickers"
    model = torch.load("model/{}".format(model_name))
    # prediction
    pred_model_name = "LSTM_Prediction_{}_with_AllTrained".format(stock_name)
    print("stock name: ", stock_name)
    prediction(model, test_data, stock_name, pred_model_name, True, True, False)

