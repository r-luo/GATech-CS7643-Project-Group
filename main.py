from src.utilities import *
from src.LSTM import LSTM
from src.GRU import GRU
from pathlib import Path
import src.model_data as md
import sys
import time
import argparse
sys.path.append(Path(".").absolute().parent.as_posix())

"""
===================================
script running options:
for example
    python main.py -t AAPL -m LSTM 
        : train AAPL with LSTM model
    
    python main.py -t all -m LSTM 
        : run all tickers with LSTM model
    
    python main.py -t AAPL -m GRU 
        : run AAPL with GRU model

    python main.py -t all -m GRU 
        : run all tickers with GRU model
======================================
"""

parser = argparse.ArgumentParser(
    description="training pipeline"
)
parser.add_argument("-t", help="name of the ticker")
parser.add_argument("-m", help="LSTM or GRU")
# arguments
args = parser.parse_args()
assert args.m in ["LSTM", "GRU"], "model type specified wrong"

if __name__ == "__main__":

    """
    ============================
    Load data from pipeline
    ============================
    """
    start_time = time.time()
    # stock name is from input arguments
    stock_name = args.t
    # define data config
    # single ticker:
    if stock_name != "all":
        ticker_pipeline = md.SingleTickerPipeline(
            target="price",
            target_type="single",
            model_seq_len=5,
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
        ticker_pipeline.prepare_data(stock_name)
        # load data
        ticker_pipeline.load_data(stock_name)
    else:
        ticker_pipeline = md.MultiTickerPipeline(
            target="price",
            target_type="single",
            model_seq_len=5,
            max_overlap=2,
            # normalization_method="quantile",
            train_periods=[
                ("2012-01-01", "2019-12-31"),
            ],
            test_periods=[
                ("2020-01-01", "2021-04-01"),
            ],
            cross_validation_folds=5, )
        # Prepare data into folds
        # ticker_pipeline.prepare_data(['_all_'])
        ticker_pipeline.load_data("96tickers")
    #
    train_data = ticker_pipeline._train_out
    test_data = ticker_pipeline._test_out
    """
    ======================================================================
    convert from numpy data type to pytorch and divide training data into batches
    data_in_folds are used for rolling cross validation
    ======================================================================
    """
    # get data for rolling cross validation
    # data_in_folds = convert_data(train_data)
    """
    =================================================================
    get input & output dimensions and set up optimization criterion
    =================================================================
    """
    # input_dim = data_in_folds[0]["train"]["x"][0].shape[2]
    # output_dim = data_in_folds[0]["train"]["y"][0].shape[1]
    # define criterion
    criterion = torch.nn.L1Loss(reduction='mean')
    """
    =====================================================
    Hyper parameters tuning with rolling cross validation
    =====================================================
    """
    # hyper_parameters = {"hidden_dim": [32], "num_layers": [2], "num_epochs": [65], "learning_rate": [0.001]}
    # best_combo = hyper_parameters_tunning(hyper_parameters, data_in_folds, criterion)
    """
    ================================================
    get data for final training
    ================================================
    """
    # get all data for final training
    #
    x_all = train_data['_all_']["x"]
    y_all = train_data['_all_']['y']
    # # delete untransformed data
    # x_all = x_all[:,:,:1]
    # split data and convert into batches
    x_train, y_train, x_valid, y_valid = split_data(x_all, y_all, batch_size=64)
    """
    ==================================================
    Start training, using all the data in training set
    ===================================================
    """
    # hidden_dim = best_combo["hidden_dim"]
    # num_layers = best_combo["num_layers"]
    # num_epochs = best_combo["num_epochs"]
    # learning_rate = best_combo["learning_rate"]
    input_dim = x_train[0].shape[2]
    output_dim = y_train[0].shape[1]
    hidden_dim = 192
    num_layers = 8
    num_epochs = 400
    learning_rate = 0.001
    # Use LSTM model
    if args.m == "LSTM":
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        if stock_name != "all":
            model_name = "LSTM_{}".format(stock_name)
        else:
            model_name = "LSTM_all_tickers"
    else:
        model = GRU(input_dim, hidden_dim, num_layers, output_dim)
        if stock_name != "all":
            model_name = "GRU_{}".format(stock_name)
        else:
            model_name = "GRU_all_tickers"
    #
    val_loss = train(model, num_epochs, x_train, y_train, x_valid, y_valid, criterion, learning_rate, model_name, True, True)
    #
    end_time = time.time()
    # save training config file
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'model'))
    log_file = "{}_config.txt".format(model_name)
    file = open(os.path.join(saved_folder, log_file), "w")
    lines = ["input_dim: {} \n".format(input_dim), "output_dim: {} \n".format(output_dim),
             "epochs_number: {} \n".format(num_epochs), "validation loss: {} \n".format(val_loss),
             "hidden_dim: {} \n".format(hidden_dim), "num_layers: {} \n".format(num_layers),
             "learning_rate: {} \n".format(learning_rate), "training total time: {} \n".format(end_time-start_time)]
    file.writelines(lines)
    file.close()



    

