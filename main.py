from src.utilities import *
from src.LSTM import LSTM
from pathlib import Path
import src.model_data as md
import sys
import time
sys.path.append(Path(".").absolute().parent.as_posix())


if __name__ == "__main__":
    """
    ===================================
    run script as main.py stock, 
    i.e. python main.py TEAM
    ======================================
    """

    """
    ============================
    Load data from pipeline
    ============================
    """
    start_time = time.time()
    # stock name is from input arguments
    stock_name = sys.argv[1]
    # define data config
    single_ticker_pipeline = md.SingleTickerPipeline(
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
    single_ticker_pipeline.prepare_data(stock_name)
    # load data
    single_ticker_pipeline.load_data(stock_name)
    #
    train_data = single_ticker_pipeline._train_out
    test_data = single_ticker_pipeline._test_out
    """
    ======================================================================
    convert from numpy data type to pytorch and divide training data into batches
    data_in_folds are used for rolling cross validation
    ======================================================================
    """
    # get data for rolling cross validation
    data_in_folds = convert_data(train_data)
    """
    =================================================================
    get input & output dimensions and set up optimization criterion
    =================================================================
    """
    input_dim = data_in_folds[0]["train"]["x"][0].shape[2]
    output_dim = data_in_folds[0]["train"]["y"][0].shape[1]
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
    # x_all = np.append(train_data[len(train_data)-1]["train"]["x"], train_data[len(train_data)-1]["valid"]["x"], axis=0)
    # y_all = np.append(train_data[len(train_data)-1]["train"]["y"], train_data[len(train_data)-1]["valid"]["y"], axis=0)
    x_all = train_data['_all_']["x"]
    y_all = train_data['_all_']['y']
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
    hidden_dim = 84
    num_layers = 2
    num_epochs = 60
    learning_rate = 0.002
    # Use LSTM model
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    #
    model_name = "LSTM_{}".format(stock_name)
    val_loss = train(model, num_epochs, x_train, y_train, x_valid, y_valid, criterion, learning_rate, model_name, True, True)
    #
    end_time = time.time()
    # save training config file
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    log_file = "{}_config.txt".format(model_name)
    file = open(os.path.join(saved_folder, log_file), "w")
    lines = ["input_dim: {} \n".format(input_dim), "output_dim: {} \n".format(output_dim),
             "epochs_number: {} \n".format(num_epochs), "validation loss: {} \n".format(val_loss),
             "hidden_dim: {} \n".format(hidden_dim), "num_layers: {} \n".format(num_layers),
             "learning_rate: {} \n".format(learning_rate), "training total time: {} \n".format(end_time-start_time)]
    file.writelines(lines)
    file.close()
    # """
    # Curves predictions
    # """
    #


    #
    #
    #
    # """
    # Prediction Curve
    # """
    # # model_name_pred = "LSTM_prediction_V0"
    # # real_price_data = data[['Date','Close']]
    # # test_data = data[['Close']]
    # #
    # # prediction_curve(model, real_price_data, test_data, lag, scaler, model_name_pred)
