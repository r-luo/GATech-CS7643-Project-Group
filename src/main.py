from sklearn.preprocessing import MinMaxScaler
from utilities import *
from LSTM import LSTM
import sys
from pathlib import Path
import model_data as md
import sys
sys.path.append(Path(".").absolute().parent.as_posix())


if __name__ == "__main__":
    """
    run script as main.py stock, 
    i.e. main.py TEAM
    """

    """
    Load data
    """
    # define data config
    single_ticker_pipeline = md.SingleTickerPipeline(
        target="price",
        target_type="single",
        model_seq_len=30,
        max_overlap=20,
        train_periods=[
            ("2000-01-01", "2006-12-31"),
            ("2009-01-01", "2018-12-31"),
        ],
        test_periods=[
            ("2007-01-01", "2008-12-31"),
            ("2019-01-01", "2021-04-01"),
        ],
        cross_validation_folds=5, )
    # Prepare data into folds
    single_ticker_pipeline.prepare_data(sys.argv[1])
    # load data
    single_ticker_pipeline.load_data(sys.argv[1])
    #
    train_data = single_ticker_pipeline._train_out
    test_data = single_ticker_pipeline._test_out
    """
    convert from numpy data type to pytorch and divide training data into 
    """
    train_data = convert_data(train_data)
    # """
    # set model hyper parameters
    # """
    input_dim = train_data[0]["train"]["x"][0].shape[2]
    output_dim = train_data[0]["train"]["y"][0].shape[1]
    # define criterion
    criterion = torch.nn.MSELoss(reduction='mean')
    # Model name for saving
    model_name = "LSTM_trial_V0"
    """
    Hyper parameters tuning
    """
    hyper_parameters = {"hidden_dim": [16, 32, 64], "num_layers": [2, 4, 8], "num_epochs": [15], "learning_rate": [0.01]}
    best_combo = hyper_parameters_tunning(hyper_parameters, train_data)
    """
    Start training
    """
    hidden_dim = best_combo["hidden_dim"]
    num_layers = best_combo["num_layers"]
    num_epochs = best_combo["num_epochs"]
    learning_rate = best_combo["learning_rate"]
    # Use LSTM model
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    x_train = train_data[len(train_data)-1]["train"]["x"]
    y_train = train_data[len(train_data)-1]["train"]["y"]
    x_valid = train_data[len(train_data)-1]["valid"]["x"]
    y_valid = train_data[len(train_data)-1]["valid"]["y"]
    train(model, num_epochs, x_train, y_train, x_valid, y_valid, criterion, learning_rate, model_name, True, True)
    #
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
