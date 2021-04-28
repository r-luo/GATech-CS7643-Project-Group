import numpy
from itertools import product
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import torch
from LSTM import LSTM


def split_data(data_raw, lag, batch_size):
    """
    Split data for training and testing
    data_raw: raw data like price in 1 year
    lag: length of a chunk. for example, prices of day 1, day2, day3 are used a train data so lag is 3.
    return type: list of tensors, shape: (batch, seq, feature)
    """
    if type(data_raw) != np.ndarray:
        data_raw = data_raw.to_numpy()

    # data in model
    data_list = []
    for i in range(len(data_raw) - lag):
        data_list.append(data_raw[i:i+lag])

    # shuffle data
    data_list = np.array(data_list)
    np.random.shuffle(data_list)

    # split training and testing set
    test_size = int(np.round(0.2 * data_list.shape[0]))
    train_size = data_list.shape[0] - test_size
    #
    x_train = data_list[:train_size,:-1]
    y_train = data_list[:train_size,-1]
    #
    x_test = data_list[train_size:,:-1]
    y_test = data_list[train_size:,-1]

    # convert to pytorch tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # divide into batches
    x_train = list(chunks(x_train, batch_size))
    y_train = list(chunks(y_train, batch_size))
    x_test = list(chunks(x_test, batch_size))
    y_test = list(chunks(y_test, batch_size))

    return x_train, y_train, x_test, y_test


def divide_into_batches(data, batch_size):
    return list(chunks(data, batch_size))


def convert_data(data, batch_division=True, batch_size=128):
    """

    :param data: data in folds, fold data: "train": "x" : train_x, "y: ": train_y, "valid: ": "x": valid_x, "y:", valid_y,
            train_x, train_y, valid_x, valid_y shape: batch_size * sequence_length * feature size
    :param batch_size:
    :param batch_division:
    :return: data in folds and training data in batches
    """

    for i in range(len(data)):
        fold_arrs = data[i]
        x_train = fold_arrs["train"]["x"]
        y_train = fold_arrs["train"]["y"]
        x_valid = fold_arrs["valid"]["x"]
        y_valid = fold_arrs["valid"]["y"]
        # convert to torch tensor type
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        x_valid = torch.from_numpy(x_valid).type(torch.Tensor)
        y_valid = torch.from_numpy(y_valid).type(torch.Tensor)
        # divide training data into batches
        if batch_division:
            x_train = list(chunks(x_train, batch_size))
            y_train = list(chunks(y_train, batch_size))
            x_valid = list(chunks(x_valid, batch_size))
            y_valid = list(chunks(y_valid, batch_size))
        # update data folds
        data[i]["train"]["x"] = x_train
        data[i]["train"]["y"] = y_train
        data[i]["valid"]["x"] = x_valid
        data[i]["valid"]["y"] = y_valid
    return data


def rolling_cross_validation(data, model, num_epochs, criterion, optimizer):
    """

    :param data: data in folds, fold data: "train": "x" : train_x, "y: ": train_y, "valid: ": "x": valid_x, "y:", valid_y,
            train_x, train_y, valid_x, valid_y shape: batch_size * sequence_length * feature size
    :param model:
    :param num_epochs:
    :param criterion:
    :param optimizer:
    :return:
    """
    k = len(data)
    total_loss = 0

    for i in range(k-1):
        x_train = data[i]["train"]["x"]
        y_train = data[i]["train"]["y"]
        x_valid = data[i]["valid"]["x"]
        y_valid = data[i]["valid"]["y"]
        # shuffle data
        train_temp = list(zip(x_train, y_train))
        random.shuffle(train_temp)
        x_train, y_train = zip(*train_temp)
        model_name = "model_{}th_fold".format(i)
        model_copy = copy.deepcopy(model)
        val_loss = train(model_copy, num_epochs, x_train, y_train, x_valid, y_valid, criterion, optimizer, model_name, False, False)
        total_loss += val_loss
    return total_loss//(k-1)


def get_return_col(df, log=False):
    price_rat = df['adj_close'] / df['adj_close'].shift(-1)
    if log:
        return_col_name = "log_return"
        return_col_value = np.log(price_rat)
    else:
        return_col_name = "return"
        return_col_value = price_rat - 1
    df.loc[:, return_col_name] = return_col_value
    return return_col_name


def get_period_data(df, periods, date_col="date"):
    dfs_by_period = [df[pd.to_datetime(df[date_col]).between(pd.to_datetime(period[0]), pd.to_datetime(period[1]))].sort_values(date_col, ascending=True) for period in periods]
    return dfs_by_period


def write_pickle_file(obj, file):
    with Path(file).open('wb') as pkl_file:
        pickle.dump(obj, pkl_file, protocol=4)


def load_pickle_file(file):
    with Path(file).open('rb') as pkl_file:
        obj = pickle.load(pkl_file)


def data_loader(train_dfs, pipeline, batch_size=128):
    #
    max_overlap = pipeline.max_overlap
    model_seq_len = pipeline.model_seq_len
    cross_validation_folds = pipeline.cross_validation_folds
    #
    train_arrays = {"x": [], "y": [], "N": 0}
    for train_df in train_dfs:
        N = train_df.shape[0]
        step = max_overlap
        if N >= model_seq_len:
            for i in range((N - model_seq_len) // (model_seq_len - max_overlap)):
                train_arrays["x"].append(train_df[pipeline._feature_cols].iloc[
                                         (N - (i * (model_seq_len - max_overlap) + model_seq_len)):(
                                                     N - i * (model_seq_len - max_overlap))].values)
                train_arrays["y"].append([train_df["target"].iloc[(N - i * (model_seq_len - max_overlap)) - 1]])
                train_arrays["N"] += 1
    train_arrays["x"] = np.array(train_arrays['x'][::-1])
    train_arrays["y"] = np.array(train_arrays['y'][::-1])
    #
    train_val_distance = int(np.ceil(model_seq_len / (model_seq_len - max_overlap)))
    fold_size = (train_arrays["N"] - train_val_distance) // cross_validation_folds
    folds = []
    #
    # data in folds and divide into batches
    for i in range(cross_validation_folds):
        train_end_ind = fold_size * (i + 1)
        val_begin_ind = fold_size * (i + 1) + train_val_distance
        val_end_ind = val_begin_ind + fold_size
        fold_arrs = {
            "train": {
                "x": divide_into_batches(train_arrays["x"][:train_end_ind], batch_size),
                "y": divide_into_batches(train_arrays["y"][:train_end_ind], batch_size),
            },
            "valid": {
                "x": divide_into_batches(train_arrays["x"][val_begin_ind:val_end_ind], batch_size),
                "y": divide_into_batches(train_arrays["y"][val_begin_ind:val_end_ind], batch_size),
            },
        }
        folds.append(fold_arrs)
    #
    return folds


def train(model, num_epochs, x_train, y_train, x_validation, y_validation, criterion, optimizer, model_name, plot=False, model_save=False):
    """
    :param model: nlp model
    :param num_epochs:
    :param x_train:
    :param y_train:
    :param x_validation:
    :param y_validation:
    :param criterion:
    :param optimizer:
    :param model_name:
    :param plot:
    :return: None
    """

    # plot history of loss
    train_hist = np.zeros(num_epochs)
    val_hist = np.zeros(num_epochs)

    for epoch in range(num_epochs):

        # training
        model.train()
        train_loss_per_batch = 0
        # loop over batches
        # print("length of x_train: ", len(x_train))
        for x_train_sequence, target in zip(x_train, y_train):
            # Need to clear gradients before each instance
            print("x_train sequence shape: ", x_train_sequence.size())
            print("target shape: ", target.size())
            model.zero_grad()
            # training
            model.train()
            y_train_pred = model(x_train_sequence)
            train_loss = criterion(y_train_pred, target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #
            train_loss_per_batch += train_loss.item()
        train_loss_per_batch += train_loss_per_batch/len(x_train)

        # validation
        model.eval()
        val_loss_per_batch = 0
        with torch.no_grad():
            for x_val_sequence, val_target in zip(x_validation, y_validation):
                # Need to clear gradients before each instance
                model.zero_grad()

                # training
                y_val_pred = model(x_val_sequence)
                val_loss = criterion(y_val_pred, val_target)

                #
                val_loss_per_batch += val_loss.item()
            train_loss_per_batch += val_loss_per_batch/len(x_validation)

        print("Epoch ", epoch, "training MSE: ", train_loss_per_batch, "validation MSE: ", val_loss_per_batch)
        train_hist[epoch] = train_loss_per_batch
        val_hist[epoch] = val_loss_per_batch

    # plotting curves
    if plot:
        fig = plt.figure()
        plt.plot(train_hist, label="trainign loss")
        plt.plot(val_hist, label="validation loss")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss - MSE")
        plt.title("training loss")
        plt.savefig(model_name + ".jpg")

    # save model
    if model_save:
        saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
        path = os.path.join(saved_folder, model_name)
        torch.save(model, path)

    return val_loss_per_batch


def hyper_parameters_tunning(hyper_parameters, train_data, cross_validation=True):
    """

    :param hyper_parameters: hyper parameters
    :param train_data: training data in folds, used for cross validation
    :return:
    """

    combinations = list()
    for prod in product(*hyper_parameters.values()):
        temp = dict()
        for key, val in zip(hyper_parameters, prod):
            temp[key] = val
        combinations.append(temp)

    input_dim = train_data[0]["train"]["x"][0].shape[2]
    output_dim = train_data[0]["train"]["y"][0].shape[1]
    criterion = torch.nn.MSELoss(reduction='mean')
    best_loss = float('inf')
    best_params = None
    for combo in combinations:
        print("training combo: ", combo)
        hidden_dim = combo['hidden_dim']
        num_layers = combo['num_layers']
        num_epochs = combo['num_epochs']
        learning_rate = combo['learning_rate']
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model_name = "LSTM_trial_hidden_dims_{}__num_layers_{}__num_epochs_{}__lr_{}".format(hidden_dim,
                                                                                             num_layers,
                                                                                             num_epochs,
                                                                                             learning_rate)
        val_loss = rolling_cross_validation(train_data, model, num_epochs, criterion, optimizer)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = combo

    return best_params


def prediction_curve(model, real_price_data, test_data, lag, scaler, model_name):
    """
    Plot the prediction curve of real stock price vs. predicted stock price.
    :param model: nlp model trained
    :param real_price_data: dataset containing the real stock price
    :param test_data: dataset used to predict the stock price
    :param lag: length of a chunk (timestep)
    :param scaler: scaler used to process the test_data
    :param model_name: name of the output plot
    :return: None
    """
    
    # Preprocessing the test dataset
    test_data['Close'] = scaler.fit_transform(test_data['Close'].values.reshape(-1,1))
    test_data = test_data['Close']
    ## change the input shape
    test_data = np.expand_dims(test_data, axis=1)
    
    if type(real_price_data) != np.ndarray:
        real_price_data = real_price_data.to_numpy()

    if type(test_data) != np.ndarray:
        test_data = test_data.to_numpy()

    # real price dataframe
    real_price = real_price_data[lag:,]
    real_price = pd.DataFrame(real_price, columns = ['Date','Real_Price'])
    real_price['Date'] = pd.to_datetime(real_price.Date).dt.date
    
    # predict the stock price in test_data
    inputs = test_data
    X_test = []
    for i in range(len(inputs)-lag):
        X_test.append(inputs[i:i+lag])
    X_test = np.array(X_test)
    ## convert to pytorch tensor
    X_test = torch.from_numpy(X_test).type(torch.Tensor)

    # predicted price dataframe
    predicted_price = model(X_test).detach().numpy()
    predicted_price = scaler.inverse_transform(predicted_price)   
    predicted_price = pd.DataFrame(predicted_price, columns = ['Predicted_Price'])
    
    # concat two dataframes
    price_dat = pd.concat([real_price,predicted_price], axis = 1)
    
    # Visualising the results
    price_dat.plot(kind='line', x = "Date", y = ['Real_Price','Predicted_Price'], color = ['red','blue'],label = ['Real Stock Price','Predicted Stock Price'])
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(model_name)
    
def chunks(lst, n):
    """
    divide data into batches
    :param lst: input data
    :param n: batch length
    :return:
    """
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]