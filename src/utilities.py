from itertools import product
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import torch
from .LSTM import LSTM
from .GRU import GRU
import csv


def split_data(x, y, batch_size):
    """
    Split data for training and testing
    data_raw: raw data like price in 1 year
    return type: list of tensors, shape: (batch, seq, feature)
    """
    # shuffle data
    ids_shuffle = np.random.permutation(x.shape[0])
    x = x[ids_shuffle]
    y = y[ids_shuffle]
    # split training and testing set
    test_size = int(np.round(0.2 * x.shape[0]))
    train_size = x.shape[0] - test_size
    #
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_valid = x[train_size:]
    y_valid = y[train_size:]

    # convert to pytorch tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_valid = torch.from_numpy(x_valid).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_valid = torch.from_numpy(y_valid).type(torch.Tensor)

    # divide into batches
    x_train = list(chunks(x_train, batch_size))
    y_train = list(chunks(y_train, batch_size))
    x_valid = list(chunks(x_valid, batch_size))
    y_valid = list(chunks(y_valid, batch_size))

    return x_train, y_train, x_valid, y_valid


def divide_into_batches(data, batch_size):
    return list(chunks(data, batch_size))


def convert_data(data, batch_division=True, batch_size=64):
    """

    :param data: data in folds, fold data: "train": "x" : train_x, "y: ": train_y, "valid: ": "x": valid_x, "y:", valid_y,
            train_x, train_y, valid_x, valid_y shape: batch_size * sequence_length * feature size
    :param batch_size:
    :param batch_division:
    :return: data in folds and training data in batches
    """
    for i in range(len(data)-1):
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


def rolling_cross_validation(data, model, num_epochs, criterion, learning_rate):
    """

    :param data: data in folds, fold data: "train": "x" : train_x, "y: ": train_y, "valid: ": "x": valid_x, "y:", valid_y,
            train_x, train_y, valid_x, valid_y shape: batch_size * sequence_length * feature size
    :param model:
    :param num_epochs:
    :param criterion:
    :param learning_rate:
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
        val_loss = train(model_copy, num_epochs, x_train, y_train, x_valid, y_valid,
                         criterion, learning_rate, model_name, False, False)
        total_loss += val_loss
    return total_loss//k


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


def train(model, num_epochs, x_train, y_train, x_validation, y_validation, criterion, learning_rate, model_name, plot=False, model_save=False):
    """
    :param model: nlp model
    :param num_epochs:
    :param x_train:
    :param y_train:
    :param x_validation:
    :param y_validation:
    :param criterion:
    :param learning_rate:
    :param model_name:
    :param plot:
    :return: None
    """

    # plot history of loss
    train_hist = []
    val_hist = []
    # define minimum loss and epoch no improve for early stopping
    epochs_no_improve = 0
    min_val_loss = float('inf')
    n_epoch_stop = 15
    # folder to save the plot and model
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):

        # training
        model.train()
        train_loss_per_batch = 0
        # loop over batches
        # print("length of x_train: ", len(x_train))
        for x_train_sequence, target in zip(x_train, y_train):
            # Need to clear gradients before each instance
            # data shuffle
            ids_shuffle = np.random.permutation(x_train_sequence.shape[0])
            x_train_sequence = x_train_sequence[ids_shuffle]
            target = target[ids_shuffle]
            #
            model.zero_grad()
            # training
            model.train()
            y_train_pred = model(x_train_sequence)
            # print(y_train_pred.shape)

            train_loss = criterion(y_train_pred, target)
            # print("y_train_pred: ", torch.transpose(y_train_pred, 0, 1))
            # print("target: ", torch.transpose(target, 0, 1))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #
            train_loss_per_batch += train_loss.item()
        train_loss_per_batch = train_loss_per_batch/len(x_train)

        # validation
        model.eval()
        val_loss_per_batch = 0
        with torch.no_grad():
            for x_val_sequence, val_target in zip(x_validation, y_validation):
                # Need to clear gradients before each instance
                model.zero_grad()

                # training
                y_val_pred = model(x_val_sequence)
                # print(x_val_sequence.shape)
                val_loss = criterion(y_val_pred, val_target)
                # print("y_val_pred: ", torch.transpose(y_val_pred, 0, 1))
                # print("val target: ", torch.transpose(val_target, 0, 1))
                # print("difference: ", y_val_pred - val_target)
                # print("val loss: ", val_loss)
                #
                val_loss_per_batch += val_loss.item()
            val_loss_per_batch = val_loss_per_batch/len(x_validation)
        # early stopping:
        if val_loss_per_batch < min_val_loss:
            min_val_loss = val_loss_per_batch
            # save the best model
            if model_save:
                path = os.path.join(saved_folder, model_name)
                torch.save(model, path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch > 5 and epochs_no_improve == n_epoch_stop:
            print("Early stopping!")
            break
        #
        print("Epoch ", epoch, "training MAE: ", train_loss_per_batch, "validation MAE: ", val_loss_per_batch)

        train_hist.append(train_loss_per_batch)
        val_hist.append(val_loss_per_batch)

    # plotting curves
    if plot:
        fig = plt.figure()
        plt.plot(train_hist, label="training loss")
        plt.plot(val_hist, label="validation loss")
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss - MAE")
        plt.title("training loss - {}".format(model_name))
        plt.savefig(os.path.join(saved_folder, "{}.jpg".format(model_name)))

    #
    # make the dataframe
    loss_hist = pd.DataFrame(
        {'training loss': train_hist, 'validation loss': val_hist})

    # Save the dataframe
    hidden_dim = model.hidden_dim
    num_layers = model.num_layers
    loss_hist_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_curves'))
    loss_hist.to_csv(os.path.join(loss_hist_folder, "{}__hd_{}__nl_{}__lr_{}_ training_loss.csv".format(model_name,
                                                                                                        hidden_dim,
                                                                                                        num_layers,
                                                                                                        learning_rate)), index=False)
    #
    print("min loss: ", min_val_loss)
    return min_val_loss


def hyper_parameters_tunning(hyper_parameters, train_data, criterion, model_type="LSTM"):
    """

    :param hyper_parameters: hyper parameters
    :param train_data: training data in folds, used for cross validation
    :param model_type: LSTM or GRU
    :param criterion: training criterion
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
    best_loss = float('inf')
    best_params = None
    for combo in combinations:
        hidden_dim = combo['hidden_dim']
        num_layers = combo['num_layers']
        num_epochs = combo['num_epochs']
        learning_rate = combo['learning_rate']
        if model_type == "LSTM":
            model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
        else:
            model = GRU(input_dim, hidden_dim, num_layers, output_dim)
        val_loss = rolling_cross_validation(train_data, model, num_epochs, criterion, learning_rate)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = combo

    return best_params


def prediction(model, test_data, criterion, stock_name, model_type, all_ticker, model_name, save_dat=True, save_result=True, prediction_curve=True, normalized=True):
    """
    Plot the prediction curve of real stock price vs. predicted stock price.
    :param model: model trained
    :param test_data: dataset used to predict the stock price
    :param criterion: metric to evaluate the deviation between real price and predicted price
    :param stock_name: name of stock predicted
    :param model_type: model type ("LSTM" or "GRU")
    :param all_ticker: model trained on data of all tickers or single ticker ("y" or "n")
    :param model_name: name of model
    :param save_dat: whether to save the data predicted
    :param save_result: whether to save the config file of trained model with the loss of prediction
    :param prediction_curve: whether to save the prediction curve
    :param normalized: whether to normailize the test data
    :return: None
    """
    
    # Preprocessing the test dataset
    test_x = test_data['x']
    # remove untransformed data
    if normalized:
        test_x = test_x[:,:,1:]
    test_y = test_data['y']
    test_date = test_data['prediction_date']
    
    if type(test_x) != np.ndarray:
        test_x = test_x.to_numpy()
    if type(test_y) != np.ndarray:
        test_y = test_y.to_numpy()
    if type(test_date) != np.ndarray:
        test_date = test_date.to_numpy()

    # real price dataframe
    real_price_tensor = torch.from_numpy(test_y)
    real_price = test_y.squeeze()
    date = test_date.squeeze()
    stock = [stock_name for i in range(test_y.shape[0])]

    # predict the stock price in test_data
    test_x = torch.from_numpy(test_x).type(torch.Tensor)

    # predicted price
    predicted_price_tensor = model(test_x)
    
    # MAE
    mae = criterion(predicted_price_tensor, real_price_tensor)    
    predicted_price = predicted_price_tensor.detach().numpy().squeeze()

    # make the dataframe
    price_dat = pd.DataFrame({'Stock': stock, 'Date': date, 'Real_Price': real_price, 'Predicted_Price': predicted_price})
    
    # Save the dataframe
    if save_dat:
        price_dat.to_csv("prediction_data/{}.csv".format(model_name), index=False)

    # Save result file
    if save_result:
        if all_ticker == "n":
            folder_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
            log_file = "{}_{}_config.txt".format(model_type, stock_name)   
        else:
            folder_config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
            log_file = "{}_all_tickers_config.txt".format(model_type)

        folder_result = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prediction_data'))
        result_file = "{}_result.txt".format(model_name)

        with open(os.path.join(folder_config, log_file), "r") as firstfile, open(os.path.join(folder_result, result_file), "w") as secondfile:
            # read content from first file
            for line in firstfile:
                # write content to second file
                secondfile.write(line)
            
            secondfile.writelines(["prediction loss: {} \n".format(mae)])
        firstfile.close()
        secondfile.close()

    saved_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prediction_data'))
    # Visualising the results
    if prediction_curve:
        price_dat.plot(kind='line', x = "Date", y = ['Real_Price','Predicted_Price'], color = ['red','blue'],label = ['Real Stock Price','Predicted Stock Price'])
        plt.title('{}_Curve'.format(model_name), fontsize = 12, y = 1.05)
        plt.suptitle('MAE: {}'.format(mae), y = 0.92, fontsize = 10)
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(os.path.join(saved_path, model_name))

    
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