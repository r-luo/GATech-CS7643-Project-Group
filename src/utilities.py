import numpy
from itertools import product
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
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


def rolling_corss_validation(data, model, num_epochs, criterion, optimizer):
    k = len(data)

    total_loss = 0
    for i in range(k-1):
        train_data = train_data + data[i]
        valid_data = data[i+1]
        # shuffle data
        random.shuffle(train_data)
        random.shuffle(valid_data)
        x_train, y_train= get_XY(train_data)
        x_valid, y_valid = get_XY(valid_data)
        model_name = "model_{}th_fold".format(i)
        val_loss = train(model, num_epochs, x_train, y_train, x_valid, y_valid, criterion, optimizer, model_name)
        total_loss += val_loss
    return total_loss/float(k-1)



def train(model, num_epochs, x_train, y_train, x_validation, y_validation, criterion, optimizer, model_name):
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
    :return: None
    """

    # plot history of loss
    train_hist = np.zeros(num_epochs)
    val_hist = np.zeros(num_epochs)

    for epoch in range(num_epochs):

        # training
        model.train()
        train_loss_per_batch = 0
        for x_train_sequence, target in zip(x_train, y_train):
            # Need to clear gradients before each instance
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
    fig = plt.figure()
    plt.plot(train_hist, label="trainign loss")
    plt.plot(val_hist, label="validation loss")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("loss - MSE")
    plt.title("training loss")
    plt.savefig(model_name + ".jpg")

    # save model
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    path = os.path.join(saved_folder, model_name)
    torch.save(model, path)

    return val_loss_per_batch

def hyper_parameters_tunning(hyper_parameters, x_train, y_train, x_validation, y_validation, output_dim):
    """

    :param hyper_parameters: hyper parameters
    :param train_parameters: training parameters, i.e model, training x, training y
    :return:
    """

    combinations  = list()

    for prod in product(*hyper_parameters.values()):
        temp = dict()
        for key, val in zip(hyper_parameters, prod):
            temp[key] = val
        combinations.append(temp)

    input_dim = x_train[0].shape[2]
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
        val_loss = train(model, num_epochs, x_train, y_train, x_validation, y_validation, criterion, optimizer, model_name)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = combo

    print("best combinations: ", best_params)


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