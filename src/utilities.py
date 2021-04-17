import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def split_data(data_raw, lag):
    """
    Split data for training and testing
    data_raw: raw data like price in 1 year
    lag: input length of a chunk. for example, prices of day 1, day2, day3 are used a train data so lag is 3.
    """
    data_raw = data_raw.to_numpy()

    # data in model
    data_list = []
    for i in range(len(data_raw) - lag):
        data_list.append(data_raw[i:i+lag])

    # shuffle data
    data_list = np.random.shuffle(np.array(data_list))

    # split training and testing set
    test_size = int(np.round(0.2 * data_list.shape[0]))
    train_size = data_list.shape[0] - test_size
    #
    x_train = data_list[:train_size,:-1,:]
    y_train = data_list[:train_size,-1,:]
    #
    x_test = data_list[train_size:,:-1]
    y_test = data_list[train_size:,-1,:]

    # convert to pytorch tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)

    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return x_train, y_train, x_test, y_test


def train(model, num_epochs, x_train, y_train, criterion, optimizer, model_name):
    # plot history of loss
    hist = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        print("Epoch ", epoch, "MSE: ", loss.item())
        hist[epoch] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plotting curves
    plt.plot(hist)
    plt.xlabel("number of epochs")
    plt.ylabel("loss - MSE")
    plt.title("training loss")
    plt.savefig(model_name)

    # save model
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    path = os.path.join(saved_folder, model_name)
    torch.save(model, path)
