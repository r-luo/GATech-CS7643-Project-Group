import numpy
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


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
    plt.plot(train_hist, label="trainign loss")
    plt.plot(val_hist, label="validation loss")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("loss - MSE")
    plt.title("training loss")
    plt.savefig(model_name)

    # save model
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    path = os.path.join(saved_folder, model_name)
    torch.save(model, path)


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