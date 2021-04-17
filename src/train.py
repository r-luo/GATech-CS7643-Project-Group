import numpy as np
import matplotlib.pyplot as plt


class Train:

    def __init__(self, x_train, y_train, model, criterion, optimizer, num_epochs, model_name):
        """
        :param x_train: X training data
        :param y_train: y training data
        :param model: model, LSTM or GRU
        :param criterion: criterion
        :param optimizer: optimizer
        :param num_epochs: num of epochs
        :param model_name: model name
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.x_train = x_train
        self.y_train = y_train
        self.model_name = model_name

    def training(self):
        # plot history of loss
        hist = np.zeros(self.num_epochs)

        for epoch in range(self.num_epochs):
            y_train_pred = self.model(self.x_train)

            loss = self.criterion(y_train_pred, self.y_train)
            print("Epoch ", epoch, "MSE: ", loss.item())
            hist[epoch] = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # plotting curves
        plt.plot(hist)
        plt.xlabel("number of epochs")
        plt.ylabel("loss - MSE")
        plt.title("training loss")
        plt.savefig(self.model_name)
