from sklearn.preprocessing import MinMaxScaler
from utilities import *
from LSTM import LSTM
import pandas as pd


if __name__ == "__main__":
    """
    Load data
    """
    saved_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    file = os.path.join(saved_folder, "AMZN_2006-01-01_to_2018-01-01.csv")
    data = pd.read_csv(file)
    price = data[['Close']]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    raw_data = price['Close']
    raw_data = np.expand_dims(raw_data, axis=1)
    # print(raw_data.shape)
    # print(raw_data)

    """
    split training and validation data
    """
    # define lag as 4 days
    lag = 4
    batch_size = 256
    x_train, y_train, x_validation, y_validation = split_data(raw_data, lag, batch_size)
    """
    set model hyper parameters
    """
    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 15
    learning_rate = 0.01
    # Use LSTM model
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    # define criterion
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Model name for saving
    model_name = "LSTM_trial_V0"

    """
    Start training
    """
    train(model, num_epochs, x_train, y_train, x_validation, y_validation, criterion, optimizer, model_name)

    """
    Validation, plot, etc
    """

