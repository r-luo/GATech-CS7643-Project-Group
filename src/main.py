from sklearn.preprocessing import MinMaxScaler
from utilities import *
from LSTM import LSTM

"""
Load data
"""
data =
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

"""
split training and validation data
"""
# define lag as 4 days
lag = 4
x_train, y_train, x_test, y_test = split_data(data, lag)

"""
set model hyper parameters
"""
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100
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
train(model, num_epochs, x_train, y_train, criterion, optimizer, model_name)

"""
Validation, plot, etc
"""

