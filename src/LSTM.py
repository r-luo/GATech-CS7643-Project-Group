import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        #
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_state=None):
        #
        # print(x)
        # print(x.shape)
        if init_state is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        else:
            h0, c0 = init_state
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print("lstm output: ", out)
        # print("lstm output shape: ", out.shape)
        # print("out: ", out[:, -1, :])
        out = self.fc(out[:, -1, :])
        # print("fc out: ", out)
        return out
