import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self, input_size, batch_size, hidden_size, drop_out, num_classes, num_layers
    ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=drop_out,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        # out, (hn, cn) = self.lstm(x)
        # out = self.fc(hn[-1])
        # return out

        out, hidden = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(
        self, input_size, batch_size, hidden_size, drop_out, num_classes, num_layers
    ):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.num_classes = num_classes
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=drop_out,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        sequence_length,
        device,
    ):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        return out


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        sequence_length,
        num_classes,
        device,
    ):
        super(SimpleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
        self.device = device

    def forward(self, x):

        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        if len(x.shape) == 2:
            x = x.unsqueeze(2)

        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
