import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.05):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with bidirectionality
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer (output size needs to account for the bidirectionality)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # hidden_size * 2 for bidirectional

    def forward(self, x):
        # Initialize hidden state and cell state for bidirectional LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Cell state

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the last time step's output
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)
        return out
