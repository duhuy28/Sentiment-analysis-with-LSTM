import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_output_label):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=1,batch_first=True)
        self.fc = nn.Linear(hidden_size, num_output_label)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_word_embedding):
        lstm_out, _ = self.lstm(input_word_embedding)
        out = self.fc(lstm_out)
        sig_out = self.sigmoid(out)
        return sig_out

