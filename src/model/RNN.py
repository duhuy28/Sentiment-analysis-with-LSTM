import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_output_label):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.layers = nn.Sequential(nn.Linear(hidden_size, num_output_label),
                                    nn.Sigmoid())

    def forward(self, input_word_embedding):
        rnn_output, _ = self.rnn(input_word_embedding)
        return self.layers(rnn_output)