import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_dim)

    def forward(self, x):
        #print("x")
        #print(x.size())
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #print("h0")
        #print(h0.size())
        #print("c0")
        #print(c0.size())
        out, _ = self.lstm(x, (h0, c0))
       # print("lstm hidden")
       # print(out.size())
        out = self.fc(out[:, -1, :])
        return out
