import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        print("rnn dim")
        print(rnn_dim)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        print("layer norm")
        print(self.layer_norm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


#taken from the train section from: https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
def train(dataset, model):
    model.train()

    dataloader = DataLoader(dataset, batch_size=10)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(0,2):
        state_h, state_c = model.init_state(1)

        for batch, idx in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})


#for predict, model.eval()