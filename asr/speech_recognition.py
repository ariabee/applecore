import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNNLayerNorm, ResidualCNN

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_feats, stride, dropout):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride, padding=3//2)

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)])

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        print(x)