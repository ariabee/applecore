import os
import re
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from text_transform import TextTransform
from model import SpeechRecognitionModel

'''
This script serves as a foundational aspect of development for the ASR unit for a voice controlled video game
for the Language, Action, and Perception software project. 
Eventual goal: a script that can take user speech input in real time and map that speech to actions, to control an avatar in a video game
Current goal: a script that can perform speech recognition given data with the model given in this blog post: https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch

This script is partially based on: https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
The cited website is primarily serving as the model for the data and also the decoding of the predictions (in librispeech_test.py)
'''
torch.set_printoptions(profile="full")
#path for LibriSpeech training dataset
#train_dataset = "/local/morganw/librispeech/LibriSpeech/train-server"

test_url="test-clean"
train_url="train-clean-100"

#test_local_dataset = torchaudio.datasets.LIBRISPEECH("/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech", url=test_url, download=True)

train_dataset = torchaudio.datasets.LIBRISPEECH("/local/morganw/librispeech/LibriSpeech",url=train_url, download=True)

test_dataset = train_dataset = torchaudio.datasets.LIBRISPEECH("/local/morganw/librispeech/LibriSpeech",url=test_url, download=True)

#train_local_dataset = torchaudio.datasets.LIBRISPEECH("/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech", url=train_url, download=True)
#train_local_dataset = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech/train-clean-100"

path_to_model = "/local/morganw/applecore/trained_models/librispeech_server.pt"

path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/librispeech_models/libritts_laptop.pt"



'''
Data Augmentation - SpecAugment
SpecAugment- cutting out random blocks of consecutive time and frequency distributions to improve the model's generalization ability
'''
#define a neural network-like layer stack to transform the waveform into a mel spectrogram and apply frequency masking and time masking
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    #frequency masking is how torchaudio performs SpecAugment for the frequency dimension
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    #time masking is how torchaudio performs SpechAugment for the time dimension
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

'''
Data Preprocessing: Extract feature from the spectrographs and map the transcriptions to numbers to create labels 
'''
text_transform = TextTransform()


def data_processing(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))


    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)


    return spectrograms, labels, input_lengths, label_lengths

#data_processing(train_local_dataset)
#spectrograms, labels, input_lengths, label_lengths = data_processing(train_local_dataset)
#print("labels")
#print(labels)

'''
LSTM
'''
use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=20,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x),
                                **kwargs)


test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=20,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x),
                                **kwargs)

torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")
criterion = nn.CTCLoss(blank=28).to(device)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data

        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        print("model")
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)

        output = output.transpose(0,1)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))

            # save the model
            '''
            The learnable parameters in PyTorch are contained in the model's parameters.
            A state_dict is a Python dictionary object that maps each layer to its parameters tensor.
            A state_dict can be easily saved, updated, altered, and restored.
            '''
            print("saving model")
            torch.save(model.state_dict(), path_to_model)



def test(model, device, test_loader, criterion, epoch):
    print("\nevaluating...")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

    print("Test set: Average loss: {:.4f}".format(test_loss))


#parameters for the model
rnn_dim = 512
hidden_dim = 512
dropout = 0.1
batch_first = True
n_feats = 128
cnn_layers = 3
stride = 2
rnn_layers = 5
n_class = 29

#call the model
model = SpeechRecognitionModel(cnn_layers, rnn_layers, rnn_dim, n_class, n_feats, stride, dropout)
#put the model on multiple GPUs
model = nn.DataParallel(model)
model.to(device)
#define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                             steps_per_epoch=int(len(train_loader)),
                                             epochs=20,
                                             anneal_strategy='linear')

#train the model
epochs = 20
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion, epoch)




