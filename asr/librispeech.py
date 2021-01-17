import os
import re
import gc
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from text_transform import TextTransform
#from preprocess import full_train_corpus
#from lstm import BidirectionalLSTM
from gru import BidirectionalGRU
from cnn import CNNLayerNorm, ResidualCNN
from speech_recognition import SpeechRecognitionModel
torch.manual_seed(1)
'''
This script serves as a foundational aspect of development for the ASR unit for a voice controlled video game
for the Language, Action, and Perception software project. 
Eventual goal: a script that can take user speech input in real time and map that speech to actions, to control an avatar in a video game
Current goal: a script that can perform speech recognition given data with an LSTM

This script is partially based on: https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
The cited website is primarily serving as the preprocessing of the data and the data itself
Caution: this script works with torch and torchaudio versions 0.4.0 and 1.4.0, respectively. The most recent torch version available for download is 1.7.0.
'''

#declare paths for both the training and testing datasets
train_dataset = "/local/morganw/librispeech/LibriSpeech/train-clean-100/"
#test_dataset = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/asr/LibriSpeech/test-clean"

path_to_model = "/local/morganw/speech_recognition_saved_models/server.pt"

#check to see if the datasets have already been downloaded. If not, download them.
#if os.path.isfile(train_file):
   # print("Train dataset exists")
#else:
  #  train_dataset = torchaudio.datasets.LIBRISPEECH("./", url="train-clean-100", download=True)
    #TODO: add way to unzip file and create a directory like the one in the train_dataset variable

#if os.path.isfile(test_file):
  #  print("Test dataset exists")
#else:
  #  test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)
    #TODO: add way to unzip file and create a directory like the one in the test_dataset variable


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
    flac_files = []
    transcription_files = []
    for top_directory in os.listdir(train_dataset):
        for second_directory in os.listdir(os.path.join(train_dataset, top_directory)):
            working_directory = os.path.join(train_dataset, top_directory, second_directory)
            for filename in os.listdir(working_directory):
                if filename.endswith(".flac"):
                    flac_file = os.path.join(working_directory, filename)
                    waveform, sample_rate = torchaudio.load(flac_file)
                    spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
                if filename.endswith(".txt"):
                    transcription_file = open(os.path.join(working_directory, filename))
                    transcription_file = transcription_file.read().split("\n")
                    for transcription in transcription_file:
                        # preprocess using a regex to remove the identifying labels and just have the transcribed speech
                        transcription_file = re.sub("[\d-]", "", transcription)
                        #print(transcription_file)
                        label = torch.Tensor(text_transform.text_to_int(transcription_file.lower()))

        spectrograms.append(spec)

        #create the labels by taking the preprocessed transcriptions and using the text_transform class to map the characters to numbers
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)

        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    #spectrograms = spectrograms[:10]
    #print("spectrograms")
   # print(spectrograms)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
   # input_lengths = input_lengths[:10]
   # labels = labels[:10]

    return spectrograms, labels, input_lengths, label_lengths

#data_processing(train_dataset)
'''
LSTM
'''

train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=10,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x)
                                )
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cpu")
criterion = nn.CTCLoss(blank=28).to(device)


def train(model, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    with torch.no_grad():
        for batch_idx, _data in enumerate(train_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)


            optimizer.zero_grad()

            output = model(spectrograms)

            output = F.log_softmax(output, dim=2)

            output = output.transpose(0, 1)


            loss = criterion(output, labels, input_lengths, label_lengths)

            loss.backward()

            optimizer.step()
            scheduler.step()
           # gc.collect()
            #torch.cuda.empty_cache()

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
                #print("Model's state_dict:")
            # for param_tensor in model.state_dict():
                # print(param_tensor, "\t", model.state_dict()[param_tensor].size())

                print("saving model")
                torch.save(model.state_dict(), path_to_model)

rnn_dim = 512
hidden_dim = 512
dropout = 0.1
batch_first = True
n_feats = 128
cnn_layers = 3
stride = 2
rnn_layers = 5
n_class = 29

#model = BidirectionalGRU(input_dim, hidden_dim, dropout, batch_first)
model = SpeechRecognitionModel(cnn_layers, rnn_layers, rnn_dim, n_class, n_feats, stride, dropout)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=10,
                                              anneal_strategy='linear')
model.to(device)
train(model, 10)





