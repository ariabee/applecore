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
train_dataset = "/local/morganw/librispeech/LibriSpeech/train-server"

train_local_dataset = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech/train-laptop/"

path_to_model = "/local/morganw/applecore/speech_commands_model/librispeech/librispeech_server.pt"

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
    flac_files = []
    transcriptions = []
    for top_directory in os.listdir(train_local_dataset):
        for second_directory in os.listdir(os.path.join(train_local_dataset, top_directory)):
            working_directory = os.path.join(train_local_dataset, top_directory, second_directory)
            for filename in os.listdir(working_directory):
                if filename.endswith(".flac"):
                    #get path of flac file
                    flac_file = os.path.join(working_directory, filename)
                    flac_files.append(flac_file)
                if filename.endswith(".txt"):
                    transcription_file = open(os.path.join(working_directory, filename))
                    transcription_file = transcription_file.read().split("\n")
                    for transcription in transcription_file:
                        # preprocess using a regex to remove the identifying labels and just have the transcribed speech
                        transcription_file = re.sub("[\d-]", "", transcription)
                        transcriptions.append(transcription_file)

    for ff in flac_files:
        # use torch audio to load the flac file to get the waveform and sample rate
        waveform, sample_rate = torchaudio.load(ff)
        # transform the waveform using the train_audio_transforms as defined above
        spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        # append all of the newly created spectrograms to a list
        spectrograms.append(spec)
        # create a list of input lengths for the CTC loss function
        input_lengths.append(spec.shape[0] // 2)

    for tf in transcriptions:
        # create the labels by taking the preprocessed transcriptions and using the text_transform class to map the characters to numbers
        label = torch.Tensor(text_transform.text_to_int(tf.lower()))
        labels.append(label)
        # create a list of label lengths for the CTC loss function
        label_lengths.append(len(label))

    #pad both the spectrograms and labels
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    #print("spectrograms")
   # print(spectrograms.shape)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    #print("labels")
    #print(labels.shape)

    return spectrograms, labels, input_lengths, label_lengths

#data_processing(train_local_dataset)
#spectrograms, labels, input_lengths, label_lengths = data_processing(train_local_dataset)
#print("labels")
#print(labels)

'''
LSTM
'''

train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=20,
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x)
                                )
use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")
criterion = nn.CTCLoss(blank=28, zero_infinity=True).to(device)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
       # print("input lengths")
        #print(len(input_lengths))
        #print("batch index")
        #print(batch_idx)
        print("spectrograms")
        print(spectrograms.shape)

        spectrograms, labels = spectrograms.to(device), labels.to(device)
        print("labels")
        print(labels.shape)

        optimizer.zero_grad()
        print("model")
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)

        output = output.transpose(0,1)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        #scheduler.step()

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
            #print("saving model")
           # torch.save(model.state_dict(), path_to_model)

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
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4,
                                             # steps_per_epoch=int(len(train_loader)),
                                             # epochs=10,
                                             #anneal_strategy='linear')

#train the model
epochs = 1
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)




