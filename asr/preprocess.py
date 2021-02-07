import os.path
import re
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
#from lstm import BidirectionalLSTM
from gru import BidirectionalGRU
from sklearn.preprocessing import LabelEncoder
'''
Method to preprocess the data: 
'''

test_corpus = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/speech_commands_full/speech_commands/commands"

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

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def preprocess_wav(path_to_wav):
    for top_directory in os.listdir(path_to_wav):
        if os.path.isdir(os.path.join(path_to_wav, top_directory)):
            working_directory = os.path.join(path_to_wav, top_directory)
            for second_directory in os.listdir(os.path.join(path_to_wav, working_directory)):
                wav_file = os.path.join(path_to_wav, working_directory, second_directory)
                waveform, sample_rate = torchaudio.load(wav_file)
                channel = 0
                transform = torchaudio.transforms.Resample(sample_rate, 8000)(waveform[channel, :].view(1, -1))
                # print(resampled_waveform)
                waveforms.append(transform)
   # print(transform)
    return transform

def preprocess_label(path_to_wav):
    for top_directory in os.listdir(path_to_wav):
        if os.path.isdir(os.path.join(path_to_wav, top_directory)):
            working_directory = os.path.join(path_to_wav, top_directory)
            label = top_directory
            labels.append(label)
   # print(labels)
    return labels


def preprocess_test(test_corpus):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    wav_file = preprocess_wav(test_corpus)
    waveform, sample_rate = torchaudio.load(wav_file)
    spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    transcription = preprocess_transcription(test_corpus)
    spectrograms.append(spec)
    label = torch.Tensor(text_transform.text_to_int(transcription.lower()))
    # create the labels by taking the preprocessed transcriptions and using the text_transform class to map the characters to numbers
    labels.append(label)
    input_lengths.append(spec.shape[0] // 2)

    label_lengths.append(len(label))
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    # spectrograms = spectrograms[:10]
    # print("spectrograms")
    # print(spectrograms)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    # input_lengths = input_lengths[:10]
    # labels = labels[:10]

   # print(spectrograms, labels, input_lengths, label_lengths)
    return spectrograms, labels, input_lengths, label_lengths

preprocess_wav(test_corpus)






