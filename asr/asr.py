import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from text_transform import TextTransform
from preprocess import preprocess
from lstm import BidirectionalGRU, train
import speech_recognition as sr

'''
Speech recognition: take user speech input and save it into a wav file
'''

path_to_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.wav"

def userInput(path_to_wav):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

        with open(path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
    return path_to_wav

'''
Open and load saved wav file from the previous part (Speech Recognition)
'''
#get spectrograms and labels from the saved wav file
waveform, sample_rate = torchaudio.load(userInput(path_to_wav))


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

spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
#TODO: find out why we transpose and squeeze the transformed waveforms

#create the labels by taking the preprocessed transcriptions and using the text_transform class to map the characters to numbers
#label = torch.Tensor(text_transform.text_to_int(.lower()))
