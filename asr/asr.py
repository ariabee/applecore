import speech_recognition as sr
import torch
import torch.nn as nn
from model import SpeechRecognitionModel
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import os
import re
import gc
from comet_ml import Experiment
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from text_transform import TextTransform
#from preprocess import preprocess_wav, preprocess_transcription
from decoder import GreedyDecoder
'''
Speech recognition: take user speech input and save it into a wav file
'''

test_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.wav"

test_transcription = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.txt"

path_to_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/speech_recognition_saved_models/server.pt"

path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/librispeech_models/laptop.pt"

path_to_transcription = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test_transcription.txt"

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

#define a neural network-like layer stack to transform the waveform into a mel spectrogram and apply frequency masking and time masking
train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    #frequency masking is how torchaudio performs SpecAugment for the frequency dimension
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    #time masking is how torchaudio performs SpechAugment for the time dimension
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

text_transform = TextTransform()

def userInput(path_to_wav):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

        with open(path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
        return path_to_wav

def transcribeInput(test_wav, test_transcription):
    r = sr.Recognizer()
    transcription = sr.AudioFile(userInput(test_wav))
    with transcription as source:
        audio = r.record(source)
        with open(test_transcription, "wb") as f:
            f.write(r.recognize_google(audio))
        return test_transcription


def preprocess_wav(waveform):
    input_lengths = []
    waveform, sample_rate = torchaudio.load(waveform)
    spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    input_lengths.append(spec.shape[0] // 2)
    #print(spec.size())
    return spec

def preprocess_transcription(transcription):
    label_lengths = []
    label = torch.Tensor(text_transform.text_to_int(transcription.lower()))
    label_lengths.append(len(label))
    return label, label_lengths



wav_file = preprocess_wav(test_wav)
wav_file = np.expand_dims(wav_file, axis=0)
wav_file = np.expand_dims(wav_file, axis=0)
wav_file = torch.tensor(wav_file)
print(wav_file.size())




def test(model, device, test_loader, criterion, optimizer, scheduler, epoch):
    model.eval()
    data_len = len(test_loader.dataset)
    for batch_idx, _data in enumerate(test_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)


        optimizer.zero_grad()
        print("model")
        output = model(spectrograms)

        #TODO: take most of this out- only leave the spectrograms, labels, and label lengths
        #see if we can use another dataloader- the current one is just not working
        #see if we can use the GreedyDecoder to output the transcribed text
        #also, take out the transcription from the transcribe method. our goal is to have the model do it and have
        #the user tell us if it is right or wrong

        output = F.log_softmax(output, dim=2)

        output = output.transpose(0, 1)


        loss = criterion(output, labels, input_lengths, label_lengths)
        #test_loss += loss.item() / len(test_loader)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        print(decoded_preds, decoded_targets)

#predict(path_to_local_model)
#userInput(path_to_wav)
#transcribe(path_to_wav)


rnn_dim = 512
hidden_dim = 512
dropout = 0.1
batch_first = True
n_feats = 128
cnn_layers = 3
stride = 2
rnn_layers = 5
n_class = 29
model = SpeechRecognitionModel(cnn_layers, rnn_layers, rnn_dim, n_class, n_feats, stride, dropout)
model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))

output = model(wav_file)
print("output")
#print(output)

epoch = 1
#test(model, device, test_loader, criterion, optimizer, scheduler, epoch)