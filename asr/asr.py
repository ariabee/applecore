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
from sklearn.preprocessing import LabelEncoder
from m5 import M5
'''
Speech recognition: take user speech input and save it into a wav file
'''

test_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.wav"

test_transcription = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.txt"

test_corpus = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/speech_commands_full/speech_commands/commands"

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

waveforms = []
labels = []

def preprocess_label(path_to_wav):
    for top_directory in os.listdir(path_to_wav):
        if os.path.isdir(os.path.join(path_to_wav, top_directory)):
            working_directory = os.path.join(path_to_wav, top_directory)
            label = top_directory
            labels.append(label)
   # print(labels)
    return labels

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

def train_set(waveform, label):
    waveform = preprocess_wav(test_corpus)
    label = preprocess_label(test_corpus)
    return waveform[0], label[0]

waveform, label = train_set(preprocess_wav(test_corpus), preprocess_label(test_corpus))


def label_to_index(word, labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index, labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

#preprocess_label(test_corpus)
word_start = "yes"
index = label_to_index(word_start, preprocess_label(test_corpus))
word_recovered = index_to_label(index, preprocess_label(test_corpus))


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

batch_size = 256

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set(preprocess_wav(test_corpus), preprocess_label(test_corpus)),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

model = M5(n_input=1, n_output=len(labels))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
new_sample_rate = 8000
sample_rate = 1600
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)

def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        print(data)
        print(target)

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # record loss
        losses.append(loss.item())

losses = []
transform = transform.to(device)
n_epoch = 2
log_interval = 20
for epoch in range(1, n_epoch + 1):
    train(model, epoch, log_interval)