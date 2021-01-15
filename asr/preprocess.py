import os
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
'''
Method to preprocess the data: 
'''
sample_train = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/train-clean-100voice-controlled-world-game/asr/sample/"
train_corpus = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech/train-clean-100/"

def preprocess_transcription(path_to_transcription):

    for filename in os.listdir(path_to_transcription):
        if filename.endswith(".txt"):
            sample_transcription = open(os.path.join(path_to_transcription, filename))
            read_transcription = sample_transcription.read()

            #preprocess using a regex to remove the identifying labels and just have the transcribed speech
            sample_transcription_preprocessed = re.sub("^.{0,12}", "", read_transcription)
            #print(sample_train_preprocessed)
            return sample_transcription_preprocessed

#preprocess_transcription(sample_train)

def preprocess_wav(path_to_wav):
    for filename in os.listdir(path_to_wav):
        if filename.endswith(".flac"):
           # print(os.path.join(path_to_wav, filename))
            sample_wav_preprocessed = os.path.join(path_to_wav, filename)
            return sample_wav_preprocessed


#preprocess_wav(sample_train)
flac_files = []
transcription_files = []

def full_train_corpus(train_corpus):
    for top_directory in os.listdir(train_corpus):
        for second_directory in os.listdir(os.path.join(train_corpus, top_directory)):
            working_directory = os.path.join(train_corpus, top_directory, second_directory)
            for filename in os.listdir(working_directory):
                if filename.endswith(".flac"):
                    flac_files.append(filename)
                if filename.endswith(".txt"):
                    transcription_files.append(filename)
                    return flac_files, transcription_files


full_train_corpus(train_corpus)

