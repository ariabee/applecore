import speech_recognition as sr
import torch
import torch.nn as nn
from m5 import M5
import torchaudio

class SpeechToText():
    def __init__(self, model, tensor, path_to_wav, labels, device, transform):
        self.model = model
        self.tensor = tensor
        self.path_to_wav = path_to_wav
        self.labels = labels
        self.device = device
        self.transform = transform

    def userInput(self, path_to_wav):
        r = sr.Recognizer()

        with sr.Microphone() as source:
            audio = r.listen(source)

            with open(path_to_wav, "wb") as f:
                f.write(audio.get_wav_data())
            return path_to_wav

    def inputLoad(self, userInput, path_to_wav):
        waveform, sample_rate = torchaudio.load(userInput(path_to_wav))
        return waveform

    def index_to_label(self, labels, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return labels[index]

    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def get_prediction(self, tensor, device, transform, model, get_likely_index, index_to_label):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = transform(tensor)
        tensor = model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        tensor = index_to_label(tensor.squeeze())
        return tensor