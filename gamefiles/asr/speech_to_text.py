import speech_recognition as sr
import torch
from asr.m5 import M5
import torchaudio

class SpeechToText():
    def __init__(self):
        use_cuda = False
        self.model = M5(n_input=1, n_output=35)
        self.model.load_state_dict(torch.load("speech_commands_model/speech_commands_model.pt")) #FileNotFoundError
        self.path_to_wav = "asr/user_input.wav"
        with open('asr/labels.txt') as f:
            self.labels = f.read().strip().splitlines()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.transform = torchaudio.transforms.Resample(orig_freq=1600, new_freq=8000)

    def userInput(self, audio):
        with open(self.path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
        return self.path_to_wav

    def inputLoad(self):
        waveform, sample_rate = torchaudio.load(userInput(self.path_to_wav))
        return waveform

    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def get_likely_index(self, tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def get_prediction(self, waveform):
        # Use the model to predict the label of the waveform
        tensor = waveform.to(device)
        tensor = self.transform(tensor)
        tensor = self.model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        str_label = index_to_label(tensor.squeeze())
        return str_label

