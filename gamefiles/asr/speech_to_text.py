import speech_recognition as sr
import torch.nn as nn
from asr.m5 import M5
import torchaudio

class SpeechToText():
    def __init__(self, tensor, labels):
        self.model = M5(n_input=1, n_output=35)

        self.model.load_state_dict(torch.load("speech_commands_model/speech_commands_model.pt"))
        self.tensor = tensor
        self.path_to_wav = "user_input.wav"
        self.labels = labels
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.transform = torchaudio.transforms.Resample(orig_freq=1600, new_freq=8000)

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

    def get_prediction(self, tensor, device, model, get_likely_index, index_to_label):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = self.transform(tensor)
        tensor = model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        str_label = index_to_label(tensor.squeeze())
        return str_label

    def get_prediction(self):
        pass
