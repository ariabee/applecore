import speech_recognition as sr
import torch
import torch.nn as nn
from m5 import M5
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import os

test_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/bed.wav"

path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/speech_commands_model/speech_commands_model.pt"

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

model = M5(n_input=1, n_output=35)
model.load_state_dict(torch.load(path_to_local_model))


new_sample_rate = 8000
sample_rate = 1600
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def userInput(path_to_wav):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

        with open(path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
        return path_to_wav

def inputLoad(path_to_wav):
    waveform, sample_rate = torchaudio.load(userInput(path_to_wav))
    return waveform

def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    print(tensor)
    tensor = get_likely_index(tensor)
    print(tensor)
    tensor = index_to_label(tensor.squeeze())
    print(tensor)
    return tensor


waveform = inputLoad(test_wav)
#utterance = "go"
predict(waveform)
#print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")




