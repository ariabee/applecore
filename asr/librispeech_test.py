import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from model import SpeechRecognitionModel
from text_transform import TextTransform
from librispeech import data_processing

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

train_local_dataset = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech/train-laptop/"

path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/librispeech_models/librispeech_server.pt"

test_wav_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/test.wav"

test_flac_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/test.flac"



spectrograms, labels, input_lengths, label_lengths = data_processing(train_local_dataset)
print("labels")
print(labels)

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    #frequency masking is how torchaudio performs SpecAugment for the frequency dimension
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    #time masking is how torchaudio performs SpechAugment for the time dimension
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)

text_transform = TextTransform()


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
model = nn.DataParallel(model, device_ids=[0,1])
model.load_state_dict(torch.load(path_to_local_model, map_location=torch.device('cpu')))

waveform, sample_rate = torchaudio.load(test_wav_file)

spec = train_audio_transforms(waveform)
    #.squeeze(0).transpose(0, 1)
#print(spec.shape)
spec = spec.unsqueeze(0)
spec = spec.transpose(0,1)
#spec = spec.unsqueeze(1)
#print(spec.shape)

#model.eval()
output = model(spec)

decode = []
decodes = []

arg_maxes = torch.argmax(output, dim=1)
#print(arg_maxes)
for i, args in enumerate(arg_maxes):
    print("i")
    print(i)
  #  for j, index in enumerate(args):
        #print("index")
        #print(index)
       # decode.append(index.item())
   # decodes.append(text_transform.int_to_text(decode))


