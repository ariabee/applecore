import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from model import SpeechRecognitionModel
from text_transform import TextTransform

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

test_url="test-clean"

test_local_dataset = torchaudio.datasets.LIBRISPEECH("/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/LibriSpeech", url=test_url, download=True)

path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/librispeech_models/librispeech_server.pt"

test_wav_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/test.wav"

test_flac_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/test.flac"

text_transform = TextTransform()

def data_processing(data):
    labels = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        label_lengths.append(len(label))

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)


    return labels, label_lengths

labels, label_lengths = data_processing(test_local_dataset)

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    #frequency masking is how torchaudio performs SpecAugment for the frequency dimension
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    #time masking is how torchaudio performs SpechAugment for the time dimension
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)



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
print(spec)

#model.eval()
output = model(spec)

def GreedyDecoder(output, labels, label_lengths, blank_label=28):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    print(decodes, targets)
    return decodes, targets


GreedyDecoder(output, labels, label_lengths)

