import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from wav2letter_model import WaveToLetter
from text_transform import TextTransform
import json
from ctcdecode import CTCBeamDecoder
import numpy as np
from scipy.io.wavfile import read
from decoder import BeamCTCDecoder

use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")


path_to_local_model = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/corpora/wav2Letter_7.pth.tar"

test_wav_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/librispeech_test.wav"

test_flac_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/test.flac"

test_labels = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/labels.json"


text_transform = TextTransform()
model = WaveToLetter.load_model(path_to_local_model)
labels = WaveToLetter.get_labels(model)
audio_conf = WaveToLetter.get_audio_conf(model)
int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
print(int_to_char)



sample_rate, sound = read(test_wav_file)
sound = sound.astype('float32') / 32767
if len(sound.shape) > 1:
    if sound.shape[1] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=1)


seqlength = sound.shape[0]
batch_size = 1
inputs = torch.zeros(batch_size, 1, seqlength)


model.eval()
output = model(inputs)
#print(output.shape)



decoder = CTCBeamDecoder(
    labels=labels,
    model_path=None,
    alpha=0.75,
    beta=1.0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=10,
    num_processes=4,
    blank_id=0
)

beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)


transcription = "".join([labels[n] for n in beam_results[0][0][:out_lens[0][0]]])
#print(beam_results[0][0][:out_lens[0][0]])
print(transcription)


