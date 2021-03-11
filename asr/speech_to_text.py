import speech_recognition as sr
import torch
import torch.nn as nn
from wav2letter_model import WaveToLetter
import torchaudio
from scipy.io.wavfile import read
from ctcdecode import CTCBeamDecoder

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
        sample_rate, sound = read(userInput(path_to_wav))
        sound = sound.astype('float32') / 32767
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)

        return sound


    def getLabels(self):
        labels = WaveToLetter.get_labels(self.model)
        int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        return int_to_char

    def beamDecoder(self):
        decoder = CTCBeamDecoder(
            labels=self.labels,
            model_path=None,
            alpha=0.75,
            beta=1.0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=10,
            num_processes=4,
            blank_id=0
        )
        return decoder

    def getTranscription(self, model, inputLoad, userInput, path_to_wav, beamDecoder):
        model = WaveToLetter.load_model(model)
        model.eval()
        seqlength = inputLoad(userInput(path_to_wav))
        batch_size = 1
        inputs = torch.zeros(batch_size, 1, seqlength)
        output = model(inputs)
        decoder = beamDecoder()
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)
        transcription = "".join([self.labels[n] for n in beam_results[0][0][:out_lens[0][0]]])
        return transcription