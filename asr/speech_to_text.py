import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import speech_recognition as sr

class SpeechToText():
    def __init__(self, path_to_wav):
        self.path_to_wav = path_to_wav

    def userInput(self, path_to_wav):
        r = sr.Recognizer()

        with sr.Microphone() as source:
            audio = r.listen(source)

            with open(path_to_wav, "wb") as f:
                f.write(audio.get_wav_data())
            return path_to_wav


    def getTranscription(self, userInput, path_to_wav):
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                                   savedir="./pretrained_ASR")
        transcription = asr_model.transcribe_file(userInput(path_to_wav))
        return transcription