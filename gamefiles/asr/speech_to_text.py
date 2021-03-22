import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import speech_recognition as sr

#path_to_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/gamefiles/tree.wav"

class SpeechToText():
    def __init__(self):
        self.path_to_wav = "transcript.wav"

    def saveAudio(self, audio):
        with open(self.path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())

        #return self


    def getTranscription(self):
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                                   savedir="./pretrained_ASR")
        transcription = asr_model.transcribe_file(self.path_to_wav)
        return transcription.lower()



#with sr.Microphone() as source:
#    stt = SpeechToText()
#    r = sr.Recognizer()
#    audio = r.listen(source, timeout=5)
#    print("audio")
#    print(audio)
#    name = r.recognize_google(audio)
#    print("google works, why can't you?")
#    print(name)
#    stt.saveAudio(audio)
#    name = stt.getTranscription()
#    print(name)