import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import speech_recognition as sr

test_wav_file = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/61-70968-0030.wav"

def userInput(path_to_wav):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

        with open(path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
        return path_to_wav

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR")

#transcription = asr_model.transcribe_file("./LibriSpeech/dev-clean-2/1272/135031/1272-135031-0003.flac")

#transcription = asr_model.transcribe_file("/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/applecore/asr/userinput_test.wav")

transcription = asr_model.transcribe_file(userInput(test_wav_file))

print(transcription)