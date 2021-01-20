import speech_recognition as sr

test_wav = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.wav"

test_transcription = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/test.txt"

def userInput(path_to_wav):
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)

        with open(path_to_wav, "wb") as f:
            f.write(audio.get_wav_data())
        return path_to_wav

def transcribeInput(test_wav, test_transcription):
    r = sr.Recognizer()
    transcription = sr.AudioFile(userInput(test_wav))
    with transcription as source:
        audio = r.record(source)

        with open(test_transcription, "w") as f:
            f.write(r.recognize_google(audio))
        return test_transcription


transcribeInput(test_wav, test_transcription)