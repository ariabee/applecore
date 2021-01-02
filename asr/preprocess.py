import re
import os
import torchaudio
'''
Method to preprocess the data: 
'''
sample_train = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/voice-controlled-world-game/asr/sample/"
def preprocess_transcription(path_to_transcription):

    for filename in os.listdir(path_to_transcription):
        if filename.endswith(".txt"):
            sample_transcription = open(os.path.join(path_to_transcription, filename))
            read_transcription = sample_transcription.read()

            #preprocess using a regex to remove the identifying labels and just have the transcribed speech
            sample_transcription_preprocessed = re.sub("^.{0,12}", "", read_transcription)
            #print(sample_train_preprocessed)
            return sample_transcription_preprocessed

#preprocess_transcription(sample_train)

def preprocess_wav(path_to_wav):
    for filename in os.listdir(path_to_wav):
        if filename.endswith(".flac"):
           # print(os.path.join(path_to_wav, filename))
            sample_wav_preprocessed = os.path.join(path_to_wav, filename)
            return sample_wav_preprocessed


#preprocess_wav(sample_train)