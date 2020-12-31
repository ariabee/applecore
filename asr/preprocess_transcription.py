import re
'''
Method to preprocess the data: removing the numerical labels and leaving us with the transcribed speech
'''
#sample_train_transcription = "/home/morgan/Documents/saarland/fourth_semester/lap_software_project/project/asr/LibriSpeech/train-clean-100/19/198/sample_training_text.txt"
def preprocess(path_to_transcription):
    #initialize path of transcription
    sample_train_transcription = path_to_transcription

    #open and read the transcription
    sample_train_transcription = open(sample_train_transcription, "r")
    sample_train_transcription = sample_train_transcription.read()

    #preprocess using a regex to remove the identifying labels and just have the transcribed speech
    sample_train_preprocessed = re.sub("^.{0,12}", "", sample_train_transcription)
    #print(sample_train_preprocessed)
    return sample_train_preprocessed

#preprocess(sample_train_transcription)