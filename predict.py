import librosa
import numpy as np
import os
import math
import hmmlearn.hmm
from pydub import AudioSegment

class_names = ['co', 'giadinh', 'khong', 'toi', 'nguoi']

def get_mfcc(file_path):
#     try:
    y, sr = librosa.load(file_path) # read .wav file
#     except Exception as e:
#         print(e)
#         return
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=13, n_fft=1024,
        hop_length=hop_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T #

import pickle



class Predicter : 
    def __init__(self) : 
        self.model = {}
        for key in class_names:
            name = f"src/models/model_{key}.pkl"
            with open(name, 'rb') as f : 
                self.model[key] = pickle.load(f)

    def predict(self): 
        #Predict
        record_mfcc = get_mfcc("record.wav")
        scores = [self.model[cname].score(record_mfcc) for cname in class_names]
        print(scores)
        pred = np.argmax(scores)
        return class_names[pred]