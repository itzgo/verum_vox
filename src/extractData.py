import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def extract_data(caminho_audio, n_mfcc=13):
    """ Extrai MFCCs do áudio. Retorna vetor de zeros se falhar. """
    try:
        y, sr = librosa.load(caminho_audio, sr=16000)
        if len(y) == 0:
            return np.zeros(n_mfcc)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception:
        return np.zeros(n_mfcc)