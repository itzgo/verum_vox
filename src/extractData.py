import numpy as np
import librosa

def extract_data(caminho_audio, n_mfcc=13):
    """
    Extrai a média dos coeficientes MFCCs de um arquivo de áudio.
    Em versões futuras, podemos adicionar delta, delta-delta e espectrogramas.
    """
    y, sr = librosa.load(caminho_audio, sr=16000) # Reamostragem padrão para 16kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0) # Resumo do áudio pela média dos quadros
    return mfccs_mean