import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def extract_data(caminho_audio, n_mfcc=13):
    """
    Extrai a média dos coeficientes MFCCs de um arquivo de áudio.
    Se o arquivo estiver corrompido, retorna um vetor de zeros para não quebrar o treino.
    """
    try:
        # Tenta carregar o áudio
        y, sr = librosa.load(caminho_audio, sr=16000)
        
        # Se o áudio for vazio/0 bytes por algum erro de download
        if len(y) == 0:
            return np.zeros(n_mfcc)
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
        
    except Exception as e:
        print(f"\n[Aviso] Pulando arquivo corrompido ou ilegível: {caminho_audio}")
        print(f"Erro: {e}")
        return np.zeros(n_mfcc)