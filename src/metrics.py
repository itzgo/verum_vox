from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calcular_eer(y_true, y_scores):
    """
    Calcula a Equal Error Rate (EER), onde a Taxa de Falsos Positivos (FPR)
    é igual à Taxa de Falsos Negativos (FNR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    # Encontra o ponto onde FPR e FNR se cruzam
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer