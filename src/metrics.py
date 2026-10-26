import numpy as np
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    f1_score, 
    balanced_accuracy_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==========================================
# 1. Equal Error Rate (EER)
# ==========================================
def calcular_eer(y_true, y_scores):
    """
    Calcula a Equal Error Rate (EER), indicando o ponto em que 
    a taxa de falsos aceites é igual à taxa de falsos rejeites.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

# ==========================================
# 2. ROC-AUC
# ==========================================
def calcular_roc_auc(y_true, y_scores):
    """
    Calcula a Área sob a curva ROC (ROC-AUC), medindo a separabilidade 
    entre as classes em variados limiares.
    """
    return roc_auc_score(y_true, y_scores)

# ==========================================
# 3. F1-Score
# ==========================================
def calcular_f1(y_true, y_pred):
    """
    F1-Score: Métrica complementar para avaliação de precisão e recall.
    """
    return f1_score(y_true, y_pred)

# ==========================================
# 4. Balanced Accuracy
# ==========================================
def calcular_balanced_accuracy(y_true, y_pred):
    """
    Acurácia balanceada: lida com o desbalanceamento entre amostras reais e sintéticas,
    calculando a média dos acertos por classe.
    """
    return balanced_accuracy_score(y_true, y_pred)

# ==========================================
# 5. min t-DCF (minimum Tandem Detection Cost Function)
# ==========================================
def calcular_min_tdcf(cm_scores, asv_scores=None):
    """
    Calcula o min t-DCF. 
    
    Nota Metodológica: O min t-DCF oficial do protocolo ASVspoof avalia a 
    atuação conjunta do seu modelo (CM - Countermeasure) com um sistema 
    Automático de Verificação de Locutor (ASV).
    
    Para o cálculo real e submissão, será necessário utilizar o script 
    'eval_metrics.py' disponibilizado pela organização do ASVspoof, cruzando 
    os scores gerados pelo seu modelo (cm_scores) com os scores ASV (asv_scores) 
    fornecidos no dataset.
    """
    if asv_scores is None:
        print("[Aviso] min t-DCF ignorado: Requer os scores do sistema ASV para ser calculado.")
        return None
        
    # O código oficial do ASVspoof deve ser plugado aqui futuramente
    pass

# ==========================================
# Função Auxiliar de Relatório
# ==========================================
def gerar_relatorio_metricas(y_true, y_scores, y_pred):
    """
    Calcula e retorna um dicionário com todas as métricas do projeto.
    """
    return {
        "EER": calcular_eer(y_true, y_scores),
        "ROC-AUC": calcular_roc_auc(y_true, y_scores),
        "F1-Score": calcular_f1(y_true, y_pred),
        "Balanced_Accuracy": calcular_balanced_accuracy(y_true, y_pred)
    }