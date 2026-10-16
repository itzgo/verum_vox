import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ==========================================
# 1. Funções de Extração de Características
# ==========================================
def extrair_caracteristicas(caminho_audio, n_mfcc=13):
    """
    Extrai a média dos coeficientes MFCCs de um arquivo de áudio.
    Em versões futuras, podemos adicionar delta, delta-delta e espectrogramas.
    """
    y, sr = librosa.load(caminho_audio, sr=16000) # Reamostragem padrão para 16kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0) # Resumo do áudio pela média dos quadros
    return mfccs_mean

# ==========================================
# 2. Cálculo do EER (Equal Error Rate)
# ==========================================
def calcular_eer(y_true, y_scores):
    """
    Calcula a Equal Error Rate (EER), onde a Taxa de Falsos Positivos (FPR)
    é igual à Taxa de Falsos Negativos (FNR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Encontra o ponto onde FPR e FNR se cruzam
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

# ==========================================
# 3. Pipeline Simplificado de Treino e Avaliação
# ==========================================
def executar_pipeline_v0(caminhos_treino, rotulos_treino, caminhos_teste, rotulos_teste):
    """
    Treina o modelo M0 (SVM) e avalia no conjunto de teste.
    0: Áudio Real (Bonafide)
    1: Áudio Sintético (Spoof)
    """
    print("1. Extraindo características de treino...")
    X_train = np.array([extrair_caracteristicas(arq) for arq in caminhos_treino])
    y_train = np.array(rotulos_treino)

    print("2. Extraindo características de teste...")
    X_test = np.array([extrair_caracteristicas(arq) for arq in caminhos_teste])
    y_test = np.array(rotulos_teste)

    print("3. Treinando o modelo M0 (Baseline SVM)...")
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    print("4. Realizando predições...")
    y_scores = model.predict_proba(X_test)[:, 1] # Probabilidade de ser sintético (1)
    y_pred = model.predict(X_test)

    print("5. Calculando Métricas do Projeto...")
    eer = calcular_eer(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    f1 = f1_score(y_test, y_pred)

    print("-" * 35)
    print(f"Resultados da Versão 0 (M0 - SVM):")
    print(f"• EER (Métrica Principal): {eer * 100:.2f}%")
    print(f"• ROC-AUC:               {auc:.4f}")
    print(f"• F1-Score:              {f1:.4f}")
    print("-" * 35)

    return model

# ==========================================
# 4. Exemplo de Execução (Sintético para Teste)
# ==========================================
if __name__ == "__main__":
    # IMPORTANTE: Subsitua pelos caminhos dos seus arquivos do ASVspoof 2019
    # Para rodar direto de exemplo, usaremos dados fictícios simulados:
    print("--- Executando com dados simulados para validação da estrutura ---")
    
    # 0 = Real, 1 = Sintético
    y_train_sim = [0, 0, 1, 1] * 10
    y_test_sim = [0, 1, 0, 1] * 5

    # Matrizes simuladas de MFCCs (Substitui a leitura de áudios reais no teste simples)
    X_train_sim = np.random.randn(40, 13) 
    X_test_sim = np.random.randn(20, 13)

    # Treino rápido com os vetores de teste
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_sim, y_train_sim)
    scores = svm.predict_proba(X_test_sim)[:, 1]
    preds = svm.predict(X_test_sim)

    eer = calcular_eer(y_test_sim, scores)
    print(f"Exemplo de cálculo de EER: {eer * 100:.2f}%")
    print(f"Exemplo de ROC-AUC: {roc_auc_score(y_test_sim, scores):.4f}")