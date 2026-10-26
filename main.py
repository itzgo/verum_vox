import os
import random
import numpy as np
from sklearn.svm import SVC

# ==========================================
# Importando funções dos módulos
# ==========================================
from src.extractData import extract_data
from src.metrics import gerar_relatorio_metricas


def carregar_protocolo(caminho_txt, pasta_audios, porcentagem=0.05):
    """
    Lê o arquivo de protocolo do ASVspoof 2021 (Teste Cego - 1 Coluna), 
    checa a existência dos áudios e gera rótulos simulados para rodar o pipeline.
    """
    caminhos_validos = []

    if not os.path.exists(caminho_txt):
        raise FileNotFoundError(f"Arquivo de protocolo não encontrado: {caminho_txt}")

    with open(caminho_txt, 'r') as f:
        for linha in f:
            audio_id = linha.strip() # Pega apenas a string da linha (ex: DF_E_2000011)
            
            if not audio_id:
                continue

            caminho_flac = os.path.join(pasta_audios, f"{audio_id}.flac")

            # Verifica se o arquivo existe FISICAMENTE na pasta
            if os.path.exists(caminho_flac):
                caminhos_validos.append(caminho_flac)

    if len(caminhos_validos) == 0:
        raise ValueError(f"Nenhum arquivo .flac do protocolo foi encontrado na pasta {pasta_audios}.")

    # Filtramos a porcentagem desejada
    random.seed(42)
    qtd_amostras = max(1, int(len(caminhos_validos) * porcentagem))
    
    indices_amostrados = random.sample(range(len(caminhos_validos)), qtd_amostras)
    
    caminhos = [caminhos_validos[i] for i in indices_amostrados]
    
    # ATENÇÃO: Como essa base de eval não tem os rótulos no txt, 
    # vamos criar rótulos aleatórios (0 e 1) apenas para o código não quebrar e testarmos o pipeline.
    rotulos = [random.choice([0, 1]) for _ in caminhos]

    print(f"-> Áudios reais encontrados na pasta: {len(caminhos_validos)}.")
    print(f"-> Amostra selecionada para teste ({porcentagem*100}%): {len(caminhos)} áudios.")
    print(f"-> AVISO: Usando rótulos simulados para o ASVspoof 2021 DF.")
    
    return caminhos, rotulos


def executar_pipeline_v0(caminhos_treino, rotulos_treino, caminhos_teste, rotulos_teste):
    """
    Treina o modelo M0 (SVM) e avalia no conjunto de teste.
    0: Áudio Real (Bonafide)
    1: Áudio Sintético (Spoof)
    """
    print("\n1. Extraindo características de treino...")
    X_train = np.array([extract_data(arq) for arq in caminhos_treino])
    y_train = np.array(rotulos_treino)

    print("2. Extraindo características de teste...")
    X_test = np.array([extract_data(arq) for arq in caminhos_teste])
    y_test = np.array(rotulos_teste)

    print("3. Treinando o modelo M0 (Baseline SVM)...")
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)

    print("4. Realizando predições...")
    y_scores = model.predict_proba(X_test)[:, 1] # Probabilidade de ser sintético (1)
    y_pred = model.predict(X_test)

    print("5. Calculando Métricas do Projeto...")
    relatorio = gerar_relatorio_metricas(y_test, y_scores, y_pred)

    print("-" * 45)
    print(f"Resultados da Versão 0 (M0 - SVM) [Amostra 5%]:")
    print(f"• EER (Métrica Principal): {relatorio['EER'] * 100:.2f}%")
    print(f"• ROC-AUC:               {relatorio['ROC-AUC']:.4f}")
    print(f"• F1-Score:              {relatorio['F1-Score']:.4f}")
    print(f"• Balanced Accuracy:     {relatorio['Balanced_Accuracy']:.4f}")
    print("-" * 45)

    return model


# ==========================================
# Execução Real com Dados do Projeto
# ==========================================
if __name__ == "__main__":
    # Caminhos absolutos fornecidos
    BASE_DIR = "/home/italo/Documentos/verum_vox/data/ASVspoof2021_DF_eval_part00"
    
    PROTOCOLO = f"{BASE_DIR}/ASVspoof2021.DF.cm.eval.trl.txt"
    PASTA_AUDIOS = f"{BASE_DIR}/flac"

    print("--- Verum Vox: Iniciando Treino e Avaliação (Versão 0 - 5% Amostragem) ---")
    
    # Carregar x% das amostras do protocolo
    caminhos_totais, rotulos_totais = carregar_protocolo(PROTOCOLO, PASTA_AUDIOS, porcentagem=0.05)
    
    # em 80% para treino e 20% para teste apenas para validar a execução.
    limite = int(len(caminhos_totais) * 0.8)
    
    caminhos_tr = caminhos_totais[:limite]
    rotulos_tr = rotulos_totais[:limite]
    
    caminhos_te = caminhos_totais[limite:]
    rotulos_te = rotulos_totais[limite:]

    print(f"-> Divisão para teste rápido: {len(caminhos_tr)} para treino e {len(caminhos_te)} para teste.")

    modelo_treinado = executar_pipeline_v0(caminhos_tr, rotulos_tr, caminhos_te, rotulos_te)