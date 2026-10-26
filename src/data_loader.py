import numpy as np
from src.carregar_protocolo import carregar_protocolo
from src.extractData import extract_data


def preparar_dados(caminho_txt, pasta_audios, porcentagem=0.05, proporcao_treino=0.8):
    """ Orquestra a leitura, divisão e extração de características """
    caminhos, rotulos = carregar_protocolo(caminho_txt, pasta_audios, porcentagem)

    limite = int(len(caminhos) * proporcao_treino)
    cam_tr, rot_tr = caminhos[:limite], rotulos[:limite]
    cam_te, rot_te = caminhos[limite:], rotulos[limite:]

    print("\n[Dados] Extraindo características de Treino...")
    X_train = np.array([extract_data(arq) for arq in cam_tr])
    y_train = np.array(rot_tr)

    print("[Dados] Extraindo características de Teste...")
    X_test = np.array([extract_data(arq) for arq in cam_te])
    y_test = np.array(rot_te)

    return X_train, y_train, X_test, y_test