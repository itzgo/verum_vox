import os
import random


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
            audio_id = linha.strip()
            if not audio_id:
                continue

            caminho_flac = os.path.join(pasta_audios, f"{audio_id}.flac")
            if os.path.exists(caminho_flac):
                caminhos_validos.append(caminho_flac)

    if len(caminhos_validos) == 0:
        raise ValueError(f"Nenhum arquivo .flac do protocolo foi encontrado na pasta {pasta_audios}.")

    random.seed(42)
    qtd_amostras = max(1, int(len(caminhos_validos) * porcentagem))
    indices_amostrados = random.sample(range(len(caminhos_validos)), qtd_amostras)
    caminhos = [caminhos_validos[i] for i in indices_amostrados]

    # Rótulos simulados (0 e 1) apenas para teste estrutural
    rotulos = [random.choice([0, 1]) for _ in caminhos]

    print(f"-> Áudios reais encontrados na pasta: {len(caminhos_validos)}")
    print(f"-> Amostra selecionada para teste ({porcentagem*100}%): {len(caminhos)} áudios")
    print(f"-> AVISO: Usando rótulos simulados para o ASVspoof 2021 DF.")
    return caminhos, rotulos