from src.data_loader import preparar_dados
from src.pipeline import executar_benchmark

if __name__ == "__main__":
    BASE_DIR = "/home/italo/Documentos/verum_vox/data/ASVspoof2021_DF_eval_part00"
    PROTOCOLO = f"{BASE_DIR}/ASVspoof2021.DF.cm.eval.trl.txt"
    PASTA_AUDIOS = f"{BASE_DIR}/flac"

    print("=== VERUM VOX: Inicializando Sistema ===")
    
    # 1. Carrega dados e extrai características (orquestrado pelo data_loader)
    X_train, y_train, X_test, y_test = preparar_dados(PROTOCOLO, PASTA_AUDIOS, porcentagem=0.05)
    
    # 2. Roda a pipeline de ML e gera os relatórios na pasta reportsData
    executar_benchmark(X_train, y_train, X_test, y_test)