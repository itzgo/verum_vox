from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.metrics import gerar_relatorio_metricas
from src.reportsData import gerar_relatorio_final

def executar_benchmark(X_train, y_train, X_test, y_test):
    """ Treina os algoritmos, metrifica as predições e chama o gerador de relatórios """
    modelos = {
        "M0 (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
        "M1 (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "M2 (RF)": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    resultados_relatorio = []

    print("\n[Pipeline] Iniciando Treinamento e Avaliação...")
    for nome, modelo in modelos.items():
        print(f" -> {nome}...")
        modelo.fit(X_train, y_train)
        
        y_probs = modelo.predict_proba(X_test)
        y_scores = y_probs[:, 1]
        y_pred = modelo.predict(X_test)
        
        # Calcula métricas (src/metrics.py)
        metricas = gerar_relatorio_metricas(y_test, y_scores, y_pred)
        metricas["Modelo"] = nome
        resultados_relatorio.append(metricas)

    # Gera e salva a tabela (src/reports.py)
    gerar_relatorio_final(resultados_relatorio, "benchmark_modelos.csv")