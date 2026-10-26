import os
import pandas as pd


def gerar_relatorio_final(resultados_metricas, nome_arquivo="benchmark.csv"):
    """ Recebe os dados, imprime a tabela e salva no diretório de relatórios """
    pasta_reports = "src/reportsData"
    os.makedirs(pasta_reports, exist_ok=True)

    df = pd.DataFrame(resultados_metricas)
    df = df[["Modelo", "EER", "ROC-AUC", "F1-Score", "Balanced_Accuracy"]]

    df_visual = df.copy()
    df_visual["EER"] = df_visual["EER"].apply(lambda x: f"{x*100:.2f}%")

    print("\n" + "=" * 70)
    print(" 📊 RELATÓRIO COMPARATIVO DE DETECÇÃO (VERUM VOX)")
    print("=" * 70)
    print(df_visual.to_string(index=False))
    print("=" * 70)

    caminho_csv = os.path.join(pasta_reports, nome_arquivo)
    df.to_csv(caminho_csv, index=False)
    print(f"\n📁 Relatório salvo com sucesso em: {caminho_csv}")