# Verum Vox: Detecção Robusta de Vozes Sintéticas via Análise de Artefatos Acústicos

Este repositório contém o código-fonte, dados e documentação do **Trabalho de Conclusão de Curso (TCC)** em Ciência da Computação pela **Universidade Estadual do Ceará (UECE)**.

O projeto investiga e desenvolve protótipos computacionais para a **detecção robusta e agnóstica a idioma de vozes sintéticas**, utilizando a extração de artefatos acústicos e espectrais do sinal de áudio.

---

## 📌 Contexto e Hipótese Central

Com o avanço de tecnologias de *Text-to-Speech* (TTS), *Voice Conversion* (VC) e modelos generativos (*vocoders* neurais), a distinção entre vozes humanas (*bonafide*) e sintéticas (*spoof*) tornou-se um desafio crítico de segurança e autenticidade.

A hipótese central deste trabalho prega que o processo de síntese digital deixa **assinaturas digitais imperceptíveis ao ouvido humano** — tais como anomalias de fase e distorções na representação espectral. O objetivo é demonstrar que modelos treinados nessas características acústicas e temporais conseguem **generalizar a detecção de forma robusta**, independentemente do idioma, conteúdo fonético ou degradações aplicadas ao áudio.

---

## 🎯 Objetivos do Projetos

### Objetivo Geral
Desenvolver e avaliar protótipos computacionais para a detecção de vozes sintéticas, analisando a eficácia da combinação de representações espectrais e algoritmos de Aprendizado de Máquina (*Machine Learning* e *Deep Learning*) na generalização fora do domínio e na resiliência a ruídos.

### Objetivos Específicos
* **Modelagem e Baselines:** Construir e calibrar modelos *baseline* (*Support Vector Machine* - SVM) e redes neurais (*Perceptron* Multicamadas - MLP e Redes Neurais Convolucionais - CNN alimentadas por espectrogramas), incluindo abordagem integrada com áudio bruto (*wave-assisted*).
* **Generalização Cross-Dataset e Zero-Shot:** Testar o desempenho em múltiplos algoritmos de síntese vocal não apresentados durante o treinamento.
* **Avaliação de Robustez a Degradações:** Mensurar o impacto de ruídos de fundo, compressões de áudio (MP3, AAC) e atenuações de frequência no sinal.
* **Análise do Custo-Benefício Computacional:** Comparar o consumo de recursos (memória, tempo de inferência e número de parâmetros) versus os ganhos de precisão obtidos.

---

## 🔬 Planejamento Experimental

O plano experimental do projeto está estruturado em quatro etapas principais:

| Experimento | Foco | Descrição |
| :--- | :--- | :--- |
| **E1** | In-Domain Performance | Treinamento no *ASVspoof 2019 LA Train*, seleção via *Dev* e avaliação no *Eval*. |
| **E2** | Cross-Dataset Generalization | Avaliação direta dos modelos treinados no *ASVspoof 2021 DF* (sem retreinamento). |
| **E3** | Zero-Shot Synthetic Generators | Teste de generalização contra novos geradores sintéticos e *vocoders* neurais modernos (*WaveFake* / *MLAAD-tiny*). |
| **E4** | Robustness to Degraded Audio | Avaliação do impacto de compressões de áudio, atenuações e ruídos nos limites de decisão. |

---

## 📊 Datasets Utilizados

* **ASVspoof 2019 - Logical Access (LA):** Dataset principal de treinamento e validação padrão em TTS/VC.
* **ASVspoof 2021 - DeepFake (DF):** Validação de generalização e resiliência a degradações por compressão.
* **MLAAD-tiny:** Suporte para avaliação de comportamento multilingue.
* **WaveFake:** Avaliação contra *vocoders* neurais modernos (ex.: MelGAN, HiFi-GAN).

---

## 📈 Métricas de Avaliação

A validação dos experimentos utiliza as seguintes métricas de desempenho:

* **EER (*Equal Error Rate*):** Métrica principal do projeto, indicando o ponto de igualdade entre Falsos Positivos e Falsos Negativos.
* **ROC-AUC:** Avaliação da separabilidade entre as classes *bonafide* e *spoof*.
* **F1-Score:** Complemento de precisão e *recall*.
* **Balanced Accuracy:** Utilizada para mitigar impactos de classes desbalanceadas nos conjuntos de teste.
* **min t-DCF (*minimum tandem Detection Cost Function*):** Utilizado em conformidade com o protocolo ASVspoof.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Processamento de Áudio:** `librosa`, `scipy`
* **Machine Learning & Deep Learning:** `scikit-learn`, `PyTorch` / `TensorFlow`
* **Análise de Dados:** `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## 👨‍💻 Autor

* **Ítalo Gothardo** — *Graduando em Ciência da Computação* — Universidade Estadual do Ceará (UECE)