# 🧠 Projeto: Anonimus Classification

Este projeto explora diferentes algoritmos de classificação em um conjunto de dados anônimo, utilizando técnicas como **Regressão Logística incremental (SGD)**, **Random Forest** e **SVM (SVC)**. O foco está na **avaliação de desempenho por hiperparâmetro**, com geração automática de métricas e gráficos para análise comparativa.

---

## 📁 Estrutura do Projeto
```
📦 Anonimus_Classification/
├── model_logistic.py
├── model_random_forest.py
├── model_svc.py
├── 02.csv # Base de dados de entrada
├── results_logistic/ # Resultados do modelo SGDClassifier
├── results_random_forest/ # Resultados do Random Forest
├── results_svc/ # Resultados do SVC
└── README.md
```

---

## ⚙️ Modelos Implementados

### 1. `SGDClassifier` (Logistic Regression)
- Treinamento incremental com `partial_fit`
- Hiperparâmetro variado: `C` (regularização inversa)
- Métricas analisadas por **época** (até 500)
- Salva gráficos `.png` e um CSV com as métricas

### 2. `RandomForestClassifier`
- Variação dos parâmetros:
  - `n_estimators`: número de árvores
  - `max_depth`: profundidade máxima das árvores
- Coleta métricas globais e tempo de treino

### 3. `SVC` (Support Vector Classifier)
- Variação dos parâmetros:
  - `C`: penalidade por erro
  - `kernel`: tipo de separação (linear, rbf)
- Mede desempenho com base em predição e `roc_auc`

---

## 📊 Métricas Avaliadas

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC** (quando aplicável)
- **Tempo de execução** por configuração

Cada script salva as métricas por configuração em um arquivo `.csv` na respectiva pasta de resultados, além de gerar gráficos por métrica (no caso do SGD).

---

## ▶️ Como Executar

1. Instale as dependências:
```bash
pip install pandas scikit-learn matplotlib

``` 
📌 Objetivos do Projeto
Avaliar o impacto de hiperparâmetros nos principais algoritmos de classificação.

Gerar experimentos rastreáveis, com salvamento automático de gráficos e métricas.

Construir uma base sólida para comparação de modelos em contextos supervisionados.

🧪 Próximos Passos
Adicionar modelos de Boosting (XGBoost, LightGBM)

Implementar Grid Search ou Random Search automatizado

Realizar validação cruzada k-fold

Empacotar como ferramenta interativa via CLI ou notebook

🧑‍💻 Autor
Fernando Artur Souza de Oliveira