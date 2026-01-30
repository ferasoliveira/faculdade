import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import json

# --- Configurações ---
DATA_PATH = '02.csv'                     
OUTPUT_DIR = 'results_logistic'            
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Início da contagem total
t0_total = time.time()

# Carrega dados
data = pd.read_csv(DATA_PATH).rename(columns={'0': 'target'})
X = data.drop('target', axis=1)
y = data['target']

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Valores de C e épocas
Cs = [0.01, 0.1, 1, 10, 100, 1000]
n_epochs = 100

# Métricas a coletar
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
results = {met: {C: [] for C in Cs} for met in metrics}

# Loop de treinamento incremental
for C in Cs:
    print(f"\n>> Iniciando treinamento para C={C}")
    t0 = time.time()
    
    # alpha = 1/(C * n_samples)
    alpha = 1.0 / (C * X_train.shape[0])
    clf = SGDClassifier(
        loss='log_loss',      # <--- corrigido
        penalty='l2',
        alpha=alpha,
        max_iter=1,
        tol=None,
        shuffle=True,
        random_state=42,
        warm_start=True
    )
    # Inicializa pesos
    clf.partial_fit(X_train, y_train, classes=sorted(y.unique()))
    
    for epoch in range(1, n_epochs + 1):
        clf.partial_fit(X_train, y_train)
        
        # predições
        y_pred = clf.predict(X_test)
        # decision_function (margens)
        scores = clf.decision_function(X_test)
        
        # coleta métricas
        results['accuracy'][C].append(accuracy_score(y_test, y_pred))
        results['precision'][C].append(
            precision_score(y_test, y_pred, average='weighted', zero_division=0))
        results['recall'][C].append(
            recall_score(y_test, y_pred, average='weighted', zero_division=0))
        results['f1_score'][C].append(
            f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # ROC-AUC apenas se binário
        if y.nunique() == 2:
            # para binário podemos usar scores (margens) ou calibrar
            results['roc_auc'][C].append(roc_auc_score(y_test, scores))
        else:
            results['roc_auc'][C].append(None)
    
    dur = time.time() - t0
    print(f"Treinamento C={C} concluído em {dur:.2f}s")

# Geração e salvamento dos gráficos
for met in metrics:
    plt.figure(figsize=(8, 5))
    for C in Cs:
        vals = results[met][C]
        # filtra None (caso multiclass e ROC)
        xs = list(range(1, n_epochs+1))
        ys = [v for v in vals]
        plt.plot(xs, ys, label=f"C={C}")
    plt.xlabel('Épocas')
    plt.ylabel(met.replace('_', ' ').title())
    plt.title(f'{met.replace("_", " ").title()} por Época')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{met}.png")
    plt.savefig(path)
    plt.close()
    print(f"Salvo gráfico: {path}")

records = []
for C in Cs:
    for epoch, (acc, prec, rec, f1, roc) in enumerate(zip(
            results['accuracy'][C],
            results['precision'][C],
            results['recall'][C],
            results['f1_score'][C],
            results['roc_auc'][C]
        ), start=1):
        records.append({
            'C': C,
            'epoch': epoch,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc
        })

# Cria DataFrame e salva em CSV
df_metrics = pd.DataFrame.from_records(records)
csv_path = os.path.join(OUTPUT_DIR, 'metrics_over_epochs.csv')
df_metrics.to_csv(csv_path, index=False)
print(f"Salvo CSV de métricas em: {csv_path}")

melhor_resultado = {'accuracy': 0}

for C in Cs:
    max_acc = max(results['accuracy'][C])
    if max_acc > melhor_resultado['accuracy']:
        melhor_resultado = {
            'tecnica': 'SGDClassifier',
            'C': C,
            'alpha': 1.0 / (C * X_train.shape[0]),
            'accuracy': max_acc
        }

# Encontrar a melhor acurácia e seus parâmetros
melhor_resultado = {'accuracy': 0}

for C in Cs:
    max_acc = max(results['accuracy'][C])
    if max_acc > melhor_resultado['accuracy']:
        melhor_resultado = {
            'tecnica': 'SGDClassifier',
            'C': C,
            'alpha': 1.0 / (C * X_train.shape[0]),
            'accuracy': max_acc
        }

# Caminho do arquivo JSON
json_path = os.path.join('resultados.json')

# Carrega resultados existentes (se houver e se estiver válido)
all_results = {}
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        try:
            content = f.read().strip()
            if content:
                all_results = json.loads(content)
        except json.JSONDecodeError:
            print("Aviso: O arquivo JSON existente está corrompido ou vazio. Será sobrescrito.")

# Atualiza ou adiciona o resultado da técnica
all_results[melhor_resultado['tecnica']] = melhor_resultado

# Salva o arquivo atualizado
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Melhor resultado salvo em: {json_path}")

# Tempo total
t_total = time.time() - t0_total
print(f"\nTempo total de execução: {t_total:.2f}s")