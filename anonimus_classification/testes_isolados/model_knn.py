import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import json
import matplotlib.pyplot as plt

# --- Configurações ---
DATA_PATH = '02.csv'
OUTPUT_DIR = 'results_knn'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega e prepara dados
data = pd.read_csv(DATA_PATH).rename(columns={'0': 'target'})
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Grade de hiperparâmetros
n_neighbors_list = [3, 5, 7, 9]
weights_list = ['uniform', 'distance']
metric_list = ['euclidean', 'manhattan']

records = []
t_start_total = time.time()

melhor_resultado_knn = {'accuracy': 0}

for n_neighbors in n_neighbors_list:
    for weights in weights_list:
        for metric in metric_list:
            print(f"\n>> Treinando KNN: n_neighbors={n_neighbors}, weights='{weights}', metric='{metric}'")
            t0 = time.time()

            clf = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = clf.predict(X_test)
            try:
                proba = clf.predict_proba(X_test)
                if y.nunique() == 2:
                    roc = roc_auc_score(y_test, proba[:, 1])
                else:
                    roc = roc_auc_score(y_test, proba, multi_class='ovr')
            except Exception:
                roc = None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            records.append({
                'n_neighbors': n_neighbors,
                'weights': weights,
                'metric': metric,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc,
                'train_time_s': train_time
            })

            print(f"— Concluído em {train_time:.2f}s: accuracy={acc:.4f}, roc_auc={roc}")

            if acc > melhor_resultado_knn['accuracy']:
                melhor_resultado_knn = {
                    'tecnica': 'KNeighborsClassifier',
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'metric': metric,
                    'accuracy': acc
                }

df = pd.DataFrame.from_records(records)
csv_path = os.path.join(OUTPUT_DIR, 'metrics_over_params.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ Métricas salvas em: {csv_path}")

json_path = os.path.join('resultados.json')
all_results = {}
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        content = f.read().strip()
        if content:
            try:
                all_results = json.loads(content)
            except json.JSONDecodeError:
                print("Aviso: O arquivo JSON existente está corrompido ou vazio. Será sobrescrito.")

all_results['KNeighborsClassifier'] = melhor_resultado_knn

with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Melhor resultado para KNN salvo em: {json_path}")

t_total = time.time() - t_start_total
print(f"Tempo total de execução: {t_total:.2f}s")