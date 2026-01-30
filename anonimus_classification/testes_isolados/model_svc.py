# model_svc.py
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import json
import matplotlib.pyplot as plt # Import for plotting (if desired, though not directly in original request)

# --- Configurações ---
DATA_PATH = '02.csv'
OUTPUT_DIR = 'results_svc'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega e prepara dados
data = pd.read_csv(DATA_PATH).rename(columns={'0': 'target'})
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Grade de hiperparâmetros
C_list = [0.1, 1, 10]
kernel_list = ['linear', 'rbf']

records = []
t_start_total = time.time()

melhor_resultado_svc = {'accuracy': 0} # Initialize for SVC

for C in C_list:
    for kernel in kernel_list:
        print(f"\n>> Treinando SVC: C={C}, kernel='{kernel}'")
        t0 = time.time()

        clf = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42
        )
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)
        # probabilidades para ROC-AUC
        try:
            proba = clf.predict_proba(X_test)
            if y.nunique() == 2:
                roc = roc_auc_score(y_test, proba[:, 1])
            else:
                roc = roc_auc_score(y_test, proba, multi_class='ovr')
        except Exception:
            roc = None

        # coleta métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        records.append({
            'C': C,
            'kernel': kernel,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc,
            'train_time_s': train_time
        })

        print(f"— Concluído em {train_time:.2f}s: accuracy={acc:.4f}, roc_auc={roc}")

        # Check and update best result for SVC
        if acc > melhor_resultado_svc['accuracy']:
            melhor_resultado_svc = {
                'tecnica': 'SVC',
                'C': C,
                'kernel': kernel,
                'accuracy': acc
            }

# Salva tudo num CSV
df = pd.DataFrame.from_records(records)
csv_path = os.path.join(OUTPUT_DIR, 'metrics_over_params.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ Métricas salvas em: {csv_path}")

# --- Salva o melhor resultado em JSON ---
json_path = os.path.join('resultados.json')

all_results = {}
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        content = f.read().strip()
        if content:
            all_results = json.loads(content)

all_results['SVC'] = melhor_resultado_svc # Store SVC best result

with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Melhor resultado para SVC salvo em: {json_path}")

t_total = time.time() - t_start_total
print(f"Tempo total de execução: {t_total:.2f}s")