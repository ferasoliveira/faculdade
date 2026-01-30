import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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
OUTPUT_DIR = 'results_linear_svc'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega e prepara dados
data = pd.read_csv(DATA_PATH).rename(columns={'0': 'target'})
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Grade de hiperparâmetros
C_list = [0.01, 0.1, 1, 10]
loss_list = ['hinge', 'squared_hinge']

records = []
t_start_total = time.time()

melhor_resultado_linear_svc = {'accuracy': 0}

for C in C_list:
    for loss in loss_list:
        print(f"\n>> Treinando LinearSVC: C={C}, loss='{loss}'")
        t0 = time.time()

        clf = LinearSVC(
            C=C,
            loss=loss,
            random_state=42,
            dual=True # 'auto' in newer versions, True or False explicitly to avoid warning
        )
        try:
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            y_pred = clf.predict(X_test)
            # LinearSVC does not directly provide predict_proba, use decision_function
            # For ROC-AUC, decision_function scores can be used if it's a binary classification
            roc = None
            if y.nunique() == 2:
                try:
                    scores = clf.decision_function(X_test)
                    roc = roc_auc_score(y_test, scores)
                except Exception as e:
                    print(f"Could not calculate ROC-AUC for LinearSVC: {e}")
            else:
                print("ROC-AUC for LinearSVC is typically for binary classification.")


            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            records.append({
                'C': C,
                'loss': loss,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'roc_auc': roc,
                'train_time_s': train_time
            })

            print(f"— Concluído em {train_time:.2f}s: accuracy={acc:.4f}, roc_auc={roc}")

            if acc > melhor_resultado_linear_svc['accuracy']:
                melhor_resultado_linear_svc = {
                    'tecnica': 'LinearSVC',
                    'C': C,
                    'loss': loss,
                    'accuracy': acc
                }
        except ValueError as e:
            print(f"Erro durante o treinamento de LinearSVC com C={C}, loss='{loss}': {e}")
            # Append a record with None for metrics if training fails
            records.append({
                'C': C,
                'loss': loss,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'train_time_s': None
            })


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

all_results['LinearSVC'] = melhor_resultado_linear_svc

with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Melhor resultado para LinearSVC salvo em: {json_path}")

t_total = time.time() - t_start_total
print(f"Tempo total de execução: {t_total:.2f}s")