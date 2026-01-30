import os
import pandas as pd

records = []
# Percorre todos os CSVs na pasta
for fname in os.listdir(r'C:\Users\ferna\OneDrive\Área de Trabalho\Faculdade\Trabalho regressão'):
    if fname.lower().endswith('.csv'):
        base = os.path.splitext(fname)[0]
        df = pd.read_csv(r'C:\Users\ferna\OneDrive\Área de Trabalho\Faculdade\Trabalho regressão\{}'.format(fname))
        # Supondo que a coluna de rótulo seja a última
        target = df.columns[-1]
        mode_freq = df[target].value_counts(normalize=True).max()
        records.append({
            'Base de Dados': base,
            'Baseline (maior classe)': mode_freq
        })

# Monta um DataFrame resumido
df_baseline = pd.DataFrame(records).sort_values('Base de Dados')
print(df_baseline)