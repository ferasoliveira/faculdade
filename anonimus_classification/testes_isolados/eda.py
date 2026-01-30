import pandas as pd
import os
import time

# Caminho para os dados
data_path = '02.csv'
output_dir = 'results_EDA'
os.makedirs(output_dir, exist_ok=True)

# Inicia contagem de tempo
t_start = time.time()
print("Iniciando Análise Exploratória de Dados (EDA)...")

# Carrega os dados
df = pd.read_csv(data_path)

# Renomear coluna alvo
if '0' in df.columns:
    df = df.rename(columns={'0': 'target'})

# Import para visualizações
import matplotlib.pyplot as plt
import seaborn as sns

# Distribuição da variável target
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Distribuição da Variável Target')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

# Identificação de colunas irrelevantes/redundantes
df_features = df.drop('target', axis=1)
zero_variance_cols = df_features.columns[df_features.var() == 0]

duplicated_cols = []
for i in range(df_features.shape[1]):
    for j in range(i + 1, df_features.shape[1]):
        if df_features.iloc[:, i].equals(df_features.iloc[:, j]):
            duplicated_cols.append(df_features.columns[j])

# Cálculo das variâncias para todas as colunas de features
variances = df_features.var()
non_zero_var = variances[variances > 0]

# Boxplot da distribuição de variâncias
plt.figure(figsize=(6, 4))
sns.boxplot(x=variances)
plt.title('Boxplot das Variâncias das Colunas')
plt.xlabel('Variância')
plt.savefig(os.path.join(output_dir, 'variance_boxplot.png'))
plt.close()

# Identificar coluna com menor variância (não zero)
min_var_col = None
min_var_val = None
if not non_zero_var.empty:
    min_var_col = non_zero_var.idxmin()
    min_var_val = non_zero_var.min()

# Escrita dos achados
with open(os.path.join(output_dir, 'eda_findings.txt'), 'w') as f:
    f.write('Achados da Análise Exploratória de Dados:\n\n')
    f.write(f'Número total de colunas: {df.shape[1]}\n')
    f.write(f'Número de linhas: {df.shape[0]}\n')
    f.write(f'Colunas com variância zero (irrelevantes): {list(zero_variance_cols)}\n')
    f.write(f'Colunas duplicadas (redundantes): {list(duplicated_cols)}\n')
    if min_var_col:
        f.write(f'Coluna com menor variância (não zero): {min_var_col} (variância = {min_var_val})\n')
    f.write('\nJustificativa para remoção:\n')
    f.write('Colunas com variância zero não fornecem informação para o modelo.\n')
    f.write('Colunas duplicadas podem causar multicolinearidade e devem ser removidas.\n')
    if min_var_col:
        f.write('Colunas com variância muito baixa podem contribuir pouco para o modelo e ser candidatas à remoção.\n')

# Fim da análise e cálculo de tempo
t_end = time.time()
elapsed = t_end - t_start

print(f"EDA concluída. Resultados em {output_dir}")
print(f"Tempo de execução da EDA: {elapsed:.2f} segundos")
