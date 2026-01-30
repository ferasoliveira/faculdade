import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações Globais ---
# JSON_RESULTS_PATH agora é global para salvar todos os resultados em um único arquivo
JSON_RESULTS_PATH = 'resultados.json'

# Garante que o arquivo JSON exista (ou crie um vazio se não existir ou estiver corrompido)
def load_existing_results():
    if os.path.exists(JSON_RESULTS_PATH):
        with open(JSON_RESULTS_PATH, 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print("Aviso: O arquivo JSON existente está corrompido ou vazio. Será sobrescrito.")
    return {}

def save_results(all_results):
    with open(JSON_RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)

# Carrega todos os resultados existentes uma única vez antes de iniciar o loop
all_global_results = load_existing_results()

# Loop para processar de 00.csv a 23.csv
for n in range(24): # Altere para o número total de arquivos + 1 (ex: 24 para 00-23)
    ARQUIVO = str(n).zfill(2) # Garante que o nome do arquivo tenha sempre 2 dígitos (00, 01, ..., 23)
    DATA_PATH = 'dados/'  + ARQUIVO + '.csv'
    
    # Verifica se o arquivo CSV existe antes de tentar lê-lo
    if not os.path.exists(DATA_PATH):
        print(f"Aviso: Arquivo {DATA_PATH} não encontrado. Pulando para o próximo.")
        continue

    print(f"------------------------------- INICIANDO ARQUIVO: {DATA_PATH} --------------------------------------------------")
    
    # Inicializa o dicionário para a base de dados atual se ainda não existir
    if ARQUIVO not in all_global_results:
        all_global_results[ARQUIVO] = {}

    # Carrega e prepara dados (comum a todos os modelos)
    data = pd.read_csv(DATA_PATH).rename(columns={'0': 'target'})
    X = data.drop('target', axis=1)
    y = data['target'] # Original y, potentially 1-indexed

    # Sempre ajusta rótulos para XGBoost se as classes forem 1-indexed
    # Cria uma cópia para não modificar o 'y' original usado por outros modelos
    y_xgb = y.copy()
    if y_xgb.min() == 1:
        y_xgb = y_xgb - 1 # Transforma 1,2,3,4 em 0,1,2,3

    # Split treino/teste para os modelos que usam 'y' original
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Split treino/teste para XGBoost usando 'y_xgb' (0-indexed)
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X, y_xgb, test_size=0.3, random_state=42, stratify=y_xgb
    )

    # --- Treinamento e Avaliação para cada Modelo ---

    # 1. SGDClassifier (Logistic Regression) - Parâmetros expandidos
    def run_logistic_regression(current_file_results):
        print("\n--- Treinando SGDClassifier (Logistic Regression) ---")
        # Cs: agora com mais granularidade em diferentes escalas
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000] 
        n_epochs = 100
        penalties = ['l2', 'l1', 'elasticnet'] # Adicionado 'l1' e 'elasticnet'
        
        melhor_resultado = {'accuracy': 0}

        for C in Cs:
            alpha = 1.0 / (C * X_train.shape[0])
            for penalty in penalties:
                # l1_ratio é relevante apenas para penalty='elasticnet'
                l1_ratios = [0.15, 0.5, 0.85] if penalty == 'elasticnet' else [None] 
                for l1_ratio in l1_ratios:
                    clf = SGDClassifier(
                        loss='log_loss',
                        penalty=penalty,
                        alpha=alpha,
                        l1_ratio=l1_ratio, # Incluído l1_ratio
                        max_iter=1,
                        tol=None,
                        shuffle=True,
                        random_state=42,
                        warm_start=True
                    )
                    clf.partial_fit(X_train, y_train, classes=sorted(y.unique()))
                    
                    for epoch in range(1, n_epochs + 1):
                        clf.partial_fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        
                        if acc > melhor_resultado['accuracy']:
                            melhor_resultado = {
                                'tecnica': 'SGDClassifier',
                                'C': C,
                                'alpha': alpha,
                                'penalty': penalty, # Salva o penalty
                                'l1_ratio': l1_ratio, # Salva o l1_ratio
                                'accuracy': acc
                            }
        
        current_file_results['SGDClassifier'] = melhor_resultado
        print(f"Melhor resultado para SGDClassifier salvo para {ARQUIVO}")

    # 2. SVC - Parâmetros expandidos
    def run_svc(current_file_results):
        print("\n--- Treinando SVC ---")
        C_list = [0.01, 0.1, 1, 10, 100] # Mais valores para C
        kernel_list = ['linear', 'rbf', 'poly', 'sigmoid'] # Adicionado 'poly' e 'sigmoid'
        gamma_list = ['scale', 'auto', 0.001, 0.01, 0.1, 1] # Adicionado mais valores para gamma
        
        melhor_resultado = {'accuracy': 0}

        for C in C_list:
            for kernel in kernel_list:
                for gamma in gamma_list:
                    # 'gamma' é ignorado para kernel='linear'
                    if kernel == 'linear' and gamma != 'scale': # Evita testes desnecessários
                        continue
                    clf = SVC(
                        C=C,
                        kernel=kernel,
                        gamma=gamma, # Incluído gamma
                        probability=True,
                        random_state=42
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    if acc > melhor_resultado['accuracy']:
                        melhor_resultado = {
                            'tecnica': 'SVC',
                            'C': C,
                            'kernel': kernel,
                            'gamma': gamma, # Salva gamma
                            'accuracy': acc
                        }
        
        current_file_results['SVC'] = melhor_resultado
        print(f"Melhor resultado para SVC salvo para {ARQUIVO}")

    # 3. Random Forest - Parâmetros expandidos
    def run_random_forest(current_file_results):
        print("\n--- Treinando RandomForestClassifier ---")
        n_estimators_list = [50, 100, 200, 300, 500] # Mais estimadores
        max_depth_list = [None, 5, 10, 20, 30] # Mais profundidade
        min_samples_split_list = [2, 5, 10, 20] # Adicionado min_samples_split
        
        melhor_resultado = {'accuracy': 0}

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for min_samples_split in min_samples_split_list:
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split, # Incluído min_samples_split
                        random_state=42,
                        n_jobs=-1
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    if acc > melhor_resultado['accuracy']:
                        melhor_resultado = {
                            'tecnica': 'RandomForestClassifier',
                            'n_estimators': n_estimators,
                            'max_depth': max_depth if max_depth is not None else 'None',
                            'min_samples_split': min_samples_split, # Salva min_samples_split
                            'accuracy': acc
                        }
        
        current_file_results['RandomForestClassifier'] = melhor_resultado
        print(f"Melhor resultado para RandomForestClassifier salvo para {ARQUIVO}")

    # 4. Extra Trees - Parâmetros expandidos
    def run_extra_trees(current_file_results):
        print("\n--- Treinando ExtraTreesClassifier ---")
        n_estimators_list = [50, 100, 200, 400, 600] # Mais estimadores
        max_depth_list = [None, 10, 20, 40] # Mais profundidade
        min_samples_split_list = [2, 5, 10, 15] # Mais valores
        min_samples_leaf_list = [1, 2, 4] # Adicionado min_samples_leaf
        
        melhor_resultado = {'accuracy': 0}

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                for min_samples_split in min_samples_split_list:
                    for min_samples_leaf in min_samples_leaf_list:
                        clf = ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, # Incluído min_samples_leaf
                            random_state=42,
                            n_jobs=-1
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        if acc > melhor_resultado['accuracy']:
                            melhor_resultado = {
                                'tecnica': 'ExtraTreesClassifier',
                                'n_estimators': n_estimators,
                                'max_depth': max_depth if max_depth is not None else 'None',
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf, # Salva min_samples_leaf
                                'accuracy': acc
                            }
        
        current_file_results['ExtraTreesClassifier'] = melhor_resultado
        print(f"Melhor resultado para ExtraTreesClassifier salvo para {ARQUIVO}")

    # 5. KNN - Parâmetros expandidos
    def run_knn(current_file_results):
        print("\n--- Treinando KNeighborsClassifier ---")
        n_neighbors_list = [1, 3, 5, 7, 9, 11, 15] # Mais vizinhos
        weights_list = ['uniform', 'distance']
        metric_list = ['euclidean', 'manhattan', 'minkowski'] # Adicionado 'minkowski'
        p_list = [1, 2] # Para 'minkowski' (p=1 é manhattan, p=2 é euclidean)
        
        melhor_resultado = {'accuracy': 0}

        for n_neighbors in n_neighbors_list:
            for weights in weights_list:
                for metric in metric_list:
                    for p in p_list:
                        if metric != 'minkowski' and p != 2: # p só é relevante para minkowski
                            continue
                        clf = KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            metric=metric,
                            p=p if metric == 'minkowski' else 2, # Define p para minkowski
                            n_jobs=-1
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        if acc > melhor_resultado['accuracy']:
                            melhor_resultado = {
                                'tecnica': 'KNeighborsClassifier',
                                'n_neighbors': n_neighbors,
                                'weights': weights,
                                'metric': metric,
                                'p': p if metric == 'minkowski' else None, # Salva p
                                'accuracy': acc
                            }
        
        current_file_results['KNeighborsClassifier'] = melhor_resultado
        print(f"Melhor resultado para KNeighborsClassifier salvo para {ARQUIVO}")

    # 6. LinearSVC - Parâmetros expandidos
    def run_linear_svc(current_file_results):
        print("\n--- Treinando LinearSVC ---")
        C_list = [0.001, 0.01, 0.1, 1, 10, 100] # Mais valores para C
        loss_list = ['hinge', 'squared_hinge']
        # Adicionado penalty, que é importante para LinearSVC
        penalty_list = ['l1', 'l2'] 

        melhor_resultado = {'accuracy': 0}

        for C in C_list:
            for loss in loss_list:
                for penalty in penalty_list:
                    # Dual=False é preferível quando n_samples > n_features, ou quando penalty='l1'
                    dual_val = True 
                    if penalty == 'l1' or X_train.shape[0] > X_train.shape[1]:
                        dual_val = False

                    # Algumas combinações de perda e penalidade não são suportadas
                    if loss == 'hinge' and penalty == 'l1': # Esta combinação não é suportada
                        continue 

                    clf = LinearSVC(
                        C=C,
                        loss=loss,
                        penalty=penalty, # Incluído penalty
                        random_state=42,
                        dual=dual_val,
                        max_iter=2000 # Aumentar max_iter para convergência
                    )
                    try:
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)

                        if acc > melhor_resultado['accuracy']:
                            melhor_resultado = {
                                'tecnica': 'LinearSVC',
                                'C': C,
                                'loss': loss,
                                'penalty': penalty, # Salva penalty
                                'accuracy': acc
                            }
                    except ValueError as e:
                        print(f"Erro durante o treinamento de LinearSVC com C={C}, loss='{loss}', penalty='{penalty}': {e}")

        current_file_results['LinearSVC'] = melhor_resultado
        print(f"Melhor resultado para LinearSVC salvo para {ARQUIVO}")

    # 7. XGBoost - Parâmetros expandidos
    def run_xgboost(current_file_results):
        print("\n--- Treinando XGBoostClassifier ---")
        n_estimators_list = [100, 300, 500, 800] # Mais estimadores
        learning_rate_list = [0.005, 0.01, 0.05, 0.1, 0.2] # Mais granularidade para learning_rate
        max_depth_list = [3, 5, 7, 9, 11] # Mais profundidade
        subsample_list = [0.7, 0.8, 0.9, 1.0] # Adicionado subsample
        colsample_bytree_list = [0.7, 0.8, 0.9, 1.0] # Adicionado colsample_bytree

        melhor_resultado = {'accuracy': 0}

        for n_estimators in n_estimators_list:
            for learning_rate in learning_rate_list:
                for max_depth in max_depth_list:
                    for subsample in subsample_list:
                        for colsample_bytree in colsample_bytree_list:
                            clf = XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=subsample, # Incluído subsample
                                colsample_bytree=colsample_bytree, # Incluído colsample_bytree
                                eval_metric='mlogloss' if y_xgb.nunique() > 2 else 'logloss', 
                                random_state=42,
                                n_jobs=-1
                            )
                            clf.fit(X_train_xgb, y_train_xgb)
                            y_pred = clf.predict(X_test_xgb)
                            acc = accuracy_score(y_test_xgb, y_pred)

                            if acc > melhor_resultado['accuracy']:
                                melhor_resultado = {
                                    'tecnica': 'XGBoostClassifier',
                                    'n_estimators': n_estimators,
                                    'learning_rate': learning_rate,
                                    'max_depth': max_depth,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'accuracy': acc
                                }
        
        current_file_results['XGBoostClassifier'] = melhor_resultado
        print(f"Melhor resultado para XGBoostClassifier salvo para {ARQUIVO}")

    # 8. Stacking Classifier - Parâmetros expandidos
    def run_stacking_classifier(current_file_results):
        print("\n--- Treinando StackingClassifier ---")
        # Base estimators com mais opções ou com alguns de seus melhores hiperparâmetros
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)),
            ('svc', SVC(C=1, kernel='rbf', probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)) # Adicionado XGB
        ]

        meta_estimator_list = [
            LogisticRegression(random_state=42, solver='liblinear', C=0.1), # Adicionado C
            RandomForestClassifier(n_estimators=75, max_depth=10, random_state=42, n_jobs=-1), # Mais n_estimators
            # Adicionar um terceiro meta-estimador para mais testes, se desejado, ex: GradientBoostingClassifier
        ]

        melhor_resultado = {'accuracy': 0}

        for meta_estimator in meta_estimator_list:
            meta_estimator_name = type(meta_estimator).__name__
            # Tentar diferentes cv valores para o StackingClassifier se o tempo permitir
            cv_values = [3, 5] 
            for cv_val in cv_values:
                clf = StackingClassifier(
                    estimators=estimators,
                    final_estimator=meta_estimator,
                    cv=cv_val, # Testando diferentes cv
                    n_jobs=-1
                )
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                if acc > melhor_resultado['accuracy']:
                    melhor_resultado = {
                        'tecnica': 'StackingClassifier',
                        'meta_estimator': meta_estimator_name,
                        'cv': cv_val, # Salva cv
                        'accuracy': acc
                    }
        
        current_file_results['StackingClassifier'] = melhor_resultado
        print(f"Melhor resultado para StackingClassifier salvo para {ARQUIVO}")

    # --- Bloco de execução dos modelos para o arquivo atual ---
    t_start_current_file = time.time()
    
    # Obtém a referência para o dicionário da base de dados atual
    current_file_results = all_global_results[ARQUIVO] 

    # Executa cada função de treinamento de modelo
    run_logistic_regression(current_file_results)
    run_svc(current_file_results)
    run_random_forest(current_file_results)
    run_extra_trees(current_file_results)
    run_knn(current_file_results)
    run_linear_svc(current_file_results)
    #run_xgboost(current_file_results)
    run_stacking_classifier(current_file_results)

    # Salva todos os resultados acumulados após cada arquivo ser processado
    save_results(all_global_results)
    print(f"Resultados para o arquivo {ARQUIVO} salvos em: {JSON_RESULTS_PATH}")

    t_total_current_file = time.time() - t_start_current_file
    print(f"\nTempo total de execução para o arquivo {ARQUIVO}: {t_total_current_file:.2f} segundos")

print(f"\n------------------------------- TODOS OS ARQUIVOS PROCESSADOS. RESULTADOS FINAIS EM: {JSON_RESULTS_PATH} --------------------------------------------------")

# --- Parte Final: Visualização dos Resultados ---
def visualize_results():
    print("\n--- Gerando Visualização Gráfica dos Resultados ---")
    
    # Carrega todos os resultados finais do arquivo JSON
    final_results = load_existing_results()
    
    # Prepara os dados para o DataFrame
    data_for_plot = []
    for base, models_results in final_results.items():
        for model_name, model_info in models_results.items():
            if 'accuracy' in model_info: # Garante que a chave 'accuracy' exista
                data_for_plot.append({
                    'Base de Dados': base,
                    'Modelo': model_name,
                    'Acurácia': model_info['accuracy']
                })

    df_results = pd.DataFrame(data_for_plot)

    if df_results.empty:
        print("Nenhum resultado de acurácia encontrado para visualização. Certifique-se de que os arquivos CSV existem e o processamento foi bem-sucedido.")
        return

    # Ordenar modelos pelo nome para consistência na visualização
    df_results['Modelo'] = df_results['Modelo'].astype('category')
    df_results['Modelo'] = df_results['Modelo'].cat.set_categories(
        sorted(df_results['Modelo'].unique()), ordered=True
    )
    
    # Ordenar as bases de dados numericamente para o gráfico
    # Convertendo para int para ordenar corretamente (00, 01, ..., 09, 10, ...)
    df_results['Base de Dados_Num'] = pd.to_numeric(df_results['Base de Dados'])
    df_results = df_results.sort_values(by='Base de Dados_Num')
    # Voltando para string formatada para o rótulo do gráfico
    df_results['Base de Dados'] = df_results['Base de Dados_Num'].astype(str).str.zfill(2)

    # Cria o gráfico de barras agrupado
    plt.figure(figsize=(22, 12)) # Aumenta o tamanho da figura para melhor visualização com mais bases
    sns.barplot(x='Base de Dados', y='Acurácia', hue='Modelo', data=df_results, palette='viridis')
    
    plt.title('Comparação de Acurácia dos Modelos por Base de Dados')
    plt.xlabel('Base de Dados')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1) # A acurácia varia de 0 a 1
    plt.xticks(rotation=60, ha='right') # Rotaciona os rótulos do eixo X para melhor legibilidade com mais itens
    # Ajusta a legenda para evitar sobreposição, especialmente com muitos modelos
    plt.legend(title='Modelo', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) 
    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Adiciona um grid horizontal

    # Salva o gráfico em um arquivo
    plot_filename = 'comparacao_modelos_por_base.png'
    plt.savefig(plot_filename)
    print(f"Gráfico de comparação salvo como: {plot_filename}")
    plt.show() # Mostra o gráfico (opcional, pode ser removido se for apenas para salvar)

# Executa a função de visualização após o loop principal e todos os dados serem processados e salvos
if __name__ == "__main__":
    visualize_results()