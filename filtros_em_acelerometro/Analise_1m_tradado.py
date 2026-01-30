import pandas as pd
import matplotlib.pyplot as plt

# Carregar os DataFrames a partir dos arquivos CSV
dados_filtrados_passlow = pd.read_csv("1 metro tratado/1_filtrados_passlow.csv")
dados_filtrados_kalman = pd.read_csv("1 metro tratado/1_filtrados_kalman.csv")


# Função para calcular velocidade e distância
def calcular_velocidade_e_distancia(data):
    # Calcular diferença de tempo entre amostras
    dt = data["Tempo"].diff()

    # Calcular velocidade (integral da aceleração)
    data["Velocidade"] = data["Acc_Y_Filtro"].cumsum() * dt.iloc[1]

    # Calcular distância (integral da velocidade)
    data["Distancia"] = data["Velocidade"].cumsum() * dt.iloc[1]
    data["Distancia"] = data["Distancia"].apply(
        lambda x: max(0, x)
    )  # Considerar apenas valores positivos

    return data


# Calcular velocidade e distância para cada DataFrame
dados_filtrados_passlow = calcular_velocidade_e_distancia(dados_filtrados_passlow)
dados_filtrados_kalman = calcular_velocidade_e_distancia(dados_filtrados_kalman)


# Função para plotar os gráficos
def plotar_graficos(data, titulo):
    plt.figure(figsize=(12, 10))

    # Gráfico de Aceleração
    plt.subplot(311)
    plt.plot(data["Tempo"], data["Acc_Y_Filtro"], label="Aceleração")
    plt.title("Aceleração - " + titulo)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Aceleração")

    # Gráfico de Velocidade
    plt.subplot(312)
    plt.plot(data["Tempo"], data["Velocidade"], label="Velocidade")
    plt.title("Velocidade - " + titulo)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade")

    # Gráfico de Distância
    plt.subplot(313)
    plt.plot(data["Tempo"], data["Distancia"], label="Distância")
    plt.title("Distância - " + titulo)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Distância")

    plt.tight_layout()
    plt.show()


# Plotar gráficos para cada DataFrame
plotar_graficos(dados_filtrados_passlow, "Passa-baixa")
plotar_graficos(dados_filtrados_kalman, "Kalman")
