import pandas as pd
import matplotlib.pyplot as plt

# Nome do arquivo CSV
nome_arquivo_csv = "1 metro bruto/media_1_1m.csv"

# Lê o arquivo CSV usando o pandas
dados = pd.read_csv(nome_arquivo_csv)


# Função para estimar velocidade e distância
def estimar_velocidade_distancia(dados):
    # Parâmetros de integração
    delta_tempo = dados["Tempo"].diff()
    delta_tempo.iloc[0] = 0  # Define o primeiro delta_tempo como 0

    # Define a função para verificar se a aceleração está fora do intervalo
    def is_aceleracao_fora_do_intervalo(aceleracao):
        return (aceleracao < -0.01) or (aceleracao > 0.01)

    # Inicializa as colunas de velocidade e distância com 0
    dados["Velocidade"] = 0
    dados["Distancia"] = 0

    # Calcula velocidade e distância apenas quando a aceleração está fora do intervalo
    for i in range(1, len(dados)):
        if is_aceleracao_fora_do_intervalo(dados.loc[i, "Acc_Y"]):
            dados.loc[i, "Velocidade"] = (
                dados.loc[i - 1, "Velocidade"] + dados.loc[i, "Acc_Y"] * delta_tempo[i]
            )
            dados.loc[i, "Distancia"] = (
                dados.loc[i - 1, "Distancia"]
                + dados.loc[i, "Velocidade"] * delta_tempo[i]
            )

    return dados


# Estima velocidade e distância
dados_atualizados = estimar_velocidade_distancia(dados)

# Salva os dados atualizados em um novo arquivo CSV
nome_arquivo_csv_atualizado = "avd_media_1_atualizado.csv"
dados_atualizados.to_csv(nome_arquivo_csv_atualizado, index=False)

print(
    f"Velocidade, distância estimadas e dados atualizados salvos em {nome_arquivo_csv_atualizado}"
)
