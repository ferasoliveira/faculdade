import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Função para calcular a FFT (Fast Fourier Transform)
def calcular_fft(tempo, sinal):
    # Número total de pontos no sinal
    n = len(tempo)

    # Frequência de amostragem
    fs = 1.0 / (tempo.iloc[1] - tempo.iloc[0])

    # Calcula a FFT
    fft_resultado = np.fft.fft(sinal)

    # Calcula as frequências correspondentes
    frequencias = np.fft.fftfreq(n, 1 / fs)

    # A FFT retorna frequências simétricas, então pegamos apenas a parte positiva
    metade_n = n // 2
    frequencias_positivas = frequencias[:metade_n]
    fft_positiva = 2.0 / n * fft_resultado[:metade_n]

    return frequencias_positivas, fft_positiva


# Nome do arquivo CSV gerado
arquivo2 = "Parado bruto/media_1_parado.csv"
arquivo4 = "Parado bruto/media_10_parado.csv"
arquivo6 = "Parado bruto/media_50_parado.csv"

dados2 = pd.read_csv(arquivo2)
dados2 = dados2[dados2["Tempo"] >= 2.0]
tempo2 = dados2["Tempo"]
acc_y2 = dados2["Acc_Y"]

dados4 = pd.read_csv(arquivo4)
dados4 = dados4[dados4["Tempo"] >= 2.0]
tempo4 = dados4["Tempo"]
acc_y4 = dados4["Acc_Y"]

dados6 = pd.read_csv(arquivo6)
dados6 = dados6[dados6["Tempo"] >= 2.0]
tempo6 = dados6["Tempo"]
acc_y6 = dados6["Acc_Y"]

# Cria um gráfico com os dados no domínio do tempo
plt.figure(figsize=(10, 6))
plt.plot(tempo2, acc_y2, label="Acc - Media 1")
plt.plot(tempo4, acc_y4, label="Acc - Media 10")
plt.plot(tempo6, acc_y6, label="Acc - Media 50")
plt.xlabel("Tempo (s)")
plt.ylabel("Valores")
plt.title("Dados do Arduino - Domínio do Tempo")
plt.legend()

# Cria uma segunda figura com dois subplots no domínio da frequência
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Gráfico 1: Amplitude no domínio da frequência
freq2, fft2 = calcular_fft(tempo2, acc_y2)
freq4, fft4 = calcular_fft(tempo4, acc_y4)
freq6, fft6 = calcular_fft(tempo6, acc_y6)

axs[0].plot(freq2, np.abs(fft2), label="Acc - Media 1")
axs[0].plot(freq4, np.abs(fft4), label="Acc - Media 10")
axs[0].plot(freq6, np.abs(fft6), label="Acc - Media 50")
axs[0].set_xlabel("Frequência (Hz)")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Espectro de Frequência - Amplitude")
axs[0].legend()

# Gráfico 2: Fase no domínio da frequência
axs[1].plot(freq2, np.angle(fft2), label="Acc - Media 1")
axs[1].plot(freq4, np.angle(fft4), label="Acc - Media 10")
axs[1].plot(freq6, np.angle(fft6), label="Acc - Media 50")
axs[1].set_xlabel("Frequência (Hz)")
axs[1].set_ylabel("Fase (radianos)")
axs[1].set_title("Espectro de Frequência - Fase")
axs[1].legend()

plt.tight_layout()

plt.subplots_adjust(
    top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25
)  # Ajusta as margens e o espaço vertical entre os subplots
plt.show()
