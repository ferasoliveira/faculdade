import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from filterpy.kalman import KalmanFilter
import numpy as np

# Leitura dos arquivos
arquivo2 = "1 metro bruto/media_50_1m.csv"
dados2 = pd.read_csv(arquivo2)


# Função para aplicar filtro passa-baixa
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


# Função para aplicar filtro de Kalman
def kalman_filter(data):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement matrix
    kf.P *= 1e2  # Covariance matrix
    kf.R = 1  # Measurement noise
    kf.Q = np.array([[1, 0], [0, 1]])  # Process noise

    # Initialize state
    kf.x = np.array([data.iloc[0], 0])

    filtered_state_means = np.zeros_like(data)

    # Apply Kalman filter
    for i in range(len(data)):
        kf.predict()
        kf.update(data.iloc[i])
        filtered_state_means[i] = kf.x[0]

    return filtered_state_means.flatten()


# Aplicação dos filtros
fs = 1 / (dados2["Tempo"].iloc[1] - dados2["Tempo"].iloc[0])
cutoff_lowpass = 1.0  # Filtro passa-baixa

dados2["Acc_Y_Filtro_PassLow"] = butter_lowpass_filter(
    dados2["Acc_Y"], cutoff_lowpass, fs
)
dados2["Acc_Y_Filtro_Kalman"] = kalman_filter(dados2["Acc_Y"])

# Cálculo dos ruídos removidos
ruído_passlow = dados2["Acc_Y"] - dados2["Acc_Y_Filtro_PassLow"]
ruído_kalman = dados2["Acc_Y"] - dados2["Acc_Y_Filtro_Kalman"]

# Plotagem dos resultados no domínio do tempo
plt.figure(figsize=(10, 12))

plt.subplot(511)
plt.plot(dados2["Tempo"], dados2["Acc_Y"], label="Original")
plt.title("Aceleração Original")

plt.subplot(512)
plt.plot(dados2["Tempo"], dados2["Acc_Y_Filtro_PassLow"], label="Passa-baixa")
plt.title("Filtro Passa-baixa")

plt.subplot(513)
plt.plot(dados2["Tempo"], dados2["Acc_Y_Filtro_Kalman"], label="Kalman")
plt.title("Filtro de Kalman")

plt.subplot(514)
plt.plot(dados2["Tempo"], ruído_passlow, label="Ruído Passa-baixa")
plt.title("Ruído Removido por Passa-baixa")

plt.subplot(515)
plt.plot(dados2["Tempo"], ruído_kalman, label="Ruído Kalman")
plt.title("Ruído Removido por Kalman")

# Adiciona um espaço entre os gráficos
plt.subplots_adjust(hspace=0.5)

# Criação de uma nova figura para os gráficos no domínio da frequência
plt.figure(figsize=(10, 12))

# Plotagem dos resultados no domínio da frequência (Magnitude)
plt.subplot(311)
frequencies = np.fft.fftfreq(len(dados2["Tempo"]), 1 / fs)
fft_original = np.fft.fft(dados2["Acc_Y"])
fft_passlow = np.fft.fft(dados2["Acc_Y_Filtro_PassLow"])
fft_kalman = np.fft.fft(dados2["Acc_Y_Filtro_Kalman"])

plt.plot(frequencies, np.abs(fft_original), label="Original")
plt.plot(frequencies, np.abs(fft_passlow), label="Passa-baixa")
plt.plot(frequencies, np.abs(fft_kalman), label="Kalman")
plt.title("Magnitude no Domínio da Frequência")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")
plt.legend()

# Plotagem dos resultados no domínio da frequência (Ruído)
plt.subplot(312)
fft_ruído_passlow = np.fft.fft(ruído_passlow)
plt.plot(frequencies, np.abs(fft_ruído_passlow), label="Ruído Passa-baixa")
plt.title("Ruído Removido por Passa-baixa")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")

plt.subplot(313)
fft_ruído_kalman = np.fft.fft(ruído_kalman)
plt.plot(frequencies, np.abs(fft_ruído_kalman), label="Ruído Kalman")
plt.title("Ruído Removido por Kalman")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude")

# Adiciona um espaço entre os gráficos
plt.subplots_adjust(hspace=0.5)

plt.show()
