import numpy as np
import matplotlib.pyplot as plt
import math

# Valor de Beta
beta = np.pi / 3

# Valor da Altura máxima (em mm)
h = 8

# Valor do Raio da base (em mm)
rb = 16


def desloc_23(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
    elif beta < theta1 <= 2 * beta:
        eq = (2 * beta - theta1) / beta
    else:
        eq = 0
    x = h * ((3 * eq**2) - (2 * eq**3))
    return x


def veloc_23(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
    else:
        eq = 0
    x = (h / beta) * ((6 * eq) - (6 * eq**2))
    if beta < theta1 <= (2 * beta):
        x = -x
    return x


def aceler_23(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
        x = (h / (beta**2)) * (6 - (12 * eq))
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
        x = (h / (beta**2)) * (6 - (12 * eq))
    else:
        eq = 0
        x = 0

    return x


def desloc_345(theta1):
    if 0 <= theta1 <= (beta):
        eq = theta1 / (beta)
    elif (beta) < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / (beta)
    else:
        eq = 0
    x = h * ((10 * eq**3) - (15 * eq**4) + (6 * eq**5))
    return x


def veloc_345(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
    else:
        eq = 0
    x = (h / beta) * ((30 * eq**2) - (60 * eq**3) + (30 * eq**4))
    if beta < theta1 <= (2 * beta):
        x = -x
    return x


def veloc_23(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
    else:
        eq = 0
    x = (h / beta) * ((6 * eq) - (6 * eq**2))
    if beta < theta1 <= (2 * beta):
        x = -x
    return x


def aceler_345(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
        x = (h / (beta**2)) * ((60 * eq) - (180 * eq**2) + (120 * eq**3))
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
        x = (h / (beta**2)) * ((60 * eq) - (180 * eq**2) + (120 * eq**3))
    else:
        eq = 0
        x = 0

    return x


def desloc_4567(theta1):
    if 0 <= theta1 <= (beta):
        eq = theta1 / (beta)
    elif (beta) < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / (beta)
    else:
        eq = 0
    x = h * ((35 * eq**4) - (84 * eq**5) + (70 * eq**6) - (20 * eq**7))
    return x


def veloc_4567(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
    else:
        eq = 0
    x = (h / beta) * (
        (140 * eq**3) - (420 * eq**4) + (420 * eq**5) - (140 * eq**6)
    )
    if beta < theta1 <= (2 * beta):
        x = -x
    return x


def aceler_4567(theta1):
    if 0 <= theta1 <= beta:
        eq = theta1 / beta
        x = (h / (beta**2)) * (
            (420 * eq**2) - (1680 * eq**3) + (2100 * eq**4) - (840 * eq**5)
        )
    elif beta < theta1 <= (2 * beta):
        eq = (2 * beta - theta1) / beta
        x = (h / (beta**2)) * (
            (420 * eq**2) - (1680 * eq**3) + (2100 * eq**4) - (840 * eq**5)
        )
    else:
        eq = 0
        x = 0

    return x


def ang_23(theta1):
    x = math.atan(veloc_23(theta1) / (desloc_23(theta1) + rb))
    return x


def ang_345(theta1):
    x = math.atan(veloc_345(theta1) / (desloc_345(theta1) + rb))
    return x


def ang_4567(theta1):
    x = math.atan(veloc_4567(theta1) / (desloc_4567(theta1) + rb))
    return x


deslocamento_23 = []
velocidade_23 = []
aceleracao_23 = []
ang_pressao_23 = []

deslocamento_345 = []
velocidade_345 = []
aceleracao_345 = []
ang_pressao_345 = []

deslocamento_4567 = []
velocidade_4567 = []
aceleracao_4567 = []
ang_pressao_4567 = []

# Valores de theta
theta_values = np.linspace(0, 2 * np.pi, 1000)

for theta in theta_values:
    deslocamento_23.append(desloc_23(theta))
    velocidade_23.append(veloc_23(theta))
    aceleracao_23.append(aceler_23(theta))
    ang_pressao_23.append(np.degrees(ang_23(theta)))

    deslocamento_345.append(desloc_345(theta))
    velocidade_345.append(veloc_345(theta))
    aceleracao_345.append(aceler_345(theta))
    ang_pressao_345.append(np.degrees(ang_345(theta)))

    deslocamento_4567.append(desloc_4567(theta))
    velocidade_4567.append(veloc_4567(theta))
    aceleracao_4567.append(aceler_4567(theta))
    ang_pressao_4567.append(np.degrees(ang_4567(theta)))
# End for

# Plotagem
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

axs[0].plot(np.degrees(theta_values), deslocamento_23, label="Polinomio 2-3")
axs[0].plot(np.degrees(theta_values), deslocamento_345, label="Polinomio 3-4-5")
axs[0].plot(np.degrees(theta_values), deslocamento_4567, label="Polinomio 4-5-6-7")
axs[0].set_ylabel("Deslocamento em mm")
axs[0].legend()

axs[1].plot(np.degrees(theta_values), velocidade_23, label="Polinomio 2-3")
axs[1].plot(np.degrees(theta_values), velocidade_345, label="Polinomio 3-4-5")
axs[1].plot(np.degrees(theta_values), velocidade_4567, label="Polinomio 4-5-6-7")
axs[1].set_ylabel("Velocidade em mmrad/s")
axs[1].legend()

axs[2].plot(np.degrees(theta_values), aceleracao_23, label="Polinomio 2-3")
axs[2].plot(np.degrees(theta_values), aceleracao_345, label="Polinomio 3-4-5")
axs[2].plot(np.degrees(theta_values), aceleracao_4567, label="Polinomio 4-5-6-7")
axs[2].set_ylabel("Aceleração em mmrad/s^2")
axs[2].set_xlabel("Ângulo (°)")
axs[2].legend()

# Adiciona os rótulos no eixo x para todos os subplots
for ax in axs:
    ax.set_xticks(
        np.degrees(np.linspace(0, 2 * np.pi, 7))
    )  # Define os marcadores nos ângulos desejados

plt.show()

fig_x, ax_x = plt.subplots(figsize=(10, 4))

ax_x.plot(np.degrees(theta_values), ang_pressao_23, label="Ang. Pressão 2-3")
ax_x.plot(np.degrees(theta_values), ang_pressao_345, label="Ang. Pressão 3-4-5")
ax_x.plot(np.degrees(theta_values), ang_pressao_4567, label="Ang. Pressão 4-5-6-7")
ax_x.set_ylabel("Ângulo de pressão (°)")
ax_x.set_xlabel("Ângulo do came (°)")
ax_x.legend()

# Adicionar marcadores nos ângulos desejados
ax_x.set_xticks(np.degrees(np.linspace(0, 2 * np.pi, 7)))

# Mostrar a nova figura com o gráfico do vetor x
plt.show()
