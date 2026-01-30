import serial
import csv
import time

# Configuração da porta serial - ajuste a porta conforme necessário
porta_serial = serial.Serial("COM3", 115200, timeout=1)

# Nome do arquivo CSV
nome_arquivo_csv = "media_100_1m.csv"

# Tempo de leitura em segundos
tempo_leitura_segundos = 10

# Abre o arquivo CSV para escrita
with open(nome_arquivo_csv, "w", newline="") as arquivo_csv:
    # Cria um objeto de gravação CSV
    gravador_csv = csv.writer(arquivo_csv)

    # Escreve o cabeçalho no arquivo CSV
    gravador_csv.writerow(["Tempo", "Acc_Y", "Vel_Y", "Dist_Y"])

    # Tempo inicial
    tempo_inicial = time.time()

    while (time.time() - tempo_inicial) < tempo_leitura_segundos:
        # Lê uma linha da porta serial
        linha_serial = porta_serial.readline().decode("utf-8").strip()

        # Separa os valores
        valores = linha_serial.split(",")

        # Verifica se a linha tem o formato esperado
        if len(valores) == 3:
            tempo_atual = time.time() - tempo_inicial
            accy = float(valores[0])
            vely = float(valores[1])
            disty = float(valores[2])

            # Escreve os dados no arquivo CSV
            gravador_csv.writerow([tempo_atual, accy, vely, disty])

# Fecha a porta serial
porta_serial.close()

print(f"Dados gravados em {nome_arquivo_csv}")
