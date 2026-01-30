import csv
import serial

# Configuração da porta serial
ser = serial.Serial("COM3", 115200)

# Caminho do arquivo CSV
csv_file_path = "codigos.csv"

# Código do cartão mestre
master_card_code = "2699766"


# Inicia o Serial
def aguarda_inicio():
    while ser.in_waiting > 0:
        pass
    print(ser.readline().decode("utf-8").strip())


def esta_lendo():
    return ser.in_waiting


def checa_leitura(code):
    with open(csv_file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == code:
                return row[1]
    return None


def adiciona_codigo(code, letter):
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([code, letter])


def ler_RFID():
    return ser.readline().decode("utf-8").strip()


aguarda_inicio()

while True:
    if esta_lendo() > 0:
        line = ler_RFID()
        letter = checa_leitura(line)
        if letter is not None:
            print(f"Código lido: {line} / Letra: {letter} ")
        else:
            print(f"Erro: Código {line} não encontrado na base")

        if line == master_card_code:
            print("Inserir código para adicionar...")
            line2 = ler_RFID()
            letter2 = checa_leitura(line2)
            if letter2 is not None:
                print(f"Código {line} já registrado, Letra: {letter} ")
            else:
                nome = input("Digite o Nome do cartão:")
                adiciona_codigo(line2, nome)
            continue
