<<<<<<< HEAD
# Projeto Perceptron

Este projeto implementa um Perceptron de camada única com aprendizado supervisionado binário. O treinamento e teste são realizados com duas bases diferentes:

- Base Linear: Dados linearmente separáveis.
- Base Não-Linear: Dados não linearmente separáveis.

## Estrutura dos Arquivos

```
projeto_perceptron/
│
├── gerador_bases.py
├── main.c
├── linear_modelo.dat
├── nlinear_modelo.dat
├── bases/
│   ├── base_linear_X.csv
│   ├── base_linear_y.csv
│   ├── base_linear.png
│   ├── base_nlinear_X.csv
│   ├── base_nlinear_y.csv
│   └── base_nlinear.png
```

## Requisitos

### Python (para treino)

- Python 3.8 ou superior
- Bibliotecas Python:
  - numpy
  - matplotlib
  - pandas

Instalação das dependências:
```
pip install numpy matplotlib pandas
```

### Compilador C (para execução da main)

- GCC (Linux/Mac) ou MinGW (Windows)
- Capacidade de leitura de arquivos de texto

## Instruções de Uso

### Treinamento (Python)

Para gerar as bases, treinar o modelo e salvar os pesos:
```
python gerador_bases.py
```
O script:
- Carrega ou gera as bases de dados.
- Treina um perceptron em cada base por 20 épocas.
- Salva os pesos em `linear_modelo.dat` e `nlinear_modelo.dat`.
- Exibe a acurácia por época no console.

### Execução (C)

1. Compile o código:
```
gcc main.c -o perceptron_exec
```
2. Execute o programa:
```
./perceptron_exec
```
O executável:
- Lê os pesos do modelo desejado (arquivo `.dat`).
- Recebe entradas do usuário ou de arquivo.
- Calcula o produto escalar e aplica a função de ativação para classificação binária.

## Resultados Esperados

- Base Linear: Acurácia de 100% após poucas épocas.
- Base Não-Linear: Acurácia em torno de 40–50%, demonstrando limitação do perceptron de camada única em problemas não linearmente separáveis.

## Observações

- Os arquivos `.dat` contêm os pesos finais do perceptron treinado.
- O código em C pode ser adaptado para ler diferentes formatos de entrada.
- Para problemas não lineares, considere arquiteturas com múltiplas camadas ou funções de ativação não lineares.
=======
# perceptron

Teste de commit
Implementação de Perceptron em C com experimentos visuais
>>>>>>> 14cf1157e9c8f429cb55072762135f2c3c2b3054
