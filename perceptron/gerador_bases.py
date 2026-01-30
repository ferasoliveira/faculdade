# gerador_bases.py
import numpy as np
import matplotlib.pyplot as plt
import os

def gerar_base_linear(n_amostras=100, n_dimensoes=2):
    X = np.random.randn(n_amostras, n_dimensoes)
    w = np.random.randn(n_dimensoes)
    y = (X @ w > 0).astype(int)  # classe 1 se produto escalar positivo
    return X, y

def gerar_base_nlinear(n_amostras=100, n_dimensoes=2):
    if n_dimensoes == 2:
        # Gera padrão tipo XOR em 2D
        X = np.random.randn(n_amostras, 2)
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    else:
        # Em N dimensões, mistura linear + ruído forte
        X = np.random.randn(n_amostras, n_dimensoes)
        w = np.random.randn(n_dimensoes)
        y = (X @ w > 0).astype(int)
        y = np.logical_xor(y, np.random.rand(n_amostras) > 0.8).astype(int)
    return X, y

def salvar_base(X, y, nome_base):
    np.savetxt(f"{nome_base}_X.csv", X, delimiter=',')
    np.savetxt(f"{nome_base}_y.csv", y, delimiter=',')

def plotar_2d(X, y, nome_base):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Classe 0', alpha=0.6)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Classe 1', alpha=0.6)
    plt.title(f"Base: {nome_base}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{nome_base}.png")
    plt.close()

def gerar_bases(n_amostras=100, n_dimensoes=2):
    os.makedirs("bases", exist_ok=True)
    # Linearmente separável
    X_lin, y_lin = gerar_base_linear(n_amostras, n_dimensoes)
    salvar_base(X_lin, y_lin, "bases/base_linear")
    if n_dimensoes == 2:
        plotar_2d(X_lin, y_lin, "bases/base_linear")

    # Não linearmente separável
    X_nlin, y_nlin = gerar_base_nlinear(n_amostras, n_dimensoes)
    salvar_base(X_nlin, y_nlin, "bases/base_nlinear")
    if n_dimensoes == 2:
        plotar_2d(X_nlin, y_nlin, "bases/base_nlinear")

if __name__ == "__main__":
    N = int(input("Digite o número de dimensões (ex: 2): "))
    gerar_bases(n_amostras=200, n_dimensoes=N)
    print("Bases geradas e salvas na pasta 'bases/'")
