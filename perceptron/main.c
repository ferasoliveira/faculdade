// main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_AMOSTRAS   1000
#define MAX_DIMENSOES  10

// Carrega X de um CSV (n_amostras x n_dimensoes)
// Retorna número de linhas efetivamente lidas
int carregar_X_csv(const char* nome_arquivo, float X[][MAX_DIMENSOES], int n_dimensoes) {
    FILE* f = fopen(nome_arquivo, "r");
    if (!f) { perror(nome_arquivo); exit(EXIT_FAILURE); }
    char linha[1024];
    int i = 0;
    while (i < MAX_AMOSTRAS && fgets(linha, sizeof(linha), f)) {
        if (linha[0]=='\n' || linha[0]=='\0') continue;
        char* token = strtok(linha, ",");
        for (int d = 0; d < n_dimensoes; d++) {
            if (!token) break;
            X[i][d] = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(f);
    return i;
}

// --- Função de carregamento de y corrigida ---
int carregar_y_csv(const char* nome_arquivo, int Y[], int max_linhas) {
    FILE* f = fopen(nome_arquivo, "r");
    if (!f) { perror(nome_arquivo); exit(EXIT_FAILURE); }
    char linha[256];
    int i = 0;
    while (i < max_linhas && fgets(linha, sizeof(linha), f)) {
        char *p = linha;
        while (*p==' '||*p=='\t'||*p=='\r'||*p=='\n') p++;
        if (*p == '\0') continue;
        Y[i++] = atoi(p);
    }
    fclose(f);
    return i;
}

// Salva modelo (pesos + bias) em arquivo binário
void salvar_modelo(const char* nome_arquivo, float pesos[], int n_dim, float bias) {
    FILE* f = fopen(nome_arquivo, "wb");
    if (!f) { perror(nome_arquivo); exit(EXIT_FAILURE); }
    fwrite(pesos, sizeof(float), n_dim, f);
    fwrite(&bias, sizeof(float), 1, f);
    fclose(f);
}

// Carrega modelo (pesos + bias) de arquivo binário
void carregar_modelo(const char* nome_arquivo, float pesos[], int n_dim, float* bias) {
    FILE* f = fopen(nome_arquivo, "rb");
    if (!f) { perror(nome_arquivo); exit(EXIT_FAILURE); }
    fread(pesos, sizeof(float), n_dim, f);
    fread(bias, sizeof(float), 1, f);
    fclose(f);
}

// Separa 80% / 20%
void separar_treino_teste(int n_amostras, int idx_split,
                          float X[][MAX_DIMENSOES], int Y[],
                          float X_tre[][MAX_DIMENSOES], int Y_tre[],
                          float X_tes[][MAX_DIMENSOES], int Y_tes[],
                          int* n_tre, int* n_tes) {
    *n_tre = idx_split;
    *n_tes = n_amostras - idx_split;
    for (int i = 0; i < *n_tre; i++) {
        memcpy(X_tre[i], X[i], sizeof(float)*MAX_DIMENSOES);
        Y_tre[i] = Y[i];
    }
    for (int i = 0; i < *n_tes; i++) {
        memcpy(X_tes[i], X[i + idx_split], sizeof(float)*MAX_DIMENSOES);
        Y_tes[i] = Y[i + idx_split];
    }
}

// Treina perceptron
void treinar_perceptron(float X[][MAX_DIMENSOES], int Y[], int n_amostras, int n_dim,
                        float pesos[], float* bias, float lr, int epocas) {
    for (int e = 0; e < epocas; e++) {
        int acertos = 0;
        for (int i = 0; i < n_amostras; i++) {
            float soma = *bias;
            for (int d = 0; d < n_dim; d++)
                soma += X[i][d] * pesos[d];
            int pred = (soma >= 0.0f) ? 1 : 0;
            int erro = Y[i] - pred;
            if (erro == 0) acertos++;
            for (int d = 0; d < n_dim; d++)
                pesos[d] += lr * erro * X[i][d];
            *bias += lr * erro;
        }
        printf("Época %2d - Acurácia: %5.2f%%\n", e+1, 100.0f*acertos/n_amostras);
    }
}

// Testa perceptron
float testar_perceptron(float X[][MAX_DIMENSOES], int Y[], int n_amostras, int n_dim,
                        float pesos[], float bias) {
    int acertos = 0;
    for (int i = 0; i < n_amostras; i++) {
        float soma = bias;
        for (int d = 0; d < n_dim; d++)
            soma += X[i][d] * pesos[d];
        int pred = (soma >= 0.0f) ? 1 : 0;
        if (pred == Y[i]) acertos++;
    }
    return 100.0f * acertos / n_amostras;
}

int main() {
    int n_dim  = 2;
    float lr   = 0.1f;
    int epocas = 20;

    float X[MAX_AMOSTRAS][MAX_DIMENSOES];
    int   Y[MAX_AMOSTRAS];
    int   nX, nY;

    for (int tipo = 0; tipo < 2; tipo++) {
        // Define caminhos
        char path_X[64], path_y[64], model_file[64];
        sprintf(path_X, "bases/base_%s_X.csv", tipo==0?"linear":"nlinear");
        sprintf(path_y, "bases/base_%s_y.csv", tipo==0?"linear":"nlinear");
        sprintf(model_file, "%s_modelo.dat", tipo==0?"linear":"nlinear");

        // 1) Carrega X e Y
        nX = carregar_X_csv(path_X, X, n_dim);
        nY = carregar_y_csv(path_y, Y, nX);

        // Garante usar apenas o mínimo comum
        int n_amostras = (nX < nY ? nX : nY);
        if (nX != nY) {
            fprintf(stderr,
                "Aviso: X lido=%d, y lido=%d; usando %d amostras.\n",
                nX, nY, n_amostras);
        }

        // 2) Separa treino/teste
        int idx_split = (int)(0.8f * n_amostras);
        float X_tre[MAX_AMOSTRAS][MAX_DIMENSOES], X_tes[MAX_AMOSTRAS][MAX_DIMENSOES];
        int   Y_tre[MAX_AMOSTRAS], Y_tes[MAX_AMOSTRAS];
        int n_tre, n_tes;
        separar_treino_teste(n_amostras, idx_split, X, Y,
                             X_tre, Y_tre, X_tes, Y_tes, &n_tre, &n_tes);

        // 3) Treina
        float pesos[MAX_DIMENSOES] = {0}, bias = 0;
        printf("\n--- Treinando base %s ---\n", tipo==0?"Linear":"Não-Linear");
        treinar_perceptron(X_tre, Y_tre, n_tre, n_dim, pesos, &bias, lr, epocas);

        // 4) Salva modelo
        salvar_modelo(model_file, pesos, n_dim, bias);
        printf("Modelo salvo em '%s'\n", model_file);

        // 5) Testa
        float pesos2[MAX_DIMENSOES], bias2;
        carregar_modelo(model_file, pesos2, n_dim, &bias2);
        float acc = testar_perceptron(X_tes, Y_tes, n_tes, n_dim, pesos2, bias2);
        printf("Acurácia teste (%s): %5.2f%%\n",
               tipo==0?"Linear":"Não-Linear", acc);
    }

    return 0;
}
