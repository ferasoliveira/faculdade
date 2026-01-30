# simulacao_de_cames
Código direcionado para visualização e comparação entre os diagramas EVAP (deslocamento, velocidade, aceleração) e ângulo de pressão dos seguintes perfis de cames polinomiais:
Perfil de Came 2-3;
Perfil de Came 3-4-5;
Perfil de Came 4-5-6-7;

O estudo foi adaptado para um projeto com as seguintes especificações:
- Raio de base: 16 mm
- Altura máxima de deslocamento da válvula: 8 mm
- Subida: 8 mm em 60°
- Descida: 8 mm em 60°
- Espera inferior: no deslocamento 0 por 240°
- Velocidade de rotação do motor: 2370 rpm
- Diâmetro do furo do eixo: 5 mm
- Largura do came: 10 mm

Caso seu projeto tenha especificações diferentes, basta alterar os parametros no inicio do código

Para execução do código, são necessárias as bibliotecas de python: numpy, matplotlib e math (padrão do python)

Para instalar as outras duas bibliotecas, basta executar o seguinte comando em seu terminal:

```
pip install numpy matplotlib
```

Após isso, basta executar o programa.

As imagens anexadas ao projeto, são os resultados obtidos.
