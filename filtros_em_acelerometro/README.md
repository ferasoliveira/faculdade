Projeto dedicado à disciplina de Processamento Digital de Dados

O projeto tem a finalidade de entender e experimentar a aplicação de filtros em sinais gerados, ou até mesmo em tempo real para um acelerômetro GY-521.

Para executar os códigos em python, devem ser executados os seguintes comandos para instalar as bibliotecas:

```
pip install pandas
pip install matplotlib
pip install numpy
pip install scipy
pip install filterpy
pip install pyserial
```

Os dados em .csv foram gerados a partir do arduino UNO, com o código especificado na pasta GY521, e lidos pelo terminal serial a partir de Python, no código MPU6050_Leitura_em_Y.py

Com os dados brutos salvos, foram executados alguns tratamentos, incluindo os filtros de Kalman e Passa-Baixa, para obter sinais de aceleração em um eixo com menos ruídos.

A partir dai, foi efetuado uma tentativa de determinar a distância percorrida pelo acelerômetro (Que deveria ter sido 1 metro). 

Obs: Cada código refere-se a apenas um conjunto de dados .csv, mas são facilmente alterados, alterando o nome do documento csv lido para plotar diferentes bases de dados com o mesmo algoritmo.
