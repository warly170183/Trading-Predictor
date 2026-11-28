# Diario de predicciones

Sesgo aplicado al cono: 0. Las lecturas son contexto y se validan aquí hacia adelante.

## BTC

Predicciones guardadas: **8**. Puntuadas (vencidas): **10**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 6 | 17 % | 83 % |
| 72 h | 4 | 75 % | 100 % |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 6 | 83 % |
| 72 h | 4 | 100 % |

Cono implícito de opciones (SVI + Breeden-Litzenberger, medida neutral al riesgo):

| Horizonte | n | Dentro del 90 % (opciones) |
|---|---|---|
| 24 h | 5 | 100 % |
| 72 h | 3 | 100 % |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 2 | +3.13 % | 100 % |
| Funding muy negativo | 72 h | 2 | +0.02 % | 50 % |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 2 | +3.13 % | 100 % |
| OI con caída fuerte en 1 d | 72 h | 2 | +0.02 % | 50 % |
| Sin lectura extrema | 24 h | 4 | +0.45 % | 50 % |
| Sin lectura extrema | 72 h | 2 | +0.62 % | 50 % |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.

## ETH

Predicciones guardadas: **8**. Puntuadas (vencidas): **10**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 6 | 17 % | 100 % |
| 72 h | 4 | 100 % | 100 % |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 6 | 100 % |
| 72 h | 4 | 100 % |

Cono implícito de opciones (SVI + Breeden-Litzenberger, medida neutral al riesgo):

| Horizonte | n | Dentro del 90 % (opciones) |
|---|---|---|
| 24 h | 5 | 100 % |
| 72 h | 3 | 100 % |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 2 | +2.79 % | 100 % |
| Funding muy negativo | 72 h | 2 | -0.21 % | 0 % |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 0 | — | — |
| OI con caída fuerte en 1 d | 72 h | 0 | — | — |
| Sin lectura extrema | 24 h | 4 | +0.49 % | 50 % |
| Sin lectura extrema | 72 h | 2 | +0.26 % | 50 % |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.

## SOL

Predicciones guardadas: **8**. Puntuadas (vencidas): **10**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 6 | 33 % | 100 % |
| 72 h | 4 | 75 % | 100 % |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 6 | 100 % |
| 72 h | 4 | 100 % |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 0 | — | — |
| Funding muy negativo | 72 h | 0 | — | — |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 1 | -4.31 % | 0 % |
| OI con caída fuerte en 1 d | 72 h | 1 | +4.00 % | 100 % |
| Sin lectura extrema | 24 h | 5 | +2.25 % | 60 % |
| Sin lectura extrema | 72 h | 3 | -1.48 % | 0 % |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.
