# Diario de predicciones

Sesgo aplicado al cono: 0. Las lecturas son contexto y se validan aquí hacia adelante.

## BTC

Predicciones guardadas: **4**. Puntuadas (vencidas): **3**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 3 | 33 % | 67 % |
| 72 h | 0 | — | — |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 3 | 67 % |
| 72 h | 0 | — |

Cono implícito de opciones (SVI + Breeden-Litzenberger, medida neutral al riesgo):

| Horizonte | n | Dentro del 90 % (opciones) |
|---|---|---|
| 24 h | 2 | 100 % |
| 72 h | 0 | — |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 2 | +3.13 % | 100 % |
| Funding muy negativo | 72 h | 0 | — | — |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 2 | +3.13 % | 100 % |
| OI con caída fuerte en 1 d | 72 h | 0 | — | — |
| Sin lectura extrema | 24 h | 1 | -1.61 % | 0 % |
| Sin lectura extrema | 72 h | 0 | — | — |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.

## ETH

Predicciones guardadas: **4**. Puntuadas (vencidas): **3**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 3 | 33 % | 100 % |
| 72 h | 0 | — | — |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 3 | 100 % |
| 72 h | 0 | — |

Cono implícito de opciones (SVI + Breeden-Litzenberger, medida neutral al riesgo):

| Horizonte | n | Dentro del 90 % (opciones) |
|---|---|---|
| 24 h | 2 | 100 % |
| 72 h | 0 | — |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 2 | +2.79 % | 100 % |
| Funding muy negativo | 72 h | 0 | — | — |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 0 | — | — |
| OI con caída fuerte en 1 d | 72 h | 0 | — | — |
| Sin lectura extrema | 24 h | 1 | -1.40 % | 0 % |
| Sin lectura extrema | 72 h | 0 | — | — |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.

## SOL

Predicciones guardadas: **4**. Puntuadas (vencidas): **3**.

### Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 3 | 67 % | 100 % |
| 72 h | 0 | — | — |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 3 | 100 % |
| 72 h | 0 | — |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

### Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 0 | — | — |
| Funding muy negativo | 72 h | 0 | — | — |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 0 | — | — |
| OI con caída fuerte en 1 d | 72 h | 0 | — | — |
| Sin lectura extrema | 24 h | 3 | +1.11 % | 33 % |
| Sin lectura extrema | 72 h | 0 | — | — |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.
