# Diario de predicciones

Predicciones guardadas: **1**. Puntuadas (vencidas): **0**.
Sesgo aplicado al cono: 0. Las lecturas son contexto y se validan aquí hacia adelante.

## Cobertura del cono

| Horizonte | n | Dentro del 50 % | Dentro del 90 % |
|---|---|---|---|
| 24 h | 0 | — | — |
| 72 h | 0 | — | — |

Cono con ensamble GARCH+HAR (se guarda en paralelo para compararlo hacia adelante):

| Horizonte | n | Dentro del 90 % (ensamble) |
|---|---|---|
| 24 h | 0 | — |
| 72 h | 0 | — |

Nominal: 50 % y 90 %. Con pocas predicciones la cobertura es ruido; las predicciones diarias consecutivas se solapan a 72 h.

## Retorno posterior según la lectura

| Lectura | Horizonte | n | Retorno medio | % de subidas |
|---|---|---|---|---|
| Funding muy negativo | 24 h | 0 | — | — |
| Funding muy negativo | 72 h | 0 | — | — |
| Funding muy positivo | 24 h | 0 | — | — |
| Funding muy positivo | 72 h | 0 | — | — |
| OI con caída fuerte en 1 d | 24 h | 0 | — | — |
| OI con caída fuerte en 1 d | 72 h | 0 | — | — |
| Sin lectura extrema | 24 h | 0 | — | — |
| Sin lectura extrema | 72 h | 0 | — | — |

Hipótesis a contrastar: tras extremos de funding y OI, el precio tiende a continuar y no a revertir.
No se saca ninguna conclusión con menos de 30 eventos independientes por fila.
