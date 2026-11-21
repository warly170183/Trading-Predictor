# Resultados de validación

Datos: velas 1H de OKX (BTC, ETH y SOL, ~3,4 años cada uno) y DVOL de Deribit (BTC y ETH, desde dic-2023). Todo en walk-forward, sin mirar al futuro.

## 1. Volatilidad: ensamble e implícita (`src/vol_ensemble.py`)

Mejora de la puntuación del cono (pinball loss, cuantiles 5-95 %) frente al GJR-GARCH. Entre corchetes, intervalo de confianza al 95 % por bootstrap en bloques. Positivo es mejor.

| Activo | Horizonte | HAR | Ensamble GARCH+HAR | Con implícita (DVOL) |
|---|---|---|---|---|
| BTC | 24 h | +0,05 % | +0,38 % [0,0; +0,8] | +0,39 % [-0,2; +1,0] |
| BTC | 72 h | +0,20 % | **+0,78 % [+0,2; +1,4]** | +0,58 % [-0,3; +1,3] |
| ETH | 24 h | +0,66 % | **+0,76 % [+0,4; +1,2]** | **+0,90 % [+0,3; +1,5]** |
| ETH | 72 h | +1,33 % | **+1,09 % [+0,2; +2,1]** | **+1,45 % [+0,2; +2,7]** |
| SOL | 24 h | -0,04 % | +0,40 % [0,0; +0,8] | sin DVOL |
| SOL | 72 h | **+1,84 %** | **+1,69 % [+0,8; +2,7]** | sin DVOL |

Lectura:
- El ensamble GARCH+HAR supera al GARCH en los tres activos a 72 h, con intervalo por encima de cero. La mejora es pequeña (0,4 a 1,7 %).
- La volatilidad implícita sola no supera al GARCH. En ETH, sumarla al ensamble da una mejora algo mayor; en BTC no añade nada. No está probado que merezca la complejidad.
- La cobertura del 50 % y del 90 % es correcta en todos los modelos (50/90 ± 2 puntos). El cono ya estaba bien calibrado, y la mejora es de nitidez.
- El cono con ensamble se guarda en paralelo en el diario (`cone_ens`). Hoy sale bastante más estrecho que el de GARCH porque el GARCH tiene persistencia ~1 y arrastra la volatilidad actual a todo el horizonte. Hay que comprobar hacia adelante que su cobertura del 90 % se mantiene antes de usarlo en el dashboard.

## 2. Dirección con precio y volumen en ETH y SOL (`src/signals.py eth|sol`)

Regresión logística L2 en walk-forward, etiquetas sin solape, coste 0,10 % por operación.

| Activo | Horizonte | Acierto | AUC | Neto por operación [IC 95 %] |
|---|---|---|---|---|
| ETH | 6 h | 51,2 % | 0,523 | -0,106 % [-0,162; -0,048] |
| ETH | 24 h | 52,7 % | 0,512 | +0,082 % [-0,138; +0,316] |
| ETH | 72 h | 49,8 % | 0,512 | +0,075 % [-0,640; +0,774] |
| SOL | 6 h | 51,1 % | 0,522 | -0,059 % [-0,121; +0,004] |
| SOL | 24 h | 49,5 % | 0,493 | -0,009 % [-0,269; +0,250] |
| SOL | 72 h | 50,8 % | 0,505 | -0,023 % [-0,706; +0,697] |

Ninguna señal sobrevive a la corrección de Bonferroni en ningún activo, y ningún intervalo del neto queda por encima de cero. Con BTC, ETH y SOL el resultado es el mismo: no hay ventaja direccional con precio y volumen. A 72 h hay pocas muestras (333) y las probabilidades salen mal calibradas (Brier peor que la tasa base).

## 3. Intradía con 2 años de histórico de Binance (`src/binance_vision.py`, `src/hist_study.py`)

1,08 M de velas de 1 minuto por activo (nov 2024 a nov 2026, futuros USDT-M), con flujo agresor, base perp-índice, OI y ratios long/short. Mismo protocolo: Spearman sin solape, Bonferroni (52 pruebas), walk-forward logístico, coste 0,10 % ida y vuelta (taker). Sin libro de órdenes ni liquidaciones (no están en el histórico público).

| Activo | Horizonte | Acierto | AUC | Bruto por operación | Neto (coste 0,10 %) |
|---|---|---|---|---|---|
| BTC | 5 min | 53,3 % | 0,551 | +0,005 % | -0,095 % |
| BTC | 15 min | 52,5 % | 0,539 | +0,004 % | -0,096 % |
| BTC | 60 min | 52,0 % | 0,526 | -0,002 % | -0,102 % |
| ETH | 5 min | 61,2 % | 0,672 | +0,040 % | -0,060 % |
| ETH | 15 min | 57,6 % | 0,614 | +0,044 % | -0,056 % |
| ETH | 60 min | 54,1 % | 0,561 | +0,035 % | -0,066 % |

Lectura:
- Aquí sí hay estructura estadística real (18 señales superan Bonferroni en BTC, 17 en ETH): reversión a 5 y 15 minutos tras movimiento y flujo agresor fuertes. Es la primera señal genuina del proyecto.
- No es operable: el bruto medio es de 0,5 pb (BTC) a 4 pb (ETH) por operación y el coste taker es de 10 pb. Haría falta un coste total por debajo de ~4 pb en ETH, y ejecutar como maker introduce selección adversa que este estudio no mide.
- Los precios son cierres de velas de 1 minuto; parte de la reversión puede ser rebote bid-ask. No se ha verificado con trades.

## 4. Evaluación formal (`src/evaluation.py`, integrada en `vol_ensemble.py`)

CRPS desde cuantiles, Diebold-Mariano (HAC + Harvey-Leybourne-Newbold), Kupiec, Christoffersen y PSR/DSR, con pruebas sintéticas (`tests/test_evaluation.py`).

- DM, ensamble GARCH+HAR frente a GARCH a 72 h: p = 0,006 (BTC), 0,004 (ETH), 0,000 (SOL). A 24 h: p = 0,08 / 0,000 / 0,054. Las mejoras del apartado 1 son significativas y pequeñas (0,4 a 1,7 %).
- Kupiec (cobertura 90 %): no se rechaza en ningún modelo salvo GARCH en ETH a 72 h (p = 0,03; 0,01 con IV), que cubre de más. El ensamble lo corrige (p = 0,24).
- Christoffersen (independencia) se rechaza a 72 h en todos los modelos y activos. Es en gran parte un artefacto: los orígenes diarios solapan a 72 h y los fallos se agrupan por construcción. A 24 h (sin solape) pasa en todos.

## 5. Conformal adaptativo (`src/conformal.py`, `src/conformal_eval.py`)

ACI con retroalimentación retrasada sobre el intervalo al 90 % (|retorno|/sigma). Cobertura 89,8 a 90,2 % con ACI frente a 90,1 a 91,4 % con calibración estática; el ancho es similar y el Winkler sale algo peor con ACI. No aporta mejora: el cono ya estaba bien calibrado. Se deja como red de seguridad ante cambios de régimen, no como sustituto.

## 6. Cono implícito de opciones (`src/options_cone.py`)

SVI por vencimiento (error de ajuste típico 0,1 a 3 puntos de volatilidad), densidad de Breeden-Litzenberger e interpolación en varianza total. Es la medida neutral al riesgo y no tiene histórico en la API, así que no se puede validar hacia atrás. Se guarda como `cone_opt` en el diario para BTC y ETH y se mide su cobertura (`in90_opt`) hacia adelante. Hoy (BTC, IV ATM 59 % a 24 h): 5 %-95 % de -5,5 % a +4,8 %.

## 7. Rotación entre 40 monedas (`src/rotation.py`)

Protocolo preregistrado en el docstring: momentum 30 d (sin el último día), 5 mejores, rebalanceo semanal, exceso sobre pesos iguales, coste 0,10 % ida y vuelta. 45 variantes cuentan como ensayos. Datos: 40 swaps de OKX, nov 2023 a nov 2026.

| Resultado | Valor |
|---|---|
| Primaria: Sharpe del exceso | 0,60 (mitades 0,26 y 0,86) |
| t Newey-West | 1,01 (IC 95 % de la media diaria [-5,4; +16,2] pb) |
| Sharpe deflactado (45 ensayos) | 0,42 |
| Mejor variante (L14, k=5, 14 d) | Sharpe 1,48, DSR 0,91 |
| Variantes con Sharpe > 0 | 100 % |

No pasa el criterio (DSR > 0,95, t > 3). Que todas las variantes salgan positivas apunta a sesgo, no a ventaja: el universo son las 40 monedas con más volumen hoy, es decir, supervivientes que subieron. Es una cota optimista. Para confirmarlo hace falta un universo histórico con las monedas deslistadas.

## Qué falta
- Validar hacia adelante `cone_ens` y `cone_opt` con el diario (cobertura del 90 %).
- Comprobar la reversión de ETH a 5 a 15 minutos con trades reales (rebote bid-ask) y con costes maker antes de darle más peso.
- Universo histórico sin sesgo de supervivencia para la rotación.
