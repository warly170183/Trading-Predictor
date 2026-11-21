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

## Qué falta
- Probar horizontes de 5 a 60 minutos con microestructura (libro de órdenes, flujo agresivo, liquidaciones). Requiere empezar a guardar esos datos.
- Validar hacia adelante el cono con ensamble y las lecturas de funding y OI con el diario.
