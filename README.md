# Trading-Predictor

Un proyecto para una aplicación que incremente las posibilidades de predecir en qué dirección se moverá el precio de una criptomoneda.

Estado actual (BTC-USDT-SWAP, datos públicos de OKX y Deribit):

- **Cono de probabilidad**: GJR-GARCH(1,1) con colas t y simulación Monte Carlo. Está calibrado: la banda del 90 % a 24 h y 72 h contiene ~90 % de los resultados en walk-forward. Predice la anchura del cono, no el sentido.
- **Capa de operaciones**: rejilla de stop/objetivo con EV neto de costes, Kelly y acierto necesario.
- **Sesgo direccional**: probado con precio/volumen (1H) y con funding, DVOL, OI, ratio largos/cortos y régimen HMM (diario). **Ninguna señal supera la validación**, así que la deriva del cono es 0.
- **GEX de opciones**: instantánea de Deribit, sin historia para validarla. Solo contexto.
- **Ensamble de volatilidad** (GARCH+HAR, `src/volmodels.py`): mejora pequeña pero consistente de la puntuación del cono en BTC, ETH y SOL a 72 h. Se guarda en paralelo en el diario y aún no sustituye al cono principal. Detalle en [RESULTS.md](RESULTS.md).
- **Diario de predicciones** (`src/journal.py`): cada día guarda el cono y las lecturas, y puntúa lo vencido para validar hacia adelante.

## Uso

```bash
pip install -r requirements.txt
python src/journal.py run               # descarga datos, calcula, guarda la predicción y puntúa (~2-3 min)
python src/journal.py run --no-refresh  # reutiliza data/
python src/journal.py score             # solo puntúa y regenera journal/REPORT.md
```

Los datos se guardan en `data/` (no versionado; cambia la ruta con `TP_DATA`). El diario va en `journal/` y sí se versiona.
`.github/workflows/daily-journal.yml` ejecuta el diario cada día a las 00:20 UTC y sube el resultado al repo. Si OKX o Deribit bloquean las IP de GitHub Actions, el workflow fallará y habrá que ejecutarlo en otra máquina.

## Código (`src/`)

| Archivo | Qué hace |
|---|---|
| `fetch.py`, `fetch_derivs.py`, `fetch_funding.py` | Descarga velas (OKX) y derivados (OKX rubik, Deribit funding, DVOL y opciones) |
| `engine.py` | Cono: GARCH, simulación, calibración walk-forward, régimen HMM |
| `trades.py` | Operaciones largo/corto: rejilla stop/objetivo, EV, Kelly |
| `signals.py` | Validación direccional con precio y volumen (`btc`, `eth` o `sol`) |
| `bias.py` | Validación del sesgo con funding, DVOL, OI, largos/cortos y HMM |
| `vol_ensemble.py`, `volmodels.py`, `fetch_dvol.py` | Validación del ensamble de volatilidad (GARCH, HAR, DVOL) y modelo reutilizable |
| `gex.py` | GEX actual desde la cadena de opciones de Deribit |
| `journal.py` | Diario de predicciones y puntuación |
| `export*.py` | Genera los datasets del dashboard publicado en Claude |

Orden manual completo: `fetch.py` → `fetch_derivs.py` → `engine.py` → `signals.py` → `trades.py` → `bias.py` → `gex.py` → `export*.py`. Los argumentos de `fetch.py` son `instrumento barra n salida`, como en `journal.py`.

## Límites

- Los resultados de validación son de una muestra de ~3 años de BTC; no garantizan nada hacia adelante.
- El GEX supone que los dealers están largos en calls y cortos en puts.
- No es una recomendación de inversión.
