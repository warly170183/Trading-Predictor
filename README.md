# Trading-Predictor

Un proyecto para una aplicación que incremente las posibilidades de predecir en qué dirección se moverá el precio de una criptomoneda.

Estado actual (BTC-USDT-SWAP, datos públicos de OKX y Deribit):

- **Cono de probabilidad**: GJR-GARCH(1,1) con colas t y simulación Monte Carlo. Está calibrado: la banda del 90 % a 24 h y 72 h contiene ~90 % de los resultados en walk-forward. Predice la anchura del cono, no el sentido.
- **Capa de operaciones**: rejilla de stop/objetivo con EV neto de costes, Kelly y acierto necesario.
- **Sesgo direccional**: probado con precio/volumen (1H) y con funding, DVOL, OI, ratio largos/cortos y régimen HMM (diario). **Ninguna señal supera la validación**, así que la deriva del cono es 0.
- **GEX de opciones**: instantánea de Deribit, sin historia para validarla. Solo contexto.
- **Ensamble de volatilidad** (GARCH+HAR, `src/volmodels.py`): mejora pequeña pero consistente de la puntuación del cono en BTC, ETH y SOL a 72 h. Se guarda en paralelo en el diario y aún no sustituye al cono principal. Detalle en [RESULTS.md](RESULTS.md).
- **Diario de predicciones** (`src/journal.py`): cada día guarda, para BTC, ETH y SOL, el cono y las lecturas, y puntúa lo vencido para validar hacia adelante. ETH y SOL usan lecturas ligeras (funding, OI, largos/cortos y DVOL solo en ETH), sin HMM ni GEX.

## Uso

```bash
pip install -r requirements.txt
python src/journal.py run                     # BTC, ETH y SOL: descarga, calcula, guarda la predicción y puntúa (~4-5 min)
python src/journal.py run --assets btc        # solo algunos activos
python src/journal.py run --no-refresh        # reutiliza data/
python src/journal.py score                   # solo puntúa y regenera journal/REPORT.md
```

Los datos se guardan en `data/` (no versionado; cambia la ruta con `TP_DATA`). El diario va en `journal/` y sí se versiona.
`.github/workflows/daily-journal.yml` ejecuta el diario cada día a las 00:20 UTC y sube el resultado al repo. Si OKX o Deribit bloquean las IP de GitHub Actions, el workflow fallará y habrá que ejecutarlo en otra máquina.

## Microestructura (datos para el intradía)

`src/micro_collect.py` guarda una fila por minuto de OKX (públicos, sin clave): spread, desequilibrio del libro (5 y 20 niveles), profundidad a 5 pb, microprecio, flujo agresivo comprador y vendedor, liquidaciones largas y cortas, base mark/índice, funding y open interest. No hay historia pública de esto, así que hay que acumularla.

```bash
python src/micro_collect.py                      # sin fin; prueba corta: --duration 120 --interval 20
python src/micro_rollup.py                       # CSV comprimidos en journal/micro/ para versionar
```

Tiene que correr en una máquina siempre encendida: `deploy/micro-collector.service` es un servicio systemd listo para copiar. GitHub Actions no sirve para esto. `journal.py run` llama al rollup si existe `data/micro/`. Con la columna `missed` a 1 se sabe qué minutos perdieron operaciones (lote de 500 trades superado). Hacen falta unos 2 a 3 meses de datos antes de poder contrastar señales de 5 a 60 minutos.

`src/micro_study.py` hace ese contraste: correlación de Spearman de cada señal con el retorno a 5, 15 y 60 minutos (muestras sin solape, Bonferroni) y un walk-forward logístico con coste. Se niega a correr con menos de ~3,5 días de datos (`--force` lo salta). `python tests/test_micro_study.py` comprueba con datos sintéticos que detecta una señal plantada y no inventa ninguna en ruido.

## Código (`src/`)

| Archivo | Qué hace |
|---|---|
| `fetch.py`, `fetch_derivs.py`, `fetch_funding.py` | Descarga velas (OKX) y derivados (OKX rubik, Deribit funding, DVOL y opciones) |
| `engine.py` | Cono: GARCH, simulación, calibración walk-forward, régimen HMM |
| `trades.py` | Operaciones largo/corto: rejilla stop/objetivo, EV, Kelly |
| `signals.py` | Validación direccional con precio y volumen (`btc`, `eth` o `sol`) |
| `bias.py` | Validación del sesgo con funding, DVOL, OI, largos/cortos y HMM |
| `vol_ensemble.py`, `volmodels.py`, `fetch_dvol.py` | Validación del ensamble de volatilidad (GARCH, HAR, DVOL) y modelo reutilizable |
| `micro_collect.py`, `micro_rollup.py`, `micro_study.py` | Captura y estudio de microestructura (libro, flujo, liquidaciones, base) |
| `readings.py` | Lecturas de funding, DVOL, OI y largos/cortos de ETH y SOL para el diario |
| `gex.py` | GEX actual desde la cadena de opciones de Deribit |
| `journal.py` | Diario de predicciones y puntuación |
| `export*.py` | Genera los datasets del dashboard publicado en Claude (`export4.py`: seguimiento del diario y volatilidad) |

Orden manual completo: `fetch.py` → `fetch_derivs.py` → `engine.py` → `signals.py` → `trades.py` → `bias.py` → `gex.py` → `export*.py`. Los argumentos de `fetch.py` son `instrumento barra n salida`, como en `journal.py`.

## Límites

- Los resultados de validación son de una muestra de ~3 años de BTC; no garantizan nada hacia adelante.
- El GEX supone que los dealers están largos en calls y cortos en puts.
- No es una recomendación de inversión.
