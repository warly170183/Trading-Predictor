"""Rutas del proyecto. Los datos van a data/ salvo que TP_DATA diga otra cosa."""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.environ.get("TP_DATA", os.path.join(ROOT, "data"))
JOURNAL = os.path.join(ROOT, "journal")
os.makedirs(DATA, exist_ok=True)
