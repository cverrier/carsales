from pathlib import Path

import pandas as pd


def load_autos(path=Path("..", "data", "raw", "autos.csv")):
    autos = pd.read_csv(path, encoding="Latin-1")

    return autos
