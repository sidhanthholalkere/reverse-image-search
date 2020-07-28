import pickle
from pathlib import Path

def load_file(path):
    path = Path(path)
    with open(path, mode="rb") as opened_file:
        return pickle.load(opened_file)
