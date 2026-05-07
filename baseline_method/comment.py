from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

ENTROPY = [
    "B-SUB", "I-SUB",
    "B-PRED", "I-PRED",
    "B-OBJ", "I-OBJ",
    "B-ATT", "I-ATT",
    "O"
]

entropy2id = {value: index for index, value in enumerate(ENTROPY)}
id2entropy = {index: value for index, value in enumerate(ENTROPY)}