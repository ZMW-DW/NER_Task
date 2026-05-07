from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

ENTROPY = [

    # ===== Action =====
    "B-PRED", "I-PRED",

    # ===== Core Arguments =====
    "B-OBJ",  "I-OBJ",
    "B-VAL",  "I-VAL",
    "B-LOC",  "I-LOC",

    # ===== Optional Semantic Roles =====
    "B-ATTR", "I-ATTR",

    # ===== Control / Logic =====
    "B-COND", "I-COND",
    "B-TIME", "I-TIME",

    # ===== Outside =====
    "O"
]

entropy2id = {value: index for index, value in enumerate(ENTROPY)}
id2entropy = {index: value for index, value in enumerate(ENTROPY)}
