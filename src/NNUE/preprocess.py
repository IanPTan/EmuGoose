import numpy as np
import h5py as hp
from tqdm import tqdm
import pandas as pd
from utils.data import encode_fen


fen_dataset_path = "NNUE/data/fen_dataset.csv"
output_dataset_path = "NNUE/data/dataset.h5"


dataset_df = pd.read_csv(fen_dataset_path)
dataset_size = len(dataset_df)
print(f"Preprocessing {dataset_size} boards...")

with hp.File(output_dataset_path, "w") as f:
    board_dataset = f.create_dataset("board", (dataset_size, 768), dtype=bool)
    eval_dataset = f.create_dataset("eval", (dataset_size,), dtype=np.int8)

    for i in tqdm(range(dataset_size), unit="boards", desc="Processing..."):
        row = dataset_df.iloc[i]
        board_representation = encode_fen(row["FEN"])
        evaluation = row["WDL"] * 2

        board_dataset[i] = board_representation.flatten()
        eval_dataset[i] = evaluation
