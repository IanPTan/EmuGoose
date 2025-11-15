import numpy as np
import h5py as hp
from tqdm import tqdm
import json
from utils.data import encode_fen


fen_dataset_path = "NNUE/data/fen_dataset.json"
output_dataset_path = "NNUE/data/dataset.h5"


def get_dataset_size(file_path):
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def parse_evaluation(evaluation: str) -> float:
    """
    Parses the evaluation string.
    Mate in X is converted to a large number (positive for white, negative for black).
    """
    if "M" in evaluation:
        mate_in = int(evaluation.replace("M", ""))
        # Positive for white mate, negative for black mate
        # We can use a large number to represent mate, assuming normal evals are smaller.
        # A smaller mate_in value is better, so we can use 1000 - abs(mate_in)
        return np.sign(mate_in) * (1000 - abs(mate_in))
    return float(evaluation)


if __name__ == "__main__":
    dataset_size = get_dataset_size(fen_dataset_path)
    print(f"Preprocessing {dataset_size} boards...")

    with hp.File(output_dataset_path, "w") as f, open(fen_dataset_path, "r") as data_file:
        board_dataset = f.create_dataset("board", (dataset_size, 768), dtype=np.bool)
        eval_dataset = f.create_dataset("eval", (dataset_size,), dtype=np.float16)

        for i, line in tqdm(enumerate(data_file), total=dataset_size, unit="boards", desc="Processing..."):
            row = json.loads(line)
            board_representation = encode_fen(row["fen"])
            evaluation = parse_evaluation(row["evaluation"])

            board_dataset[i] = board_representation.flatten()
            eval_dataset[i] = evaluation
