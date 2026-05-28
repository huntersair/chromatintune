import numpy as np

DNA_VOCAB = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}

nucleotide_to_index = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}

def one_hot_encode(sequence: str) -> np.ndarray:

    sequence = sequence.upper()

    encoding = [
        DNA_VOCAB.get(nucleotide, [0, 0, 0, 0])
        for nucleotide in sequence
    ]

    return np.array(encoding, dtype=np.float32)

def reverse_complement(sequence: str):

    sequence = sequence.upper()

    complement = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N"
    }

    reversed_sequence = sequence[::-1]

    reverse_complement_sequence = "".join(
        complement.get(base, "N")
        for base in reversed_sequence
    )

    return reverse_complement_sequence

def tokenize_sequence(sequence: str):

    token_map = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4
    }

    sequence = sequence.upper()

    return [
        token_map.get(base, 4)
        for base in sequence
    ]