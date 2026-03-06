from typing import Iterable
import numpy as np

def cosine_sym(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    This is on the fastest cosine sym function on the internal hardware.
    """
    if vector_a is None or vector_b is None:
        return -1

    answer = -1.
    if np.round(sum(vector_a) * sum(vector_b), 2) != 0.00:
        answer = float(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
    return answer


def compare(records: Iterable[tuple[str,list[str]]], model) -> Iterable[float]:
    # encoding a sentence can be expensive, so ensuring every sentence is only computed once can help with performance
    # we ensure to have a generator of tuples[np.ndarray, tuple[np.ndarray]] with the enclosed tuples representing the negatives
    vectors = ((model.encode(original), tuple(map(model.encode, negatives)))
               for original, negatives in records)
    # we now compute the cosine sim between the original and all negatives
    return (cosine_sym(original_vector, negative_vector)
            for original_vector, negative_vectors in vectors
            for negative_vector in negative_vectors)
