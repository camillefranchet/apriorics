from math import ceil
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from nptyping import NDArray


def split_data_k_fold(
    data: Sequence[Any],
    k: int = 5,
    test_size: Union[int, float] = 0.1,
    seed: Optional[int] = None,
    previous_splits: Optional[Dict[str, NDArray[Any]]] = None,
) -> Dict[str, NDArray[Any]]:
    r"""
    Split input data sequence into k folds and a test fold.

    Args:
        data: input data sequence.
        k: number of folds.
        test_size: if int, number of items to keep for test; if float, portion of data
            to keep for test.
        seed: seed to use for RNG.

    Returns:
        A dictionary that maps fold numbers (or "test") to the corresponding data
        sequence.
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    if isinstance(test_size, float):
        test_size = ceil(test_size * n)

    p = np.ones(k)

    if previous_splits is not None:
        n_per_fold = (n - test_size) / k
        test_size -= len(previous_splits["test"])
        n_to_add = n - test_size - sum([len(v) for v in previous_splits.values()])

        for v in previous_splits.values():
            data = np.delete(data, np.argwhere(np.in1d(data, v)).squeeze(1))

        if n_to_add:
            for i in range(k):
                try:
                    n_fold = max(0, n_per_fold - len(previous_splits[str(i)]))
                except KeyError:
                    n_fold = n_per_fold

                p[i] = n_fold / n_to_add

    p /= p.sum()

    n = len(data)
    if test_size:
        test_idxs = rng.choice(np.arange(n), size=test_size, replace=False)
        test_data = data[test_idxs]
        data = np.delete(data, test_idxs)
    folds = rng.choice(np.arange(k), size=len(data), p=p)

    splits = {}
    for i in range(k):
        splits[str(i)] = data[folds == i]
        if previous_splits is not None and i in previous_splits:
            splits[str(i)] = np.concatenate((previous_splits[str(i)], splits[str(i)]))

    splits["test"] = test_data
    if previous_splits is not None and "test" in previous_splits:
        splits["test"] = np.concatenate((previous_splits["test"], splits["test"]))

    return splits


def split_data_k_fold_batched(
    data: Sequence[Any],
    k: int = 5,
    test_size: Union[int, float] = 0.1,
    seed: Optional[int] = None,
    previous_splits: Optional[Dict[str, NDArray[Any]]] = None,
    batch_size: int = 10,
) -> Dict[str, NDArray[Any]]:
    r"""
    Split input data sequence into k folds and a test fold using small batches to ensure
    uniform distribution.

    Args:
        data: input data sequence.
        k: number of folds.
        test_size: if int, number of items to keep for test; if float, portion of data
            to keep for test.
        seed: seed to use for RNG.

    Returns:
        A dictionary that maps fold numbers (or "test") to the corresponding data
        sequence.
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    if isinstance(test_size, float):
        test_size = ceil(test_size * n)

    p = np.ones(k)

    if previous_splits is not None:
        n_per_fold = (n - test_size) / k
        test_size -= len(previous_splits["test"])
        n_to_add = n - test_size - sum([len(v) for v in previous_splits.values()])

        for v in previous_splits.values():
            data = np.delete(data, np.argwhere(np.in1d(data, v)).squeeze(1))

        if n_to_add:
            for i in range(k):
                try:
                    n_fold = max(0, n_per_fold - len(previous_splits[str(i)]))
                except KeyError:
                    n_fold = n_per_fold

                p[i] = n_fold / n_to_add

    p /= p.sum()

    n = len(data)
    splits = {str(i): [] for i in range(k)}
    splits["test"] = []
    if previous_splits is not None:
        for i in splits:
            if i in previous_splits:
                splits[i].append(previous_splits[i])

    for b in range(0, n, batch_size):
        batch_data = data[b : b + batch_size].copy()
        n_test = len(batch_data) * test_size / n
        p_test = n_test - int(n_test)
        n_test_batch = int(n_test) + 1 if rng.random() < p_test else int(n_test)
        test_idxs = rng.choice(
            np.arange(len(batch_data)), size=n_test_batch, replace=False
        )
        test_data = batch_data[test_idxs]
        batch_data = np.delete(batch_data, test_idxs)
        folds = rng.choice(np.arange(k), size=len(batch_data), p=p)

        for i in range(k):
            fold_data = batch_data[folds == i]
            if len(fold_data):
                splits[str(i)].append(fold_data)
        if len(test_data):
            splits["test"].append(test_data)

    for i in splits:
        splits[i] = np.sort(np.concatenate(splits[i]))

    return splits
