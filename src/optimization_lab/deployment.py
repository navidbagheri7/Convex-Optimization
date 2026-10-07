"""Portable factor inference with strict numeric checkpoint validation."""

from pathlib import Path

import numpy as np


def load_factors(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        required = ["mean", "user", "item", "user_bias", "item_bias"]
        state = {key: np.array(archive[key], dtype=np.float64, copy=True) for key in required}
    if any(not np.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite checkpoint")
    if state["mean"].shape != () or state["user"].ndim != 2 or state["item"].ndim != 2:
        raise ValueError("Invalid mean/factor shapes")
    if (
        state["user"].shape[1] != state["item"].shape[1]
        or state["user_bias"].shape != (len(state["user"]),)
        or state["item_bias"].shape != (len(state["item"]),)
    ):
        raise ValueError("Inconsistent factor/bias shapes")
    for value in state.values():
        value.setflags(write=False)
    return state


def predict_ratings(path, users, items, clip=True):
    state = load_factors(path)
    users, items = np.asarray(users), np.asarray(items)
    if (
        users.shape != items.shape
        or not np.issubdtype(users.dtype, np.integer)
        or not np.issubdtype(items.dtype, np.integer)
    ):
        raise ValueError("Matching integer IDs required")
    shape = users.shape
    users, items = users.ravel(), items.ravel()
    values = np.full(len(users), float(state["mean"]))
    known_user = (users >= 0) & (users < len(state["user"]))
    known_item = (items >= 0) & (items < len(state["item"]))
    values[known_user] += state["user_bias"][users[known_user]]
    values[known_item] += state["item_bias"][items[known_item]]
    both = known_user & known_item
    values[both] += np.sum(state["user"][users[both]] * state["item"][items[both]], axis=1)
    values = values.reshape(shape)
    return np.clip(values, 1.0, 5.0) if clip else values
