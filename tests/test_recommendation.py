import numpy as np
import torch

from optimization_lab.recommendation import (
    FactorModel,
    fit_als,
    initialize,
    objective,
    predict,
    rating_loss,
)


def toy():
    rows = np.array([[0, 0, 5], [0, 1, 2], [1, 0, 3], [1, 1, 4]])
    return dict(
        fit=rows,
        validation=rows,
        users=3,
        items=3,
        mean=3.5,
        user_degree=np.array([2, 2, 0]),
        item_degree=np.array([2, 2, 0]),
    )


def test_rating_loss_matches_degree_objective_and_repeated_id_gradient():
    data = toy()
    state = initialize(data, 2, 4)
    model = FactorModel(state).double()
    rows = data["fit"]
    u, i = (torch.tensor(rows[:, j]) for j in [0, 1])
    loss = rating_loss(model, u, i, torch.tensor(rows[:, 2]).double(), 0.1, 0.03)
    exported = model.export()
    np.testing.assert_allclose(loss.item(), objective(exported, data, 0.1, 0.03), rtol=1e-6)
    loss.backward()
    epsilon = 1e-5
    exported["user_bias"][0] += epsilon
    plus = objective(exported, data, 0.1, 0.03)
    exported["user_bias"][0] -= 2 * epsilon
    minus = objective(exported, data, 0.1, 0.03)
    np.testing.assert_allclose(model.user_bias.grad[0], (plus - minus) / (2 * epsilon), atol=1e-7)


def test_als_monotonicity_and_cold_fallback():
    data = toy()
    state, history, _ = fit_als(data, 2, 0, 0.1, 0.03, 12)
    assert np.all(np.diff(history[:, 1]) <= 1e-10)
    assert history[-1, 1] < history[0, 1]
    values = predict(state, np.array([-1, 2, 0]), np.array([-1, 2, -1]))
    np.testing.assert_allclose(values[:2], data["mean"])
    np.testing.assert_allclose(values[2], data["mean"] + state["user_bias"][0])
