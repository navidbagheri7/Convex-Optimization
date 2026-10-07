"""Sequential validation-only Optuna search with complete candidate evidence.

Reuse the audited degree-regularized MovieLens factor objective and portable
inference. Search state lives in private SQLite; public trial records and numeric
curves are enough to independently check selection and held-out metrics.
"""

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import optuna

from .artifacts import Run, sha256, write_json
from .recommendation import fit_stochastic, load_ratings, test_statistics


def fit(
    config, seed, strategy, device="cuda:0", output="results/reference/adaptive_search", quick=False
):
    """Finish every configured candidate; never access test outcomes during search."""
    if strategy not in config["strategies"] or strategy not in ["random", "tpe"]:
        raise ValueError("A declared random or TPE search strategy is required")
    data = load_ratings(config["data_root"], config["validation_fraction"], config["split_seed"])
    directory = Path(output) / strategy / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    sources = [__file__, Path(__file__).with_name("recommendation.py")]
    protocol = config | dict(
        seed=seed,
        strategy=strategy,
        device=device,
        quick=quick,
        source_data_sha256=data["source_sha256"],
    )
    if (directory / "run.json").exists():
        prior = json.loads((directory / "run.json").read_text())
        assert prior["config"] == protocol
        assert prior["source_sha256"] == {Path(p).name: sha256(p) for p in sources}
        if prior["status"] == "complete":
            print("Preserved complete search", strategy, seed, flush=True)
            return
    database = directory / "checkpoints/search.sqlite"
    if database.exists():
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            previous_trials = connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        if previous_trials:
            raise ValueError(
                "Partial adaptive search requires a fresh output; sampler RNG continuation is not claimed. Existing records are preserved."
            )
    run = Run(directory, protocol, sources)
    run.record["versions"]["optuna"] = optuna.__version__
    write_json(directory / "run.json", run.record)
    checkpoint = directory / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    numeric = (
        Path(output) / "source_data" / strategy
        if quick
        else Path("assets/source_data/adaptive_search") / strategy
    )
    numeric.mkdir(parents=True, exist_ok=True)
    sample_seed = config["search_seed"] + seed
    sampler = (
        optuna.samplers.RandomSampler(seed=sample_seed)
        if strategy == "random"
        else optuna.samplers.TPESampler(seed=sample_seed, n_startup_trials=config["startup_trials"])
    )
    storage = f"sqlite:///{(checkpoint / 'search.sqlite').resolve()}"
    search = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=f"{strategy}_{seed}",
        load_if_exists=True,
    )
    if search.trials:
        raise ValueError(
            "Partially completed adaptive search requires a fresh output. Sampler continuation is not claimed to be reproducible by recreating its RNG."
        )

    def objective(trial):
        rank = trial.suggest_categorical("rank", config["rank_choices"])
        penalty = trial.suggest_float("penalty", *config["penalty_bounds"], log=True)
        bias = trial.suggest_float("bias_penalty", *config["bias_penalty_bounds"], log=True)
        learning_rate = trial.suggest_float(
            "learning_rate", *config["learning_rate_bounds"], log=True
        )
        started = time.perf_counter()
        state, history, epoch = fit_stochastic(
            data,
            rank,
            seed,
            penalty,
            bias,
            config["epochs"],
            "adam",
            learning_rate,
            config["batch_size"],
            device,
        )
        path = checkpoint / f"trial{trial.number}.npz"
        np.savez_compressed(path, **state)
        curve = numeric / f"seed{seed}_trial{trial.number}.npz"
        np.savez_compressed(curve, history=history)
        trial.set_user_attr("selected_epoch", epoch)
        trial.set_user_attr("elapsed_seconds", time.perf_counter() - started)
        trial.set_user_attr("checkpoint_sha256", sha256(path))
        trial.set_user_attr("curve_sha256", sha256(curve))
        trial.set_user_attr(
            "generator_updates",
            config["epochs"] * int(np.ceil(len(data["fit"]) / config["batch_size"])),
        )
        print(
            "Finished trial", strategy, seed, trial.number, float(history[:, 2].min()), flush=True
        )
        return float(history[:, 2].min())

    search.optimize(objective, n_trials=config["trials"], n_jobs=1)
    assert len(search.trials) == config["trials"] and all(
        t.state == optuna.trial.TrialState.COMPLETE for t in search.trials
    )
    chosen = min(search.trials, key=lambda trial: (trial.value, trial.number))
    with np.load(checkpoint / f"trial{chosen.number}.npz", allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    selected = checkpoint / "selected.npz"
    np.savez_compressed(selected, **state)
    test, arrays = test_statistics(state, data, seed)
    evidence = numeric / f"seed{seed}_test.npz"
    np.savez_compressed(evidence, **arrays)
    trials = [
        dict(number=t.number, parameters=t.params, validation_rmse=t.value, **t.user_attrs)
        for t in search.trials
    ]
    run.finish(
        dict(
            trials=trials,
            selected_trial=chosen.number,
            selected_parameters=chosen.params,
            selected_epoch=chosen.user_attrs["selected_epoch"],
            validation_rmse=chosen.value,
            test=test,
            test_evidence_sha256=sha256(evidence),
            selected_checkpoint_sha256=sha256(selected),
            data_roles=dict(
                fit=len(data["fit"]),
                validation=len(data["validation"]),
                test=len(data["test"]),
                users=data["users"],
                items=data["items"],
            ),
            scope=config["scope"],
            timing_scope="This attempt including all sequential fitting, evidence export and final test; complete prior campaigns skip, partial continuation is rejected.",
        )
    )
