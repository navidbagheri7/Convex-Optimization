"""Build a local optimization explorer using complete, hash-bound public evidence."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.reporting import records

ROOT = Path(__file__).resolve().parents[1]
IMAGE_RECORDS = []
SOURCES = set()


def metric(family, case=None):
    path = ROOT / "results/reference" / family
    if case is not None:
        path /= case
    record = path / "run.json"
    assert json.loads(record.read_text())["status"] == "complete"
    SOURCES.update([record, path / "metrics.json"])
    return json.loads((path / "metrics.json").read_text())


def arrays(family, case):
    path = ROOT / "assets/source_data" / family / (case + ".npz")
    SOURCES.add(path)
    return path, np.load(path, allow_pickle=False)


def picture(array, name, source, key, index=None):
    arr = np.asarray(array)
    assert np.isfinite(arr).all()
    path = ROOT / "assets/explorer" / (name + ".png")
    pixels = (arr.clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)
    IMAGE_RECORDS.append(
        dict(
            path=str(path.relative_to(ROOT)),
            source=str(source.relative_to(ROOT)),
            key=key,
            index=index,
            sha256=sha256(path),
        )
    )
    return "../" + str(path.relative_to(ROOT))


def main():
    records(strict=True)
    (ROOT / "assets/explorer").mkdir(parents=True, exist_ok=True)
    evidence = dict(
        routing=[],
        tv=[],
        composite=[],
        quadratic=[],
        recommendation=[],
        adaptive=[],
        spectral={},
        applications=[],
        geometry=metric("geometry"),
    )
    for instance in ["berlin52", "ch150"]:
        source, base = arrays("routing", instance)
        main = metric("routing", instance)
        for family in ["routing", "memetic"]:
            path, data = arrays(family, instance)
            metrics = metric(family, instance)
            for row in metrics["rows"]:
                method, seed = row["method"], row["seed"]
                key = f"{method}_seed{seed}"
                evidence["routing"].append(
                    row
                    | dict(
                        instance=instance,
                        family=family,
                        tour=data[key + "_tour"].tolist(),
                        coordinates=base["coordinates"].tolist(),
                        history=data[key + "_history"].tolist() if key + "_history" in data else [],
                        certificate=main["one_tree_certificate"],
                        lower_bound_history=base["one_tree_history"].tolist(),
                        published_optimum=main["published_optimum"],
                    )
                )
            data.close()
        base.close()
    config = json.loads((ROOT / "configs/primal_dual.json").read_text())
    for case in config["datasets"]:
        for seed in config["seeds"]:
            for lam in config["lambdas"]:
                name = f"{case}_seed{seed}_lambda{lam:g}"
                metrics = metric("primal_dual", name)
                source, data = arrays("primal_dual", name)
                truth = picture(data["truth"], f"tv_{case}_truth", source, "truth")
                observed = picture(
                    data["observed"], f"tv_{case}_{seed}_observed", source, "observed"
                )
                for row in metrics["rows"]:
                    method = row["method"]
                    images = []
                    for i, iteration in enumerate(data[method + "_snapshot_iterations"]):
                        images.append(
                            dict(
                                iteration=int(iteration),
                                path=picture(
                                    data[method + "_snapshots"][i],
                                    f"tv_{name}_{method}_{iteration}",
                                    source,
                                    method + "_snapshots",
                                    i,
                                ),
                            )
                        )
                    evidence["tv"].append(
                        row
                        | dict(
                            dataset=case,
                            seed=seed,
                            penalty=lam,
                            truth=truth,
                            observed=observed,
                            images=images,
                            history=data[method + "_history"].tolist(),
                            observation_psnr=metrics["observation_psnr"],
                            pixels=int(data["truth"].size),
                        )
                    )
                data.close()
    for family in ["lasso", "nuclear"]:
        for seed in [0, 1, 2]:
            case = f"{family}_seed{seed}"
            metrics = metric("first_order", case)
            source, data = arrays("first_order", case)
            for row in metrics["rows"]:
                evidence["composite"].append(
                    row
                    | dict(
                        problem=family, seed=seed, history=data[row["method"] + "_history"].tolist()
                    )
                )
            data.close()
    metrics = metric("first_order", "quadratic")
    source, data = arrays("first_order", "quadratic")
    for row in metrics["rows"]:
        evidence["quadratic"].append(
            row | dict(history=data[f"{row['condition']}_{row['method']}_history"].tolist())
        )
    data.close()
    for family in ["recommendation_fixed", "recommendation"]:
        for seed in [0, 1, 2]:
            metrics = metric(family, f"seed{seed}")
            for row in metrics["rows"]:
                candidates = []
                parameter_list = row.get("candidates")
                if parameter_list is None:
                    original_config = json.loads(
                        (ROOT / "results/reference" / family / f"seed{seed}/run.json").read_text()
                    )["config"]
                    rates = original_config["learning_rates"].get(row["method"], [None])
                    parameter_list = [
                        dict(learning_rate=rate, penalty=original_config["lambda"])
                        for rate in rates
                    ]
                    assert len(parameter_list) == len(row["candidate_validation_rmse"])
                for i, parameters in enumerate(parameter_list):
                    source, data = arrays(family, f"{row['method']}_{seed}_candidate{i}")
                    candidates.append(parameters | dict(history=data["history"].tolist()))
                    data.close()
                evidence["recommendation"].append(
                    row | dict(protocol=family, candidates=candidates)
                )
    for strategy in ["random", "tpe"]:
        for seed in [0, 1, 2]:
            metrics = metric("adaptive_search", f"{strategy}/seed{seed}")
            for trial in metrics["trials"]:
                source, data = arrays(
                    "adaptive_search/" + strategy, f"seed{seed}_trial{trial['number']}"
                )
                trial["history"] = data["history"].tolist()
                data.close()
            evidence["adaptive"].append(metrics | dict(strategy=strategy, seed=seed))
    metrics = metric("spectral")
    source, data = arrays("spectral", "svd")
    evidence["spectral"]["compression"] = []
    original = picture(data["original"], "svd_original", source, "original")
    for row in metrics["compression"]:
        key = f"rank{row['rank']}"
        evidence["spectral"]["compression"].append(
            row | dict(original=original, prediction=picture(data[key], f"svd_{key}", source, key))
        )
    data.close()
    source, data = arrays("spectral", "pca")
    evidence["spectral"]["pca"] = {
        key: data[key].tolist() for key in ["species", "raw_units_scores", "standardized_scores"]
    }
    data.close()
    evidence["spectral"]["qr"] = metrics["qr"]
    evidence["spectral"]["scope"] = metrics["scope"]
    for case in ["speed", "revenue", "blend", "psd", "regulator", "evacuation", "portfolio"]:
        metrics = metric("convex", case)
        source, data = arrays("convex", case)
        evidence["applications"].append(
            dict(case=case, metrics=metrics, arrays={key: data[key].tolist() for key in data.files})
        )
        data.close()
    template = ROOT / "scripts/templates/optimization_explorer.html"
    javascript = ROOT / "scripts/templates/optimization_explorer.js"
    SOURCES.update([template, javascript, Path(__file__).resolve()])
    html = (
        template.read_text()
        .replace(
            "__EVIDENCE__",
            json.dumps(evidence, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c"),
        )
        .replace("__JAVASCRIPT__", javascript.read_text())
    )
    path = ROOT / "docs/explorer.html"
    path.write_text(html)
    # Shared truth/observation names intentionally have identical pixels across penalties;
    # retain every source relation while deduplicating the file-hash map.
    write_json(
        ROOT / "docs/explorer_manifest.json",
        dict(
            status="complete",
            html_sha256=sha256(path),
            sections=8,
            routing_results=84,
            tv_results=72,
            adaptive_campaigns=6,
            source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(SOURCES)},
            images=IMAGE_RECORDS,
        ),
    )
    print(
        "Built eight-concept explorer from 55 complete protocols;",
        len(IMAGE_RECORDS),
        "image/source relations",
    )


if __name__ == "__main__":
    main()
