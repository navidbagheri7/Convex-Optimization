"""Check every published optimization image and embedded measurement against its source."""

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.routing import tour_length


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/explorer_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["sections"] == 8
    page = root / "docs/explorer.html"
    assert sha256(page) == manifest["html_sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert sha256(root / name) == digest, name
    data = json.loads(
        re.search(
            r'<script id="evidence" type="application/json">(.*?)</script>', page.read_text(), re.S
        ).group(1)
    )
    assert (
        len(data["routing"]) == 84
        and len(data["tv"]) == 72
        and len(data["recommendation"]) == 30
        and len(data["adaptive"]) == 6
    )
    cache = {}
    for row in manifest["images"]:
        assert sha256(root / row["path"]) == row["sha256"]
        if row["source"] not in cache:
            with np.load(root / row["source"], allow_pickle=False) as source:
                cache[row["source"]] = {key: source[key] for key in source.files}
        arr = cache[row["source"]][row["key"]]
        if row["index"] is not None:
            arr = arr[row["index"]]
        expected = (arr.clip(0, 1) * 255).round().astype(np.uint8)
        with Image.open(root / row["path"]) as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
    for row in data["routing"]:
        with np.load(
            root / f"assets/source_data/routing/{row['instance']}.npz", allow_pickle=False
        ) as source:
            assert tour_length(source["distance"], np.array(row["tour"])) == row["length"]
            np.testing.assert_array_equal(source["coordinates"], row["coordinates"])
        metrics = json.loads(
            (root / f"results/reference/{row['family']}/{row['instance']}/metrics.json").read_text()
        )
        original = next(
            r for r in metrics["rows"] if r["method"] == row["method"] and r["seed"] == row["seed"]
        )
        assert all(row[key] == value for key, value in original.items())
    for row in data["tv"]:
        name = f"{row['dataset']}_seed{row['seed']}_lambda{row['penalty']:g}"
        metrics = json.loads(
            (root / f"results/reference/primal_dual/{name}/metrics.json").read_text()
        )
        original = next(r for r in metrics["rows"] if r["method"] == row["method"])
        assert all(row[key] == value for key, value in original.items())
        with np.load(
            root / f"assets/source_data/primal_dual/{name}.npz", allow_pickle=False
        ) as source:
            np.testing.assert_array_equal(source[row["method"] + "_history"], row["history"])
            assert row["pixels"] == source["truth"].size
            assert [v["iteration"] for v in row["images"]] == source[
                row["method"] + "_snapshot_iterations"
            ].tolist()
    for row in data["recommendation"]:
        metrics = json.loads(
            (
                root / f"results/reference/{row['protocol']}/seed{row['seed']}/metrics.json"
            ).read_text()
        )
        original = next(r for r in metrics["rows"] if r["method"] == row["method"])
        assert all(row[key] == value for key, value in original.items() if key != "candidates")
        for index, candidate in enumerate(row["candidates"]):
            with np.load(
                root
                / f"assets/source_data/{row['protocol']}/{row['method']}_{row['seed']}_candidate{index}.npz",
                allow_pickle=False,
            ) as source:
                np.testing.assert_array_equal(source["history"], candidate["history"])
    for row in data["adaptive"]:
        metrics = json.loads(
            (
                root
                / f"results/reference/adaptive_search/{row['strategy']}/seed{row['seed']}/metrics.json"
            ).read_text()
        )
        assert all(row[key] == value for key, value in metrics.items() if key != "trials")
        for index, trial in enumerate(row["trials"]):
            assert all(trial[key] == value for key, value in metrics["trials"][index].items())
            with np.load(
                root
                / f"assets/source_data/adaptive_search/{row['strategy']}/seed{row['seed']}_trial{index}.npz",
                allow_pickle=False,
            ) as source:
                np.testing.assert_array_equal(source["history"], trial["history"])
    browser = json.loads((root / "docs/explorer_browser_validation.json").read_text())
    assert (
        browser["status"] == "passed"
        and not browser["javascript_errors"]
        and browser["sections"] == 8
    )
    write_json(
        root / "docs/explorer_validation.json",
        dict(
            status="passed",
            source_bound=True,
            sections=8,
            routing_results=84,
            tv_results=72,
            recommender_models=30,
            adaptive_candidates=96,
            image_source_checks=len(manifest["images"]),
            browser_presets=browser["checked_presets"],
            html_sha256=sha256(page),
        ),
    )
    print("Verified all explorer source hashes, images and fitted-result controls")


if __name__ == "__main__":
    main()
