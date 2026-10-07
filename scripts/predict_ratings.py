"""Predict ratings from a portable checkpoint and zero-based integer ID pairs."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from optimization_lab.artifacts import sha256, write_json
from optimization_lab.deployment import predict_ratings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with user_id,item_id; IDs are zero-based, unknown IDs use the documented bias/mean fallback",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"user_id", "item_id"} <= set(reader.fieldnames):
            raise ValueError("CSV requires user_id and item_id columns")
        rows = list(reader)
        fields = reader.fieldnames
    if not rows:
        raise ValueError("At least one observation is required")
    arrays = []
    for name in ["user_id", "item_id"]:
        values = [row[name] for row in rows]
        if any(not re.fullmatch(r"[+-]?[0-9]+", value or "") for value in values):
            raise ValueError("ID fields must contain finite integer literals")
        ids = [int(value) for value in values]
        if any(value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max for value in ids):
            raise ValueError("ID exceeds the int64 schema")
        arrays.append(np.array(ids, dtype=np.int64))
    predictions = predict_ratings(args.checkpoint, *arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, "predicted_rating"])
        writer.writeheader()
        writer.writerows(
            row | dict(predicted_rating=float(value)) for row, value in zip(rows, predictions)
        )
    write_json(
        args.output.with_suffix(".json"),
        dict(
            checkpoint_sha256=sha256(args.checkpoint),
            input_sha256=sha256(args.input),
            output_sha256=sha256(args.output),
            observations=len(rows),
            id_schema="zero-based integer user_id,item_id; unknown user/item IDs use available bias plus fitting mean",
            rating_range=[1, 5],
            inference_source_sha256=sha256(Path(__file__).resolve()),
            scope="Point predictions from validation-selected factors; no predictive uncertainty or online recommendation claim.",
        ),
    )
    print("Predicted", len(rows), "rating pairs")


if __name__ == "__main__":
    main()
