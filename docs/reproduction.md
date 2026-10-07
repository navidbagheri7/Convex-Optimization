# Reproduction

Use Python 3.10 or newer. The measured Python 3.10 stack is recorded in `requirements-reference.txt`; install it in a private environment. GPU recommendation was fitted with PyTorch 2.7.1+cu128 on L40S devices, float32 and TF32 disabled. All convex / spectral / routing references run on CPU. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for the recorded small dense studies; machine load still affects time measurements.

```bash
python -m pip install -e '.[recommendation,routing,notebooks,dev]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
optimization-lab convex
optimization-lab first-order
optimization-lab spectral
optimization-lab data --dataset movielens --data-root "$OPTIMIZATION_DATA_ROOT"
optimization-lab data --dataset routing --data-root "$OPTIMIZATION_DATA_ROOT"
CUDA_VISIBLE_DEVICES=1 optimization-lab recommendation --data-root "$OPTIMIZATION_DATA_ROOT" --seed 0 --device cuda:0
CUDA_VISIBLE_DEVICES=2 optimization-lab recommendation --data-root "$OPTIMIZATION_DATA_ROOT" --seed 1 --device cuda:0
CUDA_VISIBLE_DEVICES=1 optimization-lab recommendation --data-root "$OPTIMIZATION_DATA_ROOT" --seed 2 --device cuda:0
optimization-lab routing --instance-root "$OPTIMIZATION_DATA_ROOT/optimization/routing"
python -m optimization_lab.memetic --instance-root "$OPTIMIZATION_DATA_ROOT/optimization/routing"
python scripts/run_auxiliary.py
optimization-lab report
python scripts/execute_notebooks.py
optimization-lab validate
python scripts/audit_checkpoints.py --data-root "$OPTIMIZATION_DATA_ROOT"
```

The commands are sequential; shell-level concurrent recommendation fits are optional if devices have room. Leave unrelated shared GPU workloads alone. In this execution GPU0 was left untouched, while GPU1 / GPU2 hosted the fitting jobs.

The initial fixed-penalty study is reproducible with `configs/recommendation_fixed.json` and a separate output directory. Its exact authored source snapshot is retained in `docs/source_snapshots/recommendation_fixed.py`. The main current implementation supports the same fixed grid when `lambdas` is omitted. Full budgets are in the JSON configs and `docs/protocol.md`. All selected parameters and every candidate history are saved. Existing reference evidence should be copied or removed deliberately before a new campaign; aborted runs are marked running and cannot pass release validation.

The MovieLens ZIP is checked against the recorded SHA256 before extraction. ZIP paths are checked for traversal. Routing loader checks the embedded NAME and EUC_2D convention and detects gzip by bytes; the historical extensions are not trusted. Direct TSPLIB downloads depend on availability of the original host; the executed campaign used the supplied course instances, whose hashes are recorded. An offline fallback is `python scripts/recover_instances.py --data-root "$OPTIMIZATION_DATA_ROOT"`: it reconstructs the original instance coordinates from the included attributed numeric witnesses, verifies the EUC_2D distance matrix after roundtrip and never overwrites a different cache file. Their verified copies are under `/data/navid.bagheri-shouraki/ML_data/optimization/routing` on the author's workstation. On another machine, choose your own personal cache and download directly.

The notebooks execute small oracles and inspect included complete reference evidence. They do not require the private coursework archive, raw MovieLens data or a GPU. Full recommendation reruns and the optional checkpoint audit require the official personal cache. The installed CLI's small deterministic examples work from an arbitrary directory. Packaging checks build a wheel / source distribution, extract the latter to a fresh directory, and execute tests, public evidence validation and all notebooks again. A wheel is also installed into a separate private environment and tested outside the source tree.

Floating-point and GPU execution need not be bitwise identical across machines. Scientific tolerances, data roles, objective conventions, seed sets and certificate scope are explicit. Checkpoint NPZ files have portable NumPy inference; exact optimizer-state continuation is not promised. This repository has no hidden automatic data upload or GitHub push.

## Version 1.1 comparison and browser extension

Install `.[search]` for pinned Optuna 5.0.0. The existing scientific stack remains
a tested reference, rather than a claim that all pins are the newest versions.

```bash
optimization-lab primal-dual
# Repeat seeds 0,1,2 and strategies random,tpe; each campaign is sequential.
optimization-lab adaptive-search --seed 0 --strategy random --device cuda:0 --data-root /data/navid.bagheri-shouraki/ML_data
python scripts/report_extensions.py
python scripts/audit_adaptive_checkpoints.py
python scripts/create_explorer.py
python scripts/execute_notebooks.py
optimization-lab validate
```

Primal–dual comparisons require no download: the attributed installed photo and
procedural geometry are bundled. The 96 search candidates use the already prepared
official MovieLens-100K split. Searches select on validation; test is evaluated once
for the winner of each completed campaign. Incomplete adaptive searches reject
continuation before replacing their records; use a fresh `--output` root.
Complete campaigns skip after checking config and scientific source identities.

The [browser guide](interactive_explorer.md) provides rebuild, optional Chromium
verification and manual hosting instructions. Templates and views are in the source
archive; the wheel contains reusable solvers/configs and portable factor inference.
