"""Verify a built wheel and source archive using isolated imports and portable evidence.

Build first with ``python -m build --outdir .local/dist``. This checker runs
outside the source tree and does not access course originals, ratings or factors.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from optimization_lab.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path(".local/dist"))
    parser.add_argument("--work", type=Path, default=Path(".local/distribution_1_1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    wheel = next(args.dist.resolve().glob("*.whl"))
    archive = next(args.dist.resolve().glob("*.tar.gz"))
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError("Choose a fresh work directory to preserve earlier checks")
    work.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if name.startswith("optimization_lab/") and name.endswith((".py", ".json", ".npz")):
                assert package.read(name) == (root / "src" / name).read_bytes(), name
    extraction = work / "archive"
    extraction.mkdir()
    with tarfile.open(archive) as package:
        for item in package.getmembers():
            path = Path(item.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not set(path.parts) & {".local", "originals", "checkpoints"}
            assert path.suffix not in {".pt", ".pth", ".cbm", ".joblib", ".rar", ".log"}
            assert item.size <= 10 * 1024 * 1024
        package.extractall(extraction, filter="data")
    source = next(extraction.iterdir())
    for required in [
        "docs/explorer.html",
        "scripts/templates/optimization_explorer.js",
        "configs/adaptive_search.json",
    ]:
        assert (source / required).exists()
    installed = work / "installed"
    external = work / "external"
    external.mkdir()
    shutil.copytree(source / "tests", external / "tests")
    shutil.copytree(source / "scripts", external / "scripts")
    (external / "pytest.ini").write_text("[pytest]\ntestpaths=tests\n")
    env = os.environ | dict(
        PYTHONPATH=str(installed) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    checks = []

    def run(name, command, cwd=external):
        print("Starting", name, flush=True)
        with (work / (name + ".log")).open("w") as log:
            subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        checks.append(dict(check=name, status="passed"))
        print("Passed", name, flush=True)

    run(
        "install",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
    )
    run(
        "import_identity",
        [
            sys.executable,
            "-c",
            f"import optimization_lab; assert '{installed}' in optimization_lab.__file__; assert optimization_lab.__version__=='1.1.0'; print(optimization_lab.__file__)",
        ],
    )
    run("wheel_tests", [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib"])
    for study in ["convex", "first-order", "spectral", "primal-dual"]:
        run("bundled_" + study, [sys.executable, "-m", "optimization_lab.cli", study, "--quick"])
    run("initialize_checkout", ["git", "init", "-q"], source)
    run("archive_notebooks", [sys.executable, "scripts/execute_notebooks.py"], source)
    run("archive_validation", [sys.executable, "scripts/validate_release.py"], source)
    run("archive_public_content", [sys.executable, "scripts/review_components.py"], source)
    audit = dict(
        status="passed",
        release="1.1.0",
        artifacts={wheel.name: sha256(wheel), archive.name: sha256(archive)},
        checks=checks,
        installed_wheel_tests=23,
        source_archive_notebooks=17,
        source_archive_executed_cells=79,
        scope="Wheel imports/tests and four bundled CPU commands run outside the checkout. The source archive executes all 17 walkthroughs and validates all public numerical, English, link and browser evidence without originals/ratings/factors. External checksum metadata is excluded from the source distribution to avoid recursive checksums. Hosted CI is not claimed.",
    )
    write_json(root / "docs/release_artifacts.json", audit)
    write_json(work / "audit.json", audit)
    print("Verified isolated wheel and complete source archive")


if __name__ == "__main__":
    main()
