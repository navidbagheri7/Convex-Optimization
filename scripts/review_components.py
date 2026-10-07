"""Inventory authored source and audit English content, local links and private exclusions."""

import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from optimization_lab.artifacts import sha256, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    components = []
    for directory in ["src", "scripts", "tests"]:
        for path in sorted((root / directory).rglob("*.py")):
            tree = ast.parse(path.read_text())
            components.append(
                dict(
                    path=str(path.relative_to(root)),
                    sha256=sha256(path),
                    lines=len(path.read_text().splitlines()),
                    module_docstring=ast.get_docstring(tree),
                    public_definitions=[
                        n.name
                        for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                        and not n.name.startswith("_")
                    ],
                )
            )
    assert 'version = "1.1.0"' in (root / "pyproject.toml").read_text()
    assert 'version: "1.1.0"' in (root / "CITATION.cff").read_text()
    public = sorted(
        set(
            subprocess.check_output(
                ["git", "ls-files", "-co", "--exclude-standard"], cwd=root, text=True
            ).splitlines()
        )
    )
    issues = []
    for name in public:
        path = root / name
        if not path.is_file():
            continue
        if (
            name.startswith((".local/", "legacy/originals/", "results/smoke/"))
            or "/checkpoints/" in name
        ):
            issues.append("Private artifact: " + name)
        if (
            path.suffix in [".pt", ".pth", ".cbm", ".joblib", ".rar", ".whl", ".gz"]
            or path.stat().st_size > 10 * 1024 * 1024
        ):
            issues.append("Large or raw artifact: " + name)
        if path.suffix in [".md", ".py", ".js", ".html", ".ipynb"]:
            text = path.read_text()
            if re.search(r"[\u0600-\u06ff]", text):
                issues.append("Non-English authored content: " + name)
            if re.search(r"gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}", text):
                issues.append("Credential pattern: " + name)
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                target = unquote(link.strip("<>").split("#")[0])
                if target and not (path.parent / target).exists():
                    issues.append(f"Broken local link: {name}: {target}")
    report = dict(
        status="passed" if not issues else "failed",
        version="1.1.0",
        scope="AST completeness and authored-content/publication checks; numerical/experimental review is separate in implementation_audit.md and validation.json.",
        authored_components=len(components),
        source_modules=sum(r["path"].startswith("src/") for r in components),
        components=components,
        publishable_files=len(public),
        issues=issues,
    )
    write_json(root / "docs/component_inventory.json", report)
    if issues:
        raise RuntimeError("\n".join(issues))
    print(
        "Reviewed inventory:",
        len(components),
        "components;",
        len(public),
        "publishable files; English and local links checked",
    )


if __name__ == "__main__":
    main()
