# Contributing

Contributions should preserve the mathematical objective and declared data roles. Add a small analytical / independent oracle for a new algorithm or corrected numerical behavior; do not add a test that merely copies the implementation. Explain boundary conditions, information access and certificate scope.

Run `ruff check src tests scripts`, `ruff format --check src tests scripts`, `pytest -q --tb=short`, and `optimization-lab validate`. A change to a source-bound scientific implementation requires rerunning the affected reference study and regenerating its figures / notebooks. Keep smoke runs separate. Report every fitting seed and use validation, never final-test scores, for model selection. New datasets require provenance, correct licensing and placement in a personal cache.

Do not commit raw course archives, ratings, private environments, secrets or checkpoints. Keep all new documentation and notebooks in English. PR descriptions should state the numerical problem, resulting behavior and meaningful validation performed.
