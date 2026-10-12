"""Console entry point with normal process exit semantics."""

from .cli import main as run


def main():
    # Study functions return data for Python callers. Console scripts must not
    # pass those objects to sys.exit, which treats them as error messages.
    run()
