"""Repository and installed-package resource resolution."""

from pathlib import Path


def config_path(name):
    local = Path("configs") / name
    return local if local.exists() else Path(__file__).with_name("default_configs") / name


def photo_path():
    local = Path("assets/source_data/spectral/source_photo.npz")
    return (
        local if local.exists() else Path(__file__).with_name("example_assets") / "source_photo.npz"
    )
