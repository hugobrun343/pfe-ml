"""Paths and stack↔file helpers for stats."""

from pathlib import Path
from typing import Any, Dict


def get_data_dir(norm_config: Dict[str, Any], config_path: Path) -> Path:
    """Resolve data_dir from config or default to <project>/data.

    Project root = config_path.parent.parent (config assumed under config/).
    """
    data_dir = norm_config.get("data_dir")
    if data_dir is not None:
        p = Path(data_dir)
        return p.resolve() if p.is_absolute() else (config_path.parent.parent / p).resolve()
    return (config_path.parent.parent / "data").resolve()


def _ensure_data_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _nii_name(stack: Dict[str, Any]) -> str:
    nii = stack.get("nii_path") or ""
    name = Path(nii).name
    return name.replace(".nii", ".nii.gz") if name else ""


def _vol_path(stack: Dict[str, Any], data_root: Path) -> Path:
    return data_root / _nii_name(stack)
