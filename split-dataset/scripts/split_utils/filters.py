"""Dataset filtering and exclusion."""

from typing import Any, Dict, List


# config key -> dataset key
_FIELD_MAP = {
    "axial_stretch": "Axial Stretch",
    "pressure":      "Pressure",
    "region":        "Region ",      # trailing space in dataset
    "classe":        "Classe",
    "genetic":       "Genetic",
    "sex":           "Sex",
    "orientation":   "Orientation",
}


def _matches(value: Any, criterion: Any) -> bool:
    """True if *value* matches *criterion* (single value, list, or None)."""
    if criterion is None:
        return True
    if not isinstance(criterion, list):
        criterion = [criterion]
    return str(value) in [str(c) for c in criterion]


def filter_stacks(stacks: List[Dict], filters: Dict) -> List[Dict]:
    """Keep only stacks that pass every configured filter."""
    result: List[Dict] = []
    for stack in stacks:
        if "infos" not in stack:
            continue
        infos = stack["infos"]
        try:
            keep = True
            # Age (range)
            age_cfg = filters.get("age")
            if age_cfg is not None:
                age = int(infos.get("Age (wk)", "999"))
                if age_cfg.get("min") is not None and age < age_cfg["min"]:
                    keep = False
                if age_cfg.get("max") is not None and age > age_cfg["max"]:
                    keep = False
            # Other fields (exact / list)
            for cfg_key, ds_key in _FIELD_MAP.items():
                if not keep:
                    break
                fval = filters.get(cfg_key)
                if fval is None:
                    continue
                raw = infos.get(ds_key, "")
                val = raw.strip() if isinstance(raw, str) else raw
                if not _matches(val, fval):
                    keep = False
            if keep:
                result.append(stack)
        except (ValueError, KeyError):
            continue
    return result


def exclude_stacks_by_id(stacks: List[Dict], ids: List[str]) -> List[Dict]:
    """Remove stacks whose id appears in *ids*."""
    bad = {str(s).strip() for s in ids if s}
    if not bad:
        return stacks
    before = len(stacks)
    out = [s for s in stacks if s.get("id") not in bad]
    print(f"  Excluded {before - len(out)} stack(s): {', '.join(sorted(bad))}")
    return out
