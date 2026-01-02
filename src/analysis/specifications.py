"""
Specification Management for Analysis Engines.

Handles loading and validation of language-agnostic estimation specifications.
Specifications define what to estimate, independent of how (which engine).

Usage
-----
    from analysis.specifications import load_specifications, get_specification

    # Load all specifications
    specs = load_specifications()

    # Get a specific specification
    spec = get_specification('baseline')

    # Validate a specification
    errors = validate_specification(spec)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import yaml


def load_specifications(path: Optional[Path] = None) -> dict[str, dict]:
    """
    Load specifications from YAML file.

    Parameters
    ----------
    path : Path, optional
        Path to specifications file. Defaults to SPECIFICATIONS_FILE from config,
        or specifications.yml in project root.

    Returns
    -------
    dict[str, dict]
        Dictionary of specification_name -> specification_dict

    Raises
    ------
    FileNotFoundError
        If specifications file not found
    yaml.YAMLError
        If YAML parsing fails
    """
    if path is None:
        path = _get_default_spec_path()

    if not path.exists():
        raise FileNotFoundError(f"Specifications file not found: {path}")

    with open(path) as f:
        specs = yaml.safe_load(f)

    if specs is None:
        return {}

    return specs


def get_specification(name: str, path: Optional[Path] = None) -> dict:
    """
    Get a single specification by name.

    Parameters
    ----------
    name : str
        Specification name
    path : Path, optional
        Path to specifications file

    Returns
    -------
    dict
        Specification dictionary

    Raises
    ------
    KeyError
        If specification not found
    """
    specs = load_specifications(path)

    if name not in specs:
        available = ', '.join(sorted(specs.keys()))
        raise KeyError(
            f"Unknown specification: '{name}'. Available: {available}"
        )

    spec = specs[name].copy()
    spec['name'] = name  # Ensure name is included
    return spec


def validate_specification(spec: dict) -> list[str]:
    """
    Validate a specification dictionary.

    Parameters
    ----------
    spec : dict
        Specification to validate

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required fields
    required = ['outcome', 'treatment']
    for field in required:
        if field not in spec:
            errors.append(f"Missing required field: '{field}'")
        elif not isinstance(spec[field], str):
            errors.append(f"Field '{field}' must be a string")

    # Optional fields with type checks
    if 'controls' in spec:
        if not isinstance(spec['controls'], list):
            errors.append("Field 'controls' must be a list")
        elif not all(isinstance(c, str) for c in spec['controls']):
            errors.append("All items in 'controls' must be strings")

    if 'fixed_effects' in spec:
        if not isinstance(spec['fixed_effects'], list):
            errors.append("Field 'fixed_effects' must be a list")
        elif not all(isinstance(fe, str) for fe in spec['fixed_effects']):
            errors.append("All items in 'fixed_effects' must be strings")

    if 'cluster' in spec:
        if spec['cluster'] is not None and not isinstance(spec['cluster'], str):
            errors.append("Field 'cluster' must be a string or null")

    return errors


def list_specifications(path: Optional[Path] = None) -> list[str]:
    """
    List all available specification names.

    Parameters
    ----------
    path : Path, optional
        Path to specifications file

    Returns
    -------
    list[str]
        List of specification names
    """
    specs = load_specifications(path)
    return sorted(specs.keys())


def create_specification(
    name: str,
    outcome: str,
    treatment: str,
    controls: Optional[list[str]] = None,
    fixed_effects: Optional[list[str]] = None,
    cluster: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """
    Create a specification dictionary programmatically.

    Parameters
    ----------
    name : str
        Specification name
    outcome : str
        Outcome variable name
    treatment : str
        Treatment variable name
    controls : list[str], optional
        Control variable names
    fixed_effects : list[str], optional
        Fixed effect variable names
    cluster : str, optional
        Cluster variable for standard errors
    description : str, optional
        Human-readable description

    Returns
    -------
    dict
        Specification dictionary
    """
    spec = {
        'name': name,
        'outcome': outcome,
        'treatment': treatment,
        'controls': controls or [],
        'fixed_effects': fixed_effects or [],
        'cluster': cluster,
    }

    if description:
        spec['description'] = description

    return spec


def spec_to_formula(spec: dict, engine: str = 'python') -> str:
    """
    Convert specification to formula string for a specific engine.

    Parameters
    ----------
    spec : dict
        Specification dictionary
    engine : str
        Target engine ('python', 'r', 'stata')

    Returns
    -------
    str
        Formula string in engine-specific syntax
    """
    outcome = spec['outcome']
    treatment = spec['treatment']
    controls = spec.get('controls', [])
    fixed_effects = spec.get('fixed_effects', [])

    # Build RHS
    rhs_vars = [treatment] + controls
    rhs = ' + '.join(rhs_vars)

    if engine == 'r':
        # R/fixest formula: outcome ~ treatment + controls | fe1 + fe2
        if fixed_effects:
            fe_part = ' + '.join(fixed_effects)
            return f"{outcome} ~ {rhs} | {fe_part}"
        return f"{outcome} ~ {rhs}"

    elif engine == 'stata':
        # Stata/reghdfe: reghdfe outcome treatment controls, absorb(fe1 fe2)
        if fixed_effects:
            fe_part = ' '.join(fixed_effects)
            return f"{outcome} {rhs}, absorb({fe_part})"
        return f"{outcome} {rhs}"

    else:
        # Python/default: simple formula
        return f"{outcome} ~ {rhs}"


def _get_default_spec_path() -> Path:
    """Get default specifications file path."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import SPECIFICATIONS_FILE
        return SPECIFICATIONS_FILE
    except ImportError:
        # Fallback to project root
        return Path(__file__).parent.parent.parent / 'specifications.yml'
