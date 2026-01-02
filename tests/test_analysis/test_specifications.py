"""Tests for analysis.specifications module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


class TestSpecificationLoading:
    """Tests for specification loading."""

    def test_load_specifications_from_file(self):
        """Test loading specifications from YAML file."""
        from analysis.specifications import load_specifications

        # Create temp YAML file
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'specs.yml'
            spec_path.write_text("""
baseline:
  outcome: y
  treatment: x
  controls: []
  fixed_effects: [unit, time]
  cluster: unit
""")
            specs = load_specifications(spec_path)

            assert 'baseline' in specs
            assert specs['baseline']['outcome'] == 'y'
            assert specs['baseline']['treatment'] == 'x'

    def test_get_specification(self):
        """Test getting a single specification."""
        from analysis.specifications import load_specifications, get_specification

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'specs.yml'
            spec_path.write_text("""
test_spec:
  outcome: outcome_var
  treatment: treatment_var
  controls: [control1, control2]
  fixed_effects: [fe1, fe2]
  cluster: cluster_var
""")
            spec = get_specification('test_spec', path=spec_path)

            assert spec['name'] == 'test_spec'
            assert spec['outcome'] == 'outcome_var'
            assert spec['treatment'] == 'treatment_var'
            assert spec['controls'] == ['control1', 'control2']

    def test_get_unknown_specification_raises(self):
        """Test that unknown specification raises KeyError."""
        from analysis.specifications import get_specification

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / 'specs.yml'
            spec_path.write_text("""
baseline:
  outcome: y
  treatment: x
""")
            with pytest.raises(KeyError, match="Unknown specification"):
                get_specification('nonexistent', path=spec_path)


class TestSpecificationValidation:
    """Tests for specification validation."""

    def test_valid_specification(self):
        """Test that valid specification passes validation."""
        from analysis.specifications import validate_specification

        spec = {
            'outcome': 'y',
            'treatment': 'x',
            'controls': ['z1', 'z2'],
            'fixed_effects': ['unit', 'time'],
            'cluster': 'unit',
        }

        errors = validate_specification(spec)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test that missing required fields are reported."""
        from analysis.specifications import validate_specification

        spec = {'controls': []}  # Missing outcome and treatment

        errors = validate_specification(spec)
        assert len(errors) >= 2
        assert any('outcome' in e for e in errors)
        assert any('treatment' in e for e in errors)

    def test_invalid_field_types(self):
        """Test that invalid field types are reported."""
        from analysis.specifications import validate_specification

        spec = {
            'outcome': 123,  # Should be string
            'treatment': 'x',
            'controls': 'not_a_list',  # Should be list
        }

        errors = validate_specification(spec)
        assert len(errors) >= 2


class TestSpecificationCreation:
    """Tests for programmatic specification creation."""

    def test_create_specification(self):
        """Test creating specification programmatically."""
        from analysis.specifications import create_specification

        spec = create_specification(
            name='my_spec',
            outcome='y',
            treatment='x',
            controls=['z1', 'z2'],
            fixed_effects=['unit', 'time'],
            cluster='unit',
            description='Test specification',
        )

        assert spec['name'] == 'my_spec'
        assert spec['outcome'] == 'y'
        assert spec['treatment'] == 'x'
        assert spec['controls'] == ['z1', 'z2']
        assert spec['description'] == 'Test specification'

    def test_create_minimal_specification(self):
        """Test creating minimal specification."""
        from analysis.specifications import create_specification

        spec = create_specification(
            name='minimal',
            outcome='y',
            treatment='x',
        )

        assert spec['name'] == 'minimal'
        assert spec['controls'] == []
        assert spec['fixed_effects'] == []
        assert spec['cluster'] is None


class TestFormulaConversion:
    """Tests for specification to formula conversion."""

    def test_r_formula_with_fe(self):
        """Test R formula generation with fixed effects."""
        from analysis.specifications import spec_to_formula

        spec = {
            'outcome': 'y',
            'treatment': 'x',
            'controls': ['z1', 'z2'],
            'fixed_effects': ['unit', 'time'],
        }

        formula = spec_to_formula(spec, engine='r')

        assert 'y ~' in formula
        assert 'x + z1 + z2' in formula
        assert '| unit + time' in formula

    def test_r_formula_without_fe(self):
        """Test R formula generation without fixed effects."""
        from analysis.specifications import spec_to_formula

        spec = {
            'outcome': 'y',
            'treatment': 'x',
            'controls': [],
            'fixed_effects': [],
        }

        formula = spec_to_formula(spec, engine='r')

        assert formula == 'y ~ x'
        assert '|' not in formula
