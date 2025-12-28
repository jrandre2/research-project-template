#!/usr/bin/env python3
"""
Tests for src/agents/structure_mapper.py

Tests cover:
- MappingRule dataclass
- StructureMapping dataclass
- StructureMapper class
- map_project convenience function
"""
from __future__ import annotations

import pytest
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.structure_mapper import (
    MappingRule,
    StructureMapping,
    StructureMapper,
    map_project,
)
from agents.project_analyzer import (
    ProjectAnalysis,
    ModuleInfo,
    DirectoryInfo,
)


# ============================================================
# MAPPING RULE TESTS
# ============================================================

class TestMappingRule:
    """Tests for the MappingRule dataclass."""

    def test_creates_mapping_rule(self):
        """Test creating a mapping rule."""
        rule = MappingRule(
            source_pattern='src/*.py',
            target_location='src/stages/',
            action='copy',
            notes='Python files'
        )

        assert rule.source_pattern == 'src/*.py'
        assert rule.target_location == 'src/stages/'
        assert rule.action == 'copy'
        assert rule.notes == 'Python files'

    def test_default_notes(self):
        """Test default empty notes."""
        rule = MappingRule(
            source_pattern='data/*',
            target_location='data_raw/',
            action='move'
        )

        assert rule.notes == ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rule = MappingRule(
            source_pattern='tests/*',
            target_location='tests/',
            action='copy',
            notes='Test suite'
        )

        d = rule.to_dict()

        assert d['source_pattern'] == 'tests/*'
        assert d['target_location'] == 'tests/'
        assert d['action'] == 'copy'
        assert d['notes'] == 'Test suite'

    def test_action_types(self):
        """Test various action types."""
        actions = ['move', 'merge', 'transform', 'copy']
        for action in actions:
            rule = MappingRule(
                source_pattern='*',
                target_location='dest/',
                action=action
            )
            assert rule.action == action


# ============================================================
# STRUCTURE MAPPING TESTS
# ============================================================

class TestStructureMapping:
    """Tests for the StructureMapping dataclass."""

    def test_creates_structure_mapping(self):
        """Test creating a structure mapping."""
        mapping = StructureMapping(
            source_project='/path/to/source',
            target_template='CENTAUR'
        )

        assert mapping.source_project == '/path/to/source'
        assert mapping.target_template == 'CENTAUR'
        assert mapping.rules == []
        assert mapping.warnings == []

    def test_with_rules(self):
        """Test mapping with rules."""
        rules = [
            MappingRule('src/*', 'src/stages/', 'copy'),
            MappingRule('data/*', 'data_raw/', 'move'),
        ]
        mapping = StructureMapping(
            source_project='/project',
            target_template='CENTAUR',
            rules=rules
        )

        assert len(mapping.rules) == 2

    def test_with_warnings(self):
        """Test mapping with warnings."""
        mapping = StructureMapping(
            source_project='/project',
            target_template='CENTAUR',
            warnings=['Unmapped file: config.ini', 'Notebooks need review']
        )

        assert len(mapping.warnings) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rule = MappingRule('src/*', 'dest/', 'copy')
        mapping = StructureMapping(
            source_project='/src',
            target_template='target',
            rules=[rule],
            warnings=['warning1']
        )

        d = mapping.to_dict()

        assert d['source_project'] == '/src'
        assert d['target_template'] == 'target'
        assert len(d['rules']) == 1
        assert d['warnings'] == ['warning1']

    def test_to_json(self):
        """Test JSON serialization."""
        mapping = StructureMapping(
            source_project='/src',
            target_template='CENTAUR'
        )

        json_str = mapping.to_json()
        parsed = json.loads(json_str)

        assert parsed['source_project'] == '/src'
        assert parsed['target_template'] == 'CENTAUR'

    def test_to_json_with_indent(self):
        """Test JSON serialization with custom indent."""
        mapping = StructureMapping(
            source_project='/src',
            target_template='CENTAUR'
        )

        json_str = mapping.to_json(indent=4)
        assert '\n    ' in json_str  # 4-space indent

    def test_summary(self):
        """Test summary generation."""
        rule = MappingRule('data/*', 'data_raw/', 'copy', 'Raw data files')
        mapping = StructureMapping(
            source_project='/project',
            target_template='CENTAUR',
            rules=[rule],
            warnings=['Check notebooks']
        )

        summary = mapping.summary()

        assert '# Structure Mapping' in summary
        assert '/project' in summary
        assert 'CENTAUR' in summary
        assert 'data/*' in summary
        assert 'data_raw/' in summary
        assert 'Warnings' in summary
        assert 'Check notebooks' in summary

    def test_summary_no_warnings(self):
        """Test summary without warnings."""
        mapping = StructureMapping(
            source_project='/project',
            target_template='CENTAUR'
        )

        summary = mapping.summary()

        assert 'Warnings' not in summary


# ============================================================
# STRUCTURE MAPPER TESTS
# ============================================================

class TestStructureMapper:
    """Tests for the StructureMapper class."""

    def test_template_stages_defined(self):
        """Test that template stages are defined."""
        assert len(StructureMapper.TEMPLATE_STAGES) > 0

        # Check expected stages exist
        stage_names = [s[0] for s in StructureMapper.TEMPLATE_STAGES]
        assert 's00_ingest' in stage_names
        assert 's03_estimation' in stage_names
        assert 's05_figures' in stage_names

    def test_creates_mapper(self, temp_dir):
        """Test creating a structure mapper."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test'
        )
        mapper = StructureMapper(analysis)

        assert mapper.analysis is analysis

    def test_generate_mapping_empty_project(self, temp_dir):
        """Test mapping generation for empty project."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test'
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        assert mapping.source_project == str(temp_dir)
        assert mapping.target_template == 'CENTAUR'

    def test_maps_data_directory(self, temp_dir):
        """Test mapping of data directories."""
        dir_info = DirectoryInfo(
            path=temp_dir / 'data',
            name='data',
            purpose='data files',
            file_count=5,
            subdirs=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            directories=[dir_info]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a rule for data directory
        data_rules = [r for r in mapping.rules if 'data' in r.source_pattern]
        assert len(data_rules) >= 1
        assert any(r.target_location == 'data_raw/' for r in data_rules)

    def test_maps_output_directory(self, temp_dir):
        """Test mapping of output directories."""
        dir_info = DirectoryInfo(
            path=temp_dir / 'figures',
            name='figures',
            purpose='figure outputs',
            file_count=10,
            subdirs=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            directories=[dir_info]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a rule for output directory
        fig_rules = [r for r in mapping.rules if 'figures' in r.source_pattern]
        assert len(fig_rules) >= 1

    def test_maps_docs_directory(self, temp_dir):
        """Test mapping of documentation."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_docs=True
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a rule for docs
        doc_rules = [r for r in mapping.rules if 'docs' in r.source_pattern]
        assert len(doc_rules) >= 1

    def test_maps_tests_directory(self, temp_dir):
        """Test mapping of test suite."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_tests=True
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a rule for tests
        test_rules = [r for r in mapping.rules if 'tests' in r.source_pattern]
        assert len(test_rules) >= 1

    def test_maps_module_to_ingest_stage(self, temp_dir):
        """Test mapping module with data loading keywords to s00_ingest."""
        module_path = temp_dir / 'loader.py'
        module_path.write_text('# Data loader')

        module = ModuleInfo(
            path=module_path,
            name='loader',
            docstring='Loads data',
            imports=[],
            functions=['load_data'],
            classes=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=[module]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should map to s00_ingest
        ingest_rules = [r for r in mapping.rules if 's00_ingest' in r.target_location]
        assert len(ingest_rules) >= 1

    def test_maps_module_to_estimation_stage(self, temp_dir):
        """Test mapping module with model keywords to s03_estimation."""
        module_path = temp_dir / 'regression_model.py'
        module_path.write_text('# Model')

        module = ModuleInfo(
            path=module_path,
            name='regression_model',
            docstring='Regression models',
            imports=[],
            functions=['fit_model'],
            classes=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=[module]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should map to s03_estimation
        est_rules = [r for r in mapping.rules if 's03_estimation' in r.target_location]
        assert len(est_rules) >= 1

    def test_maps_utility_module(self, temp_dir):
        """Test mapping utility modules."""
        util_dir = temp_dir / 'utils'
        util_dir.mkdir()
        module_path = util_dir / 'helpers.py'
        module_path.write_text('# Utilities')

        module = ModuleInfo(
            path=module_path,
            name='helpers',
            docstring='Helper functions',
            imports=[],
            functions=['helper1'],
            classes=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=[module]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should map to src/utils/
        util_rules = [r for r in mapping.rules if 'utils' in r.target_location]
        assert len(util_rules) >= 1

    def test_generates_warning_for_unmapped_module(self, temp_dir):
        """Test warnings for unmapped modules."""
        module_path = temp_dir / 'config.py'
        module_path.write_text('# Config')

        module = ModuleInfo(
            path=module_path,
            name='config',
            docstring='Configuration',
            imports=[],
            functions=[],
            classes=[]
        )
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=[module]
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a warning about unmapped module
        assert any('Unmapped module' in w for w in mapping.warnings)

    def test_generates_warning_for_notebooks(self, temp_dir):
        """Test warnings for Jupyter notebooks."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_notebooks=True
        )
        mapper = StructureMapper(analysis)

        mapping = mapper.generate_mapping()

        # Should have a warning about notebooks
        assert any('notebook' in w.lower() for w in mapping.warnings)


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================

class TestMapProject:
    """Tests for the map_project convenience function."""

    def test_maps_project(self, temp_dir):
        """Test convenience function."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test'
        )

        mapping = map_project(analysis)

        assert isinstance(mapping, StructureMapping)
        assert mapping.source_project == str(temp_dir)

    def test_returns_complete_mapping(self, temp_dir):
        """Test that convenience function returns complete mapping."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_docs=True,
            has_tests=True
        )

        mapping = map_project(analysis)

        assert len(mapping.rules) >= 2  # At least docs and tests

