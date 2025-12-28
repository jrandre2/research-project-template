#!/usr/bin/env python3
"""
Tests for src/agents/migration_planner.py

Tests cover:
- MigrationStep dataclass
- MigrationPlan dataclass
- MigrationPlanner class
- generate_migration_plan convenience function
"""
from __future__ import annotations

import pytest
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.migration_planner import (
    MigrationStep,
    MigrationPlan,
    MigrationPlanner,
    generate_migration_plan,
)
from agents.structure_mapper import (
    MappingRule,
    StructureMapping,
)
from agents.project_analyzer import (
    ProjectAnalysis,
    ModuleInfo,
)


# ============================================================
# MIGRATION STEP TESTS
# ============================================================

class TestMigrationStep:
    """Tests for the MigrationStep dataclass."""

    def test_creates_migration_step(self):
        """Test creating a migration step."""
        step = MigrationStep(
            order=1,
            category='setup',
            action='Create directory structure'
        )

        assert step.order == 1
        assert step.category == 'setup'
        assert step.action == 'Create directory structure'
        assert step.completed is False

    def test_with_source_and_target(self):
        """Test step with source and target."""
        step = MigrationStep(
            order=2,
            category='copy',
            action='Copy data files',
            source='data/*',
            target='data_raw/'
        )

        assert step.source == 'data/*'
        assert step.target == 'data_raw/'

    def test_default_values(self):
        """Test default values."""
        step = MigrationStep(
            order=1,
            category='verify',
            action='Check imports'
        )

        assert step.source is None
        assert step.target is None
        assert step.details == ""
        assert step.completed is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        step = MigrationStep(
            order=3,
            category='transform',
            action='Merge modules',
            source='src/mod1.py, src/mod2.py',
            target='src/stages/s00_ingest.py',
            details='Combine data loading code',
            completed=True
        )

        d = step.to_dict()

        assert d['order'] == 3
        assert d['category'] == 'transform'
        assert d['action'] == 'Merge modules'
        assert d['source'] == 'src/mod1.py, src/mod2.py'
        assert d['target'] == 'src/stages/s00_ingest.py'
        assert d['details'] == 'Combine data loading code'
        assert d['completed'] is True

    def test_to_markdown_uncompleted(self):
        """Test markdown formatting for uncompleted step."""
        step = MigrationStep(
            order=1,
            category='setup',
            action='Initialize git',
            details='git init'
        )

        md = step.to_markdown()

        assert '[ ]' in md  # Unchecked
        assert 'SETUP' in md
        assert 'Initialize git' in md

    def test_to_markdown_completed(self):
        """Test markdown formatting for completed step."""
        step = MigrationStep(
            order=1,
            category='copy',
            action='Copy files',
            source='src/*',
            target='dest/',
            completed=True
        )

        md = step.to_markdown()

        assert '[x]' in md  # Checked
        assert 'COPY' in md
        assert 'src/*' in md
        assert 'dest/' in md

    def test_category_types(self):
        """Test various category types."""
        categories = ['setup', 'copy', 'transform', 'generate', 'verify']
        for i, cat in enumerate(categories):
            step = MigrationStep(order=i, category=cat, action='Test')
            assert step.category == cat


# ============================================================
# MIGRATION PLAN TESTS
# ============================================================

class TestMigrationPlan:
    """Tests for the MigrationPlan dataclass."""

    def test_creates_migration_plan(self):
        """Test creating a migration plan."""
        plan = MigrationPlan(
            source_project='/path/to/source',
            target_location='/path/to/target'
        )

        assert plan.source_project == '/path/to/source'
        assert plan.target_location == '/path/to/target'
        assert plan.steps == []
        assert plan.estimated_complexity == 'medium'
        assert plan.notes == []

    def test_created_at_auto_generated(self):
        """Test that created_at is auto-generated."""
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst'
        )

        assert plan.created_at is not None
        assert len(plan.created_at) > 0

    def test_with_steps(self):
        """Test plan with steps."""
        steps = [
            MigrationStep(1, 'setup', 'Create dirs'),
            MigrationStep(2, 'copy', 'Copy files'),
        ]
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst',
            steps=steps
        )

        assert len(plan.steps) == 2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        step = MigrationStep(1, 'setup', 'Init')
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst',
            steps=[step],
            estimated_complexity='low',
            notes=['Note 1']
        )

        d = plan.to_dict()

        assert d['source_project'] == '/src'
        assert d['target_location'] == '/dst'
        assert len(d['steps']) == 1
        assert d['estimated_complexity'] == 'low'
        assert d['notes'] == ['Note 1']

    def test_to_json(self):
        """Test JSON serialization."""
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst'
        )

        json_str = plan.to_json()
        parsed = json.loads(json_str)

        assert parsed['source_project'] == '/src'

    def test_to_markdown(self):
        """Test markdown generation."""
        steps = [
            MigrationStep(1, 'setup', 'Create dirs'),
            MigrationStep(2, 'verify', 'Run tests'),
        ]
        plan = MigrationPlan(
            source_project='/project',
            target_location='/target',
            steps=steps,
            estimated_complexity='high',
            notes=['Check notebooks']
        )

        md = plan.to_markdown()

        assert '# Migration Plan' in md
        assert '/project' in md
        assert '/target' in md
        assert 'high' in md
        assert 'Setup' in md
        assert 'Verify' in md
        assert 'Notes' in md
        assert 'Check notebooks' in md

    def test_completion_percentage_empty(self):
        """Test completion percentage with no steps."""
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst'
        )

        assert plan.completion_percentage == 0.0

    def test_completion_percentage_none_completed(self):
        """Test completion percentage with no steps completed."""
        steps = [
            MigrationStep(1, 'setup', 'Step 1'),
            MigrationStep(2, 'copy', 'Step 2'),
        ]
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst',
            steps=steps
        )

        assert plan.completion_percentage == 0.0

    def test_completion_percentage_all_completed(self):
        """Test completion percentage with all steps completed."""
        steps = [
            MigrationStep(1, 'setup', 'Step 1', completed=True),
            MigrationStep(2, 'copy', 'Step 2', completed=True),
        ]
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst',
            steps=steps
        )

        assert plan.completion_percentage == 100.0

    def test_completion_percentage_partial(self):
        """Test completion percentage with partial completion."""
        steps = [
            MigrationStep(1, 'setup', 'Step 1', completed=True),
            MigrationStep(2, 'copy', 'Step 2', completed=False),
            MigrationStep(3, 'verify', 'Step 3', completed=True),
            MigrationStep(4, 'generate', 'Step 4', completed=False),
        ]
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst',
            steps=steps
        )

        assert plan.completion_percentage == 50.0


# ============================================================
# MIGRATION PLANNER TESTS
# ============================================================

class TestMigrationPlanner:
    """Tests for the MigrationPlanner class."""

    def test_creates_planner(self, temp_dir):
        """Test creating a migration planner."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        assert planner.analysis is analysis
        assert planner.mapping is mapping

    def test_generate_plan_creates_plan(self, temp_dir):
        """Test that generate_plan creates a plan."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/path/to/target')

        assert isinstance(plan, MigrationPlan)
        assert plan.source_project == str(temp_dir)
        assert plan.target_location == '/path/to/target'

    def test_plan_has_setup_steps(self, temp_dir):
        """Test that plan includes setup steps."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        setup_steps = [s for s in plan.steps if s.category == 'setup']
        assert len(setup_steps) > 0

    def test_plan_has_verify_steps(self, temp_dir):
        """Test that plan includes verification steps."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        verify_steps = [s for s in plan.steps if s.category == 'verify']
        assert len(verify_steps) > 0

    def test_plan_has_generate_steps(self, temp_dir):
        """Test that plan includes generate steps."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        generate_steps = [s for s in plan.steps if s.category == 'generate']
        assert len(generate_steps) > 0

    def test_copy_rules_become_copy_steps(self, temp_dir):
        """Test that copy rules become copy steps."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        rule = MappingRule('data/*', 'data_raw/', 'copy', 'Data files')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            rules=[rule]
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        copy_steps = [s for s in plan.steps if s.category == 'copy']
        assert len(copy_steps) >= 1
        assert any('data' in s.action.lower() for s in copy_steps)

    def test_merge_rules_become_transform_steps(self, temp_dir):
        """Test that merge rules become transform steps."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        rule = MappingRule('loader.py', 'src/stages/s00_ingest.py', 'merge')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            rules=[rule]
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        transform_steps = [s for s in plan.steps if s.category == 'transform']
        assert len(transform_steps) >= 1

    def test_mapping_warnings_become_notes(self, temp_dir):
        """Test that mapping warnings become plan notes."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            warnings=['Unmapped module: config.py']
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        assert 'Unmapped module: config.py' in plan.notes

    def test_complexity_low_for_small_project(self, temp_dir):
        """Test low complexity for small project."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=[],
            has_notebooks=False
        )
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            warnings=[]
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        assert plan.estimated_complexity == 'low'

    def test_complexity_high_for_large_project(self, temp_dir):
        """Test high complexity for large project with warnings."""
        # Create many modules
        modules = []
        for i in range(25):
            path = temp_dir / f'mod{i}.py'
            path.write_text(f'# Module {i}')
            modules.append(ModuleInfo(
                path=path,
                name=f'mod{i}',
                docstring='',
                imports=[],
                functions=[],
                classes=[]
            ))

        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            modules=modules,
            has_notebooks=True
        )
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            warnings=['w1', 'w2', 'w3', 'w4', 'w5', 'w6']
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        assert plan.estimated_complexity == 'high'

    def test_steps_are_ordered(self, temp_dir):
        """Test that steps have sequential order."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )
        planner = MigrationPlanner(analysis, mapping)

        plan = planner.generate_plan('/target')

        orders = [s.order for s in plan.steps]
        assert orders == sorted(orders)
        assert orders[0] == 1


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================

class TestGenerateMigrationPlan:
    """Tests for the generate_migration_plan convenience function."""

    def test_generates_plan(self, temp_dir):
        """Test convenience function."""
        analysis = ProjectAnalysis(root_path=temp_dir, project_name='test')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR'
        )

        plan = generate_migration_plan(analysis, mapping, '/target')

        assert isinstance(plan, MigrationPlan)
        assert plan.target_location == '/target'

    def test_generates_complete_plan(self, temp_dir):
        """Test that convenience function generates complete plan."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_docs=True,
            has_tests=True
        )
        rule = MappingRule('src/*', 'src/stages/', 'copy')
        mapping = StructureMapping(
            source_project=str(temp_dir),
            target_template='CENTAUR',
            rules=[rule]
        )

        plan = generate_migration_plan(analysis, mapping, '/target')

        # Should have multiple step categories
        categories = set(s.category for s in plan.steps)
        assert 'setup' in categories
        assert 'copy' in categories
        assert 'generate' in categories
        assert 'verify' in categories

