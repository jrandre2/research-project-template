#!/usr/bin/env python3
"""
Tests for src/agents/project_analyzer.py

Tests cover:
- FileInfo dataclass
- ModuleInfo dataclass
- DirectoryInfo dataclass
- ProjectAnalysis dataclass
"""
from __future__ import annotations

import pytest
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.project_analyzer import (
    FileInfo,
    ModuleInfo,
    DirectoryInfo,
    ProjectAnalysis,
)


# ============================================================
# FILE INFO TESTS
# ============================================================

class TestFileInfo:
    """Tests for the FileInfo dataclass."""

    def test_creates_file_info(self, temp_dir):
        """Test creating file info."""
        info = FileInfo(
            path=temp_dir / 'test.py',
            relative_path='test.py',
            size_bytes=100,
            extension='.py',
            category='code'
        )

        assert info.extension == '.py'
        assert info.category == 'code'
        assert info.size_bytes == 100

    def test_to_dict(self, temp_dir):
        """Test conversion to dictionary."""
        info = FileInfo(
            path=temp_dir / 'test.py',
            relative_path='test.py',
            size_bytes=100,
            extension='.py',
            category='code'
        )

        d = info.to_dict()

        assert 'path' in d
        assert 'extension' in d
        assert 'category' in d
        assert d['extension'] == '.py'


# ============================================================
# MODULE INFO TESTS
# ============================================================

class TestModuleInfo:
    """Tests for the ModuleInfo dataclass."""

    def test_creates_module_info(self, temp_dir):
        """Test creating module info."""
        info = ModuleInfo(
            path=temp_dir / 'module.py',
            name='module',
            docstring='A test module',
            imports=['os', 'sys'],
            functions=['foo', 'bar'],
            classes=['MyClass']
        )

        assert info.name == 'module'
        assert len(info.imports) == 2
        assert len(info.functions) == 2

    def test_to_dict(self, temp_dir):
        """Test conversion to dictionary."""
        info = ModuleInfo(
            path=temp_dir / 'module.py',
            name='module',
            docstring='Doc',
            imports=['os'],
            functions=['foo'],
            classes=[]
        )

        d = info.to_dict()

        assert d['name'] == 'module'
        assert d['docstring'] == 'Doc'
        assert d['imports'] == ['os']


# ============================================================
# DIRECTORY INFO TESTS
# ============================================================

class TestDirectoryInfo:
    """Tests for the DirectoryInfo dataclass."""

    def test_creates_directory_info(self, temp_dir):
        """Test creating directory info."""
        info = DirectoryInfo(
            path=temp_dir,
            name='test_dir',
            purpose='test',
            file_count=5,
            subdirs=['sub1', 'sub2']
        )

        assert info.name == 'test_dir'
        assert info.file_count == 5
        assert len(info.subdirs) == 2

    def test_to_dict(self, temp_dir):
        """Test conversion to dictionary."""
        info = DirectoryInfo(
            path=temp_dir,
            name='test_dir',
            purpose='data',
            file_count=10,
            subdirs=[]
        )

        d = info.to_dict()

        assert d['name'] == 'test_dir'
        assert d['purpose'] == 'data'
        assert d['file_count'] == 10


# ============================================================
# PROJECT ANALYSIS TESTS
# ============================================================

class TestProjectAnalysis:
    """Tests for the ProjectAnalysis dataclass."""

    def test_creates_analysis(self, temp_dir):
        """Test creating project analysis."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test_project'
        )

        assert analysis.project_name == 'test_project'
        assert analysis.directories == []
        assert analysis.has_pipeline is False

    def test_default_values(self, temp_dir):
        """Test default values are set correctly."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test'
        )

        assert analysis.has_tests is False
        assert analysis.has_docs is False
        assert analysis.has_notebooks is False
        assert analysis.has_manuscript is False
        assert analysis.data_files == []

    def test_to_dict(self, temp_dir):
        """Test conversion to dictionary."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_pipeline=True,
            pipeline_stages=['s00', 's01']
        )

        d = analysis.to_dict()

        assert d['project_name'] == 'test'
        assert d['has_pipeline'] is True
        assert d['pipeline_stages'] == ['s00', 's01']

    def test_to_json(self, temp_dir):
        """Test JSON serialization."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test'
        )

        json_str = analysis.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['project_name'] == 'test'

    def test_summary(self, temp_dir):
        """Test summary generation."""
        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            has_tests=True,
            has_docs=True
        )

        summary = analysis.summary()

        assert 'test' in summary
        assert 'Tests: Yes' in summary
        assert 'Documentation: Yes' in summary


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestProjectAnalyzerIntegration:
    """Integration tests for project analyzer."""

    def test_analysis_with_files(self, temp_dir):
        """Test analysis with files."""
        # Create some files
        (temp_dir / 'script.py').write_text('# Python script')
        (temp_dir / 'data.csv').write_text('a,b\n1,2')

        files = [
            FileInfo(
                path=temp_dir / 'script.py',
                relative_path='script.py',
                size_bytes=16,
                extension='.py',
                category='code'
            ),
            FileInfo(
                path=temp_dir / 'data.csv',
                relative_path='data.csv',
                size_bytes=8,
                extension='.csv',
                category='data'
            ),
        ]

        analysis = ProjectAnalysis(
            root_path=temp_dir,
            project_name='test',
            files=files
        )

        assert len(analysis.files) == 2
        d = analysis.to_dict()
        assert len(d['files']) == 2
