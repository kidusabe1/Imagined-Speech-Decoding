"""
Comprehensive tests for script helper functions.

Covers:
- scripts/train_fast.py: load_config, resolve_data_folder, resolve_excel_path
- scripts/benchmark.py: load_subject_predictions, load_global_predictions, process_results
"""

import os
import sys
import tempfile
import importlib.util

import numpy as np
import pandas as pd
import pytest
import yaml


def _import_script(name, path):
    """Import a script file as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
_train_fast = _import_script('train_fast', os.path.join(_scripts_dir, 'train_fast.py'))
_benchmark = _import_script('benchmark', os.path.join(_scripts_dir, 'benchmark.py'))


# ============================================================
# train_fast.py helpers
# ============================================================

load_config = _train_fast.load_config
resolve_data_folder = _train_fast.resolve_data_folder
resolve_excel_path = _train_fast.resolve_excel_path


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_loads_valid_yaml(self, tmp_dir):
        """Loads a valid YAML file and returns dict."""
        cfg = {'model': {'dim': 32}, 'training': {'epochs': 10}}
        path = os.path.join(tmp_dir, 'config.yaml')
        with open(path, 'w') as f:
            yaml.dump(cfg, f)
        result = load_config(path)
        assert result['model']['dim'] == 32
        assert result['training']['epochs'] == 10

    def test_loads_default_config(self):
        """Can load the actual default.yaml from the project."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            assert 'model' in cfg
            assert 'training' in cfg
            assert cfg['model']['n_classes'] == 5

    def test_empty_yaml(self, tmp_dir):
        """Empty YAML file returns None."""
        path = os.path.join(tmp_dir, 'empty.yaml')
        with open(path, 'w') as f:
            f.write('')
        result = load_config(path)
        assert result is None


class TestResolveDataFolder:
    """Tests for resolve_data_folder."""

    def test_finds_existing_folder(self, tmp_dir):
        """Returns path when folder exists."""
        result = resolve_data_folder(tmp_dir)
        assert result == os.path.abspath(tmp_dir)

    def test_raises_on_missing_folder(self, monkeypatch):
        """Raises FileNotFoundError for non-existent paths when no fallback exists."""
        # Patch os.path.exists to always return False so fallback doesn't match
        monkeypatch.setattr(os.path, 'exists', lambda p: False)
        with pytest.raises(FileNotFoundError):
            resolve_data_folder("/definitely/not/a/real/path")


class TestResolveExcelPath:
    """Tests for resolve_excel_path."""

    def test_finds_provided_path(self, tmp_dir):
        """Returns provided path when it exists."""
        excel = os.path.join(tmp_dir, 'labels.xlsx')
        pd.DataFrame({'a': [1]}).to_excel(excel, index=False)
        result = resolve_excel_path(tmp_dir, excel)
        assert result == os.path.abspath(excel)

    def test_falls_back_to_test_set(self, tmp_dir):
        """Falls back to base_folder/Test set/Track3_Answer Sheet_Test.xlsx."""
        test_dir = os.path.join(tmp_dir, 'Test set')
        os.makedirs(test_dir)
        excel = os.path.join(test_dir, 'Track3_Answer Sheet_Test.xlsx')
        pd.DataFrame({'a': [1]}).to_excel(excel, index=False)
        result = resolve_excel_path(tmp_dir, None)
        assert result == excel

    def test_raises_when_nothing_found(self, tmp_dir):
        """Raises FileNotFoundError when no Excel file found."""
        with pytest.raises(FileNotFoundError):
            resolve_excel_path(tmp_dir, None)


# ============================================================
# benchmark.py helpers
# ============================================================

load_subject_predictions = _benchmark.load_subject_predictions
load_global_predictions = _benchmark.load_global_predictions
process_results = _benchmark.process_results


class TestLoadSubjectPredictions:
    """Tests for loading per-subject test predictions."""

    def test_loads_csv(self, tmp_dir):
        """Loads a valid predictions CSV."""
        pred_path = os.path.join(tmp_dir, 'test_predictions.csv')
        pd.DataFrame({'# Predicted': [0, 1, 2], 'True': [0, 1, 1]}).to_csv(pred_path, index=False)
        y_pred, y_true = load_subject_predictions(tmp_dir)
        np.testing.assert_array_equal(y_pred, [0, 1, 2])
        np.testing.assert_array_equal(y_true, [0, 1, 1])

    def test_missing_file_returns_none(self, tmp_dir):
        """Returns (None, None) when file doesn't exist."""
        y_pred, y_true = load_subject_predictions(tmp_dir)
        assert y_pred is None
        assert y_true is None

    def test_fallback_columns(self, tmp_dir):
        """Falls back to iloc when column names don't match."""
        pred_path = os.path.join(tmp_dir, 'test_predictions.csv')
        pd.DataFrame({'col1': [0, 1], 'col2': [1, 0]}).to_csv(pred_path, index=False)
        y_pred, y_true = load_subject_predictions(tmp_dir)
        np.testing.assert_array_equal(y_pred, [0, 1])
        np.testing.assert_array_equal(y_true, [1, 0])


class TestLoadGlobalPredictions:
    """Tests for loading global test predictions."""

    def test_loads_csv(self, tmp_dir):
        """Loads a valid global predictions CSV."""
        pred_path = os.path.join(tmp_dir, 'global_test_predictions.csv')
        pd.DataFrame({'# Predicted': [0, 1, 2, 3], 'True': [0, 1, 2, 3]}).to_csv(pred_path, index=False)
        y_pred, y_true = load_global_predictions(tmp_dir)
        np.testing.assert_array_equal(y_pred, [0, 1, 2, 3])

    def test_missing_file_returns_none(self, tmp_dir):
        """Returns (None, None) when file doesn't exist."""
        y_pred, y_true = load_global_predictions(tmp_dir)
        assert y_pred is None
        assert y_true is None


class TestProcessResults:
    """Tests for results aggregation."""

    def test_missing_model_folder_returns_none(self, tmp_dir):
        """Returns (None, None) when model folder doesn't exist."""
        df, summary = process_results(tmp_dir, "NonExistentModel")
        assert df is None
        assert summary is None

    def test_processes_subject_folders(self, tmp_dir):
        """Correctly aggregates metrics from subject folders."""
        model_dir = os.path.join(tmp_dir, 'FAST')
        os.makedirs(model_dir)

        for sid in [0, 1, 2]:
            sub_dir = os.path.join(model_dir, f'sub-{sid}')
            os.makedirs(sub_dir)
            pred_path = os.path.join(sub_dir, 'test_predictions.csv')
            # Perfect predictions
            preds = np.arange(5)
            pd.DataFrame({'# Predicted': preds, 'True': preds}).to_csv(pred_path, index=False)

        df, summary = process_results(tmp_dir, 'FAST')
        assert df is not None
        assert summary is not None
        assert len(df) == 3
        assert summary['N_subjects'] == 3
        # Perfect predictions -> accuracy = 1.0
        assert summary['Acc_Mean'] == pytest.approx(1.0)

    def test_no_subject_predictions_found(self, tmp_dir):
        """Returns (None, None) when subject folders exist but have no predictions."""
        model_dir = os.path.join(tmp_dir, 'FAST')
        sub_dir = os.path.join(model_dir, 'sub-0')
        os.makedirs(sub_dir)
        # No test_predictions.csv created
        df, summary = process_results(tmp_dir, 'FAST')
        assert df is None
        assert summary is None
