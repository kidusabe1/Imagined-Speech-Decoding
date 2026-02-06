"""
Comprehensive tests for utility functions (src/fast/utils.py).

Covers:
- Terminal color functions: wrapping, ANSI codes
- convert_to_number: int, float, non-numeric strings
- find_available_path: existing paths, missing paths
- now: timestamp format
- random_string: length, character set
- seed_all: reproducibility
- Tick/Tock: context manager timing
"""

import os
import time
import tempfile

import numpy as np
import pytest
import torch

from fast.utils import (
    bold, dim, red, green, yellow, blue, cyan, magenta, white,
    gray, italicized, underline, blink, inverse,
    convert_to_number,
    find_available_path,
    now,
    random_string,
    seed_all,
    Tick,
    Tock,
)


# ============================================================
# Terminal color functions
# ============================================================

class TestColorFunctions:
    """Tests for ANSI color wrapper functions."""

    def test_bold_wraps_string(self):
        result = bold("hello")
        assert "hello" in result
        assert result.startswith('\033[1m')
        assert result.endswith('\033[0m')

    def test_red_wraps_string(self):
        result = red("error")
        assert "error" in result
        assert '\033[91m' in result

    def test_green_wraps_string(self):
        result = green("ok")
        assert '\033[92m' in result

    def test_yellow_wraps_string(self):
        result = yellow("warn")
        assert '\033[93m' in result

    def test_color_functions_accept_non_strings(self):
        """Color functions convert non-string args via str()."""
        assert "42" in bold(42)
        assert "3.14" in red(3.14)
        assert "None" in blue(None)
        assert "True" in green(True)

    def test_all_color_functions_return_strings(self):
        """Every color function returns a string."""
        for fn in [bold, dim, red, green, yellow, blue, cyan,
                   magenta, white, gray, italicized, underline, blink, inverse]:
            result = fn("test")
            assert isinstance(result, str)

    def test_nested_colors(self):
        """Nested color calls don't crash (even if ugly)."""
        result = bold(red("nested"))
        assert "nested" in result


# ============================================================
# convert_to_number
# ============================================================

class TestConvertToNumber:
    """Tests for convert_to_number."""

    def test_integer_string(self):
        assert convert_to_number("42") == 42
        assert isinstance(convert_to_number("42"), int)

    def test_float_string(self):
        assert convert_to_number("3.14") == pytest.approx(3.14)
        assert isinstance(convert_to_number("3.14"), float)

    def test_non_numeric_string(self):
        assert convert_to_number("hello") == "hello"

    def test_zero(self):
        assert convert_to_number("0") == 0
        assert isinstance(convert_to_number("0"), int)

    def test_negative_float(self):
        """Negative numbers: isdigit() returns False, so it tries float()."""
        assert convert_to_number("-1.5") == pytest.approx(-1.5)

    def test_empty_string(self):
        """Empty string returns as-is (not a number)."""
        assert convert_to_number("") == ""


# ============================================================
# find_available_path
# ============================================================

class TestFindAvailablePath:
    """Tests for find_available_path."""

    def test_finds_existing_path(self):
        """Returns the first existing path."""
        with tempfile.TemporaryDirectory() as d:
            result = find_available_path(["/nonexistent", d])
            assert result == d

    def test_returns_first_match(self):
        """Returns the FIRST existing path when multiple exist."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            result = find_available_path([d1, d2])
            assert result == d1

    def test_raises_on_no_match(self):
        """Raises FileNotFoundError if nothing exists."""
        with pytest.raises(FileNotFoundError):
            find_available_path(["/fake/a", "/fake/b"])

    def test_raises_on_empty_list(self):
        """Raises FileNotFoundError on empty list."""
        with pytest.raises(FileNotFoundError):
            find_available_path([])


# ============================================================
# now
# ============================================================

class TestNow:
    """Tests for the now() timestamp function."""

    def test_default_format(self):
        """Default format is YYYY-MM-DD_HH:MM:SS."""
        result = now()
        assert len(result) == 19  # "2025-01-28_14:30:00"
        assert result[4] == '-'
        assert result[10] == '_'

    def test_custom_format(self):
        result = now("%Y")
        assert len(result) == 4
        assert result.isdigit()

    def test_returns_string(self):
        assert isinstance(now(), str)


# ============================================================
# random_string
# ============================================================

class TestRandomString:
    """Tests for random_string."""

    def test_default_length(self):
        assert len(random_string()) == 10

    def test_custom_length(self):
        for l in [1, 5, 20, 100]:
            assert len(random_string(l)) == l

    def test_alphanumeric_only(self):
        """Characters are letters and digits only."""
        result = random_string(1000)
        assert result.isalnum()

    def test_randomness(self):
        """Two calls produce different strings (with very high probability)."""
        a = random_string(20)
        b = random_string(20)
        assert a != b


# ============================================================
# seed_all
# ============================================================

class TestSeedAll:
    """Tests for reproducibility seeding."""

    def test_torch_reproducible(self):
        """Same seed produces same torch random numbers."""
        seed_all(42)
        a = torch.randn(10)
        seed_all(42)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_numpy_reproducible(self):
        """Same seed produces same numpy random numbers."""
        seed_all(42)
        a = np.random.randn(10)
        seed_all(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds produce different numbers."""
        seed_all(42)
        a = torch.randn(100)
        seed_all(99)
        b = torch.randn(100)
        assert not torch.allclose(a, b)

    def test_sets_cudnn_flags(self):
        """cudnn flags are set for determinism."""
        seed_all(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


# ============================================================
# Tick / Tock context managers
# ============================================================

class TestTick:
    """Tests for the Tick timing context manager."""

    def test_measures_time(self):
        """Tick records elapsed time."""
        with Tick("test", silent=True) as t:
            time.sleep(0.05)
        assert t.delta >= 0.04
        assert t.delta < 1.0

    def test_fps_computed(self):
        """FPS is 1/delta."""
        with Tick("test", silent=True) as t:
            time.sleep(0.05)
        assert t.fps == pytest.approx(1.0 / t.delta, rel=0.1)

    def test_silent_mode(self, capsys):
        """Silent mode suppresses output."""
        with Tick("test", silent=True):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""


class TestTock:
    """Tests for the Tock timing context manager."""

    def test_measures_time(self):
        """Tock records elapsed time."""
        with Tock(report_time=False) as t:
            time.sleep(0.05)
        assert t.delta >= 0.04

    def test_named(self):
        """Name prefix is stored."""
        t = Tock(name="step")
        assert t.name == "step:"

    def test_unnamed(self):
        """Default name is empty string."""
        t = Tock()
        assert t.name == ""
