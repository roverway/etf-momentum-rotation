"""Tests for charts.py — Plotly visualization module.

TDD: tests written before implementation.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest

from charts import (
    _compute_drawdown,
    generate_all,
    plot_allocation,
    plot_drawdown,
    plot_net_worth,
)


# =====================================================================================
# Tests: _compute_drawdown
# =====================================================================================

class TestComputeDrawdown:
    """_compute_drawdown calculates daily drawdown percentages correctly."""

    def test_always_rising_no_drawdown(self):
        """Monotonically increasing series → drawdown is all zero."""
        values = pd.Series([100.0, 110.0, 120.0, 130.0])
        result = _compute_drawdown(values)
        expected = pd.Series([0.0, 0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_peak_then_trough(self):
        """Drawdown is negative after a peak, recovers to zero at new high."""
        values = pd.Series([100.0, 110.0, 105.0, 95.0, 120.0])
        result = _compute_drawdown(values)
        # cummax: [100, 110, 110, 110, 120]
        # dd%:    [0,   0,   (105-110)/110*100=-4.545, (95-110)/110*100=-13.636, 0]
        expected = pd.Series([0.0, 0.0, -4.54545454, -13.63636364, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_flat_series(self):
        """Flat series → zero drawdown."""
        values = pd.Series([100.0, 100.0, 100.0])
        result = _compute_drawdown(values)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_single_element(self):
        """Single element → zero drawdown."""
        values = pd.Series([100.0])
        result = _compute_drawdown(values)
        expected = pd.Series([0.0])
        pd.testing.assert_series_equal(result, expected)


# =====================================================================================
# Tests: plot_net_worth
# =====================================================================================

class TestPlotNetWorth:
    """plot_net_worth creates a valid HTML line chart."""

    @pytest.fixture
    def mock_df(self):
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'portfolio_value': [1_000_000.0, 1_010_000.0, 995_000.0],
            'cash': [1_000_000.0, 500_000.0, 495_000.0],
            'position_code': ['', '513100.XSHG', '513100.XSHG'],
            'shares': [0, 50_000, 50_000],
            'price': [0.0, 10.2, 10.0],
        })

    def test_creates_html_file(self, mock_df, tmp_path):
        """HTML file is created at the specified path."""
        path = os.path.join(str(tmp_path), 'net_worth.html')
        result = plot_net_worth(mock_df, path)
        assert os.path.isfile(path)

    def test_returns_path_string(self, mock_df, tmp_path):
        """Returns the output path string."""
        path = os.path.join(str(tmp_path), 'net_worth.html')
        result = plot_net_worth(mock_df, path)
        assert result == path

    def test_html_starts_with_doctype(self, mock_df, tmp_path):
        """HTML output starts with DOCTYPE declaration."""
        path = os.path.join(str(tmp_path), 'net_worth.html')
        plot_net_worth(mock_df, path)
        with open(path) as f:
            content = f.read()
        assert content.lstrip().startswith('<!DOCTYPE html>')


# =====================================================================================
# Tests: plot_drawdown
# =====================================================================================

class TestPlotDrawdown:
    """plot_drawdown creates a valid HTML drawdown chart."""

    @pytest.fixture
    def mock_df(self):
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'portfolio_value': [1_000_000.0, 1_100_000.0, 1_050_000.0, 950_000.0],
            'cash': [1_000_000.0, 100_000.0, 50_000.0, 50_000.0],
            'position_code': ['', '513100.XSHG', '513100.XSHG', '512890.XSHG'],
            'shares': [0, 50_000, 50_000, 50_000],
            'price': [0.0, 20.0, 20.0, 18.0],
        })

    def test_creates_html_file(self, mock_df, tmp_path):
        """HTML file is created at the specified path."""
        path = os.path.join(str(tmp_path), 'drawdown.html')
        result = plot_drawdown(mock_df, path)
        assert os.path.isfile(path)

    def test_returns_path_string(self, mock_df, tmp_path):
        """Returns the output path string."""
        path = os.path.join(str(tmp_path), 'drawdown.html')
        result = plot_drawdown(mock_df, path)
        assert result == path

    def test_html_starts_with_doctype(self, mock_df, tmp_path):
        """HTML output starts with DOCTYPE declaration."""
        path = os.path.join(str(tmp_path), 'drawdown.html')
        plot_drawdown(mock_df, path)
        with open(path) as f:
            content = f.read()
        assert content.lstrip().startswith('<!DOCTYPE html>')


# =====================================================================================
# Tests: plot_allocation
# =====================================================================================

class TestPlotAllocation:
    """plot_allocation creates a valid HTML stacked-area chart."""

    @pytest.fixture
    def mock_nw(self):
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'portfolio_value': [1_000_000.0, 1_010_000.0, 990_000.0],
            'cash': [500_000.0, 200_000.0, 490_000.0],
            'position_code': ['513100.XSHG', '513100.XSHG', '512890.XSHG'],
            'shares': [50_000, 80_000, 50_000],
            'price': [10.0, 10.125, 10.0],
        })

    @pytest.fixture
    def mock_trades(self):
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-04'],
            'action': ['BUY', 'SELL'],
            'code': ['513100.XSHG', '513100.XSHG'],
            'shares': [50_000, 50_000],
            'price': [10.0, 10.0],
            'value': [500_000.0, 500_000.0],
        })

    def test_creates_html_file(self, mock_nw, mock_trades, tmp_path):
        """HTML file is created at the specified path."""
        path = os.path.join(str(tmp_path), 'allocation.html')
        result = plot_allocation(mock_trades, mock_nw, path)
        assert os.path.isfile(path)

    def test_returns_path_string(self, mock_nw, mock_trades, tmp_path):
        """Returns the output path string."""
        path = os.path.join(str(tmp_path), 'allocation.html')
        result = plot_allocation(mock_trades, mock_nw, path)
        assert result == path

    def test_html_starts_with_doctype(self, mock_nw, mock_trades, tmp_path):
        """HTML output starts with DOCTYPE declaration."""
        path = os.path.join(str(tmp_path), 'allocation.html')
        plot_allocation(mock_trades, mock_nw, path)
        with open(path) as f:
            content = f.read()
        assert content.lstrip().startswith('<!DOCTYPE html>')

    def test_single_code(self, tmp_path):
        """Works with only one position code across all dates."""
        nw = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'portfolio_value': [1_000_000.0, 1_050_000.0],
            'cash': [500_000.0, 550_000.0],
            'position_code': ['513100.XSHG', '513100.XSHG'],
            'shares': [50_000, 50_000],
            'price': [10.0, 10.0],
        })
        trades = pd.DataFrame(columns=['date', 'action', 'code', 'shares', 'price', 'value'])
        path = os.path.join(str(tmp_path), 'alloc_single.html')
        result = plot_allocation(trades, nw, path)
        assert os.path.isfile(path)
        assert result == path

    def test_cash_only_no_positions(self, tmp_path):
        """Works when there are no positions (cash 100%)."""
        nw = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'portfolio_value': [1_000_000.0, 1_000_000.0],
            'cash': [1_000_000.0, 1_000_000.0],
            'position_code': ['', ''],
            'shares': [0, 0],
            'price': [0.0, 0.0],
        })
        trades = pd.DataFrame(columns=['date', 'action', 'code', 'shares', 'price', 'value'])
        path = os.path.join(str(tmp_path), 'alloc_cash_only.html')
        result = plot_allocation(trades, nw, path)
        assert os.path.isfile(path)
        assert result == path


# =====================================================================================
# Tests: generate_all
# =====================================================================================

class TestGenerateAll:
    """generate_all reads CSVs and calls all 3 plot functions."""

    def _mock_net_worth(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'portfolio_value': [1_000_000.0, 1_010_000.0],
            'cash': [1_000_000.0, 500_000.0],
            'position_code': ['', '513100.XSHG'],
            'shares': [0, 50_000],
            'price': [0.0, 10.2],
        })

    def _mock_trades(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': ['2024-01-03'],
            'action': ['BUY'],
            'code': ['513100.XSHG'],
            'shares': [50_000],
            'price': [10.0],
            'value': [500_000.0],
        })

    @patch('pandas.read_csv')
    def test_returns_three_paths(self, mock_read_csv, tmp_path):
        """Returns a list of 3 path strings."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == 3

    @patch('pandas.read_csv')
    def test_creates_three_html_files(self, mock_read_csv, tmp_path):
        """All 3 HTML files are created on disk."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        for p in result:
            assert os.path.isfile(p), f'File not created: {p}'

    @patch('pandas.read_csv')
    def test_returns_paths_in_output_dir(self, mock_read_csv, tmp_path):
        """All returned paths are inside the output directory."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        output_dir = str(tmp_path)
        result = generate_all('nw.csv', 'trades.csv', output_dir)
        for p in result:
            assert os.path.commonpath([output_dir]) == os.path.commonpath([output_dir, p])

    @patch('pandas.read_csv')
    def test_passes_date_and_value_to_plots(self, mock_read_csv, tmp_path):
        """generate_all passes parsed DataFrames to plot functions
        (smoke test: no exception, HTMl files non-empty)."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        for p in result:
            assert os.path.getsize(p) > 0, f'Empty file: {p}'
