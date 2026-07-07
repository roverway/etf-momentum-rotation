"""Tests for charts.py — Plotly visualization module.

TDD: tests written before implementation.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from charts import (
    _build_report_figure,
    _compute_drawdown,
    _metrics_table,
    generate_all,
    generate_report,
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
        """Monotonically increasing series -> drawdown is all zero."""
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
        """Flat series -> zero drawdown."""
        values = pd.Series([100.0, 100.0, 100.0])
        result = _compute_drawdown(values)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    def test_single_element(self):
        """Single element -> zero drawdown."""
        values = pd.Series([100.0])
        result = _compute_drawdown(values)
        expected = pd.Series([0.0])
        pd.testing.assert_series_equal(result, expected)


# =====================================================================================
# Tests: _metrics_table
# =====================================================================================

class TestMetricsTable:
    """_metrics_table converts a metrics dict into a go.Table."""

    def test_returns_go_table(self):
        """Returns a plotly Table instance."""
        result = _metrics_table({'sharpe_ratio': 1.25})
        assert isinstance(result, go.Table)

    def test_has_header_with_indicator_and_value(self):
        """Header cells contain Chinese labels for indicator and value."""
        result = _metrics_table({'sharpe_ratio': 1.25})
        header_vals = list(result.header.values)
        assert any('指标' in str(v) for v in header_vals)
        assert any('值' in str(v) for v in header_vals)

    def test_includes_metric_name_and_formatted_value(self):
        """Row contains the Chinese label and formatted value."""
        result = _metrics_table({'sharpe_ratio': 1.25})
        names = list(result.cells.values[0])
        vals = list(result.cells.values[1])
        assert '夏普比率' in names
        idx = names.index('夏普比率')
        assert vals[idx] == '1.25'

    def test_strategy_return_shows_plus_sign(self):
        """Positive strategy return shows + sign."""
        result = _metrics_table({'strategy_cumulative_return_pct': 15.32})
        names = list(result.cells.values[0])
        vals = list(result.cells.values[1])
        idx = names.index('策略累计收益率')
        assert vals[idx] == '+15.32%'

    def test_max_drawdown_shows_negative(self):
        """Max drawdown is displayed as negative."""
        result = _metrics_table({'max_drawdown_pct': 12.34})
        names = list(result.cells.values[0])
        vals = list(result.cells.values[1])
        idx = names.index('最大回撤')
        assert vals[idx] == '-12.34%'

    def test_multiple_metrics_all_included(self):
        """All metrics are present in the table."""
        metrics = {
            'start_date': '2019-01-18',
            'end_date': '2024-12-31',
            'total_days': 1500,
            'sharpe_ratio': 1.25,
            'max_drawdown_pct': 12.34,
            'strategy_cumulative_return_pct': 15.32,
            'benchmark_cumulative_return_pct': 5.21,
            'daily_win_rate': 55.21,
            'total_trades': 45,
        }
        result = _metrics_table(metrics)
        names = list(result.cells.values[0])
        assert '开始日期' in names
        assert '夏普比率' in names
        assert '最大回撤' in names
        assert '策略累计收益率' in names
        assert '基准累计收益率' in names
        assert '日胜率' in names
        assert '总交易次数' in names

    def test_empty_metrics_returns_empty_table(self):
        """Empty dict results in a table with no data rows."""
        result = _metrics_table({})
        assert len(result.cells.values[0]) == 0
        assert len(result.cells.values[1]) == 0


# =====================================================================================
# Tests: _build_report_figure
# =====================================================================================

class TestBuildReportFigure:
    """_build_report_figure composes all subplots into one Figure."""

    @pytest.fixture
    def mock_nw(self):
        return pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'portfolio_value': [1_000_000.0, 1_010_000.0, 995_000.0],
            'cash': [1_000_000.0, 500_000.0, 495_000.0],
            'position_code': ['', '513100.XSHG', '513100.XSHG'],
            'shares': [0, 50_000, 50_000],
            'price': [0.0, 10.2, 10.0],
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

    @pytest.fixture
    def sample_metrics(self):
        return {
            'strategy_cumulative_return_pct': 15.32,
            'max_drawdown_pct': 12.34,
            'sharpe_ratio': 1.25,
        }

    def test_returns_go_figure(self, mock_nw, mock_trades, sample_metrics):
        """Returns a plotly Figure instance."""
        result = _build_report_figure(mock_nw, mock_trades, sample_metrics)
        assert isinstance(result, go.Figure)

    def test_has_at_least_four_traces(self, mock_nw, mock_trades, sample_metrics):
        """Figure has at least 4 traces (table + net_worth + drawdown + allocation)."""
        result = _build_report_figure(mock_nw, mock_trades, sample_metrics)
        assert len(result.data) >= 4

    def test_has_title(self, mock_nw, mock_trades, sample_metrics):
        """Figure title contains report name."""
        result = _build_report_figure(mock_nw, mock_trades, sample_metrics)
        assert 'ETF动量轮动策略' in (result.layout.title.text or '')

    def test_has_net_worth_trace(self, mock_nw, mock_trades, sample_metrics):
        """One trace has '策略净值' or 'Portfolio' in its name."""
        result = _build_report_figure(mock_nw, mock_trades, sample_metrics)
        names = [t.name for t in result.data if hasattr(t, 'name') and t.name]
        assert any('策略净值' in n for n in names)

    def test_handles_no_position_data(self, sample_metrics):
        """Works when net_worth has no positions (cash only)."""
        nw = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'portfolio_value': [1_000_000.0, 1_000_000.0],
            'cash': [1_000_000.0, 1_000_000.0],
            'position_code': ['', ''],
            'shares': [0, 0],
            'price': [0.0, 0.0],
        })
        trades = pd.DataFrame(columns=['date', 'action', 'code', 'shares', 'price', 'value'])
        result = _build_report_figure(nw, trades, sample_metrics)
        assert len(result.data) >= 4


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
# Tests: generate_report
# =====================================================================================

class TestGenerateReport:
    """generate_report creates a single-file HTML report."""

    @pytest.fixture
    def mock_nw_csv(self, tmp_path):
        path = os.path.join(str(tmp_path), 'net_worth.csv')
        pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'portfolio_value': [1_000_000.0, 1_010_000.0],
            'cash': [1_000_000.0, 500_000.0],
            'position_code': ['', '513100.XSHG'],
            'shares': [0, 50_000],
            'price': [0.0, 10.2],
        }).to_csv(path, index=False)
        return path

    @pytest.fixture
    def mock_trades_csv(self, tmp_path):
        path = os.path.join(str(tmp_path), 'trades.csv')
        pd.DataFrame({
            'date': ['2024-01-03'],
            'action': ['BUY'],
            'code': ['513100.XSHG'],
            'shares': [50_000],
            'price': [10.0],
            'value': [500_000.0],
        }).to_csv(path, index=False)
        return path

    @pytest.fixture
    def sample_metrics(self):
        return {
            'strategy_cumulative_return_pct': 15.32,
            'max_drawdown_pct': 12.34,
            'sharpe_ratio': 1.25,
        }

    def test_creates_html_file(self, mock_nw_csv, mock_trades_csv, sample_metrics, tmp_path):
        """HTML file is created at the specified output path."""
        output = os.path.join(str(tmp_path), 'report.html')
        result = generate_report(mock_nw_csv, mock_trades_csv, sample_metrics, output)
        assert os.path.isfile(output)
        assert result == output

    def test_html_starts_with_doctype(self, mock_nw_csv, mock_trades_csv, sample_metrics, tmp_path):
        """HTML output starts with DOCTYPE declaration."""
        output = os.path.join(str(tmp_path), 'report.html')
        generate_report(mock_nw_csv, mock_trades_csv, sample_metrics, output)
        with open(output) as f:
            content = f.read()
        assert content.lstrip().startswith('<!DOCTYPE html>')

    def test_contains_metrics_text(self, mock_nw_csv, mock_trades_csv, sample_metrics, tmp_path):
        """HTML contains metric values rendered from the table (ASCII-safe check)."""
        output = os.path.join(str(tmp_path), 'report.html')
        generate_report(mock_nw_csv, mock_trades_csv, sample_metrics, output)
        with open(output) as f:
            content = f.read()
        # Plotly JSON-escapes non-ASCII, but formatted values like +15.32% are ASCII
        assert '+15.32%' in content
        assert '1.25' in content
        # Unicode-escaped metric names (策略累计收益率 -> \\u7b56\\u7565...)
        assert '\\u7b56\\u7565\\u7d2f\\u8ba1\\u6536\\u76ca\\u7387' in content

    def test_contains_report_title(self, mock_nw_csv, mock_trades_csv, sample_metrics, tmp_path):
        """HTML contains the report title (ASCII portion)."""
        output = os.path.join(str(tmp_path), 'report.html')
        generate_report(mock_nw_csv, mock_trades_csv, sample_metrics, output)
        with open(output) as f:
            content = f.read()
        # "ETF" is ASCII, plotly doesn't escape ASCII characters
        assert 'ETF\\u52a8\\u91cf\\u8f6e\\u52a8' in content or 'ETF' in content


# =====================================================================================
# Tests: generate_all (backward compat)
# =====================================================================================

class TestGenerateAll:
    """generate_all reads CSVs and generates a single-file report (backward compat)."""

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
    def test_returns_list_of_paths(self, mock_read_csv, tmp_path):
        """Returns a list of path strings."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == 1

    @patch('pandas.read_csv')
    def test_creates_html_file(self, mock_read_csv, tmp_path):
        """HTML file is created on disk."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        for p in result:
            assert os.path.isfile(p), f'File not created: {p}'

    @patch('pandas.read_csv')
    def test_returns_path_in_output_dir(self, mock_read_csv, tmp_path):
        """Returned path is inside the output directory."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        output_dir = str(tmp_path)
        result = generate_all('nw.csv', 'trades.csv', output_dir)
        for p in result:
            assert os.path.commonpath([output_dir]) == os.path.commonpath([output_dir, p])

    @patch('pandas.read_csv')
    def test_html_non_empty(self, mock_read_csv, tmp_path):
        """Generated HTML file is non-empty."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        for p in result:
            assert os.path.getsize(p) > 0, f'Empty file: {p}'

    @patch('pandas.read_csv')
    def test_html_contains_doctype(self, mock_read_csv, tmp_path):
        """Generated HTML starts with DOCTYPE."""
        mock_read_csv.side_effect = [self._mock_net_worth(), self._mock_trades()]
        result = generate_all('nw.csv', 'trades.csv', str(tmp_path))
        with open(result[0]) as f:
            content = f.read()
        assert content.lstrip().startswith('<!DOCTYPE html>')
