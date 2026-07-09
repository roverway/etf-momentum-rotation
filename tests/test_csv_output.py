"""Tests for backtest_engine CSV output — save_results & compute_summary.

TDD: tests written before implementation.
"""

from __future__ import annotations

import csv
import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from backtest_engine import run_backtest, save_results, compute_summary
from config import BacktestConfig
from portfolio import Portfolio


# =====================================================================================
# Helpers  (mirror test_backtest_engine.py)
# =====================================================================================

def _make_dates(start: str = '2024-01-02', periods: int = 25) -> list[date]:
    """Generate a list of consecutive business dates."""
    return pd.bdate_range(start=start, periods=periods, freq='B').date.tolist()


def _make_etf_df(dates: list[date], prices: list[float]) -> pd.DataFrame:
    """Build a mock ETF DataFrame with ``date`` and ``close`` columns."""
    return pd.DataFrame({'date': dates, 'close': prices})


# =====================================================================================
# Tests: save_results
# =====================================================================================

class TestSaveResults:
    """save_results writes CSV files correctly."""

    def test_creates_net_worth_csv(self, tmp_path):
        """net_worth.csv is created in the output directory."""
        snapshots = [{
            'date': date(2024, 1, 2),
            'portfolio_value': 1_000_000.0,
            'cash': 1_000_000.0,
            'position_code': '',
            'shares': 0,
            'price': 0.0,
        }]
        save_results(Portfolio(1_000_000), snapshots, [], output_dir=str(tmp_path))
        assert os.path.isfile(os.path.join(tmp_path, 'net_worth.csv'))

    def test_creates_trades_csv(self, tmp_path):
        """trades.csv is created in the output directory."""
        trades = [{
            'date': date(2024, 1, 2),
            'action': 'BUY',
            'code': 'ETF_A',
            'shares': 100,
            'price': 10.0,
            'value': 1000.0,
        }]
        save_results(Portfolio(1_000_000), [], trades, output_dir=str(tmp_path))
        assert os.path.isfile(os.path.join(tmp_path, 'trades.csv'))

    def test_net_worth_columns(self, tmp_path):
        """net_worth.csv has correct column names."""
        snapshots = [{
            'date': date(2024, 1, 2),
            'portfolio_value': 1_000_000.0,
            'cash': 1_000_000.0,
            'position_code': '',
            'shares': 0,
            'price': 0.0,
        }]
        save_results(Portfolio(1_000_000), snapshots, [], output_dir=str(tmp_path))
        with open(os.path.join(tmp_path, 'net_worth.csv'), newline='') as f:
            headers = next(csv.reader(f))
        assert headers == [
            'date', 'portfolio_value', 'cash', 'position_code', 'shares', 'price',
        ]

    def test_trades_columns(self, tmp_path):
        """trades.csv has correct column names."""
        trades = [{
            'date': date(2024, 1, 2),
            'action': 'BUY',
            'code': 'ETF_A',
            'shares': 100,
            'price': 10.0,
            'value': 1000.0,
        }]
        save_results(Portfolio(1_000_000), [], trades, output_dir=str(tmp_path))
        with open(os.path.join(tmp_path, 'trades.csv'), newline='') as f:
            headers = next(csv.reader(f))
        assert headers == ['date', 'action', 'code', 'shares', 'price', 'value']

    def test_creates_directory_recursively(self, tmp_path):
        """Non-existent nested directory is created automatically."""
        output_dir = os.path.join(tmp_path, 'a', 'b', 'c')
        save_results(Portfolio(1_000_000), [], [], output_dir=output_dir)
        assert os.path.isdir(output_dir)

    def test_multiple_snapshots_written(self, tmp_path):
        """All snapshot rows are written to net_worth.csv."""
        snapshots = [
            {
                'date': date(2024, 1, 2),
                'portfolio_value': 1_000_000.0,
                'cash': 1_000_000.0,
                'position_code': '',
                'shares': 0,
                'price': 0.0,
            },
            {
                'date': date(2024, 1, 3),
                'portfolio_value': 1_010_000.0,
                'cash': 500_000.0,
                'position_code': 'ETF_A',
                'shares': 50_000,
                'price': 10.2,
            },
        ]
        save_results(Portfolio(1_000_000), snapshots, [], output_dir=str(tmp_path))
        with open(os.path.join(tmp_path, 'net_worth.csv'), newline='') as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 data rows

    def test_multiple_trades_written(self, tmp_path):
        """All trade rows are written to trades.csv."""
        trades = [
            {
                'date': date(2024, 1, 2),
                'action': 'BUY',
                'code': 'ETF_A',
                'shares': 50_000,
                'price': 10.0,
                'value': 500_000.0,
            },
            {
                'date': date(2024, 1, 10),
                'action': 'SELL',
                'code': 'ETF_A',
                'shares': 50_000,
                'price': 10.5,
                'value': 525_000.0,
            },
        ]
        save_results(Portfolio(1_000_000), [], trades, output_dir=str(tmp_path))
        with open(os.path.join(tmp_path, 'trades.csv'), newline='') as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 trade rows


# =====================================================================================
# Tests: compute_summary
# =====================================================================================

class TestComputeSummary:
    """compute_summary returns correct metrics."""

    def test_returns_expected_keys(self):
        """Summary dict contains all expected keys."""
        snapshots = [
            {'date': date(2024, 1, 2), 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
            {'date': date(2024, 1, 31), 'portfolio_value': 1_050_000.0,
             'cash': 50_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
        ]
        summary = compute_summary(Portfolio(1_000_000), snapshots)
        assert 'total_return_pct' in summary
        assert 'max_drawdown_pct' in summary
        assert 'final_value' in summary
        assert 'total_return' in summary

    def test_positive_return(self):
        """Total return percentage is correct for a profitable backtest."""
        snapshots = [
            {'date': date(2024, 1, 2), 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
            {'date': date(2024, 1, 31), 'portfolio_value': 1_100_000.0,
             'cash': 100_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
        ]
        summary = compute_summary(Portfolio(1_000_000), snapshots)
        assert summary['total_return_pct'] == 10.0

    def test_negative_return(self):
        """Total return percentage is correct for a losing backtest."""
        snapshots = [
            {'date': date(2024, 1, 2), 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
            {'date': date(2024, 1, 31), 'portfolio_value': 900_000.0,
             'cash': 900_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
        ]
        summary = compute_summary(Portfolio(1_000_000), snapshots)
        assert summary['total_return_pct'] == -10.0

    def test_max_drawdown(self):
        """Max drawdown is calculated correctly from the snapshot series."""
        snapshots = [
            {'date': date(2024, 1, 2), 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
            {'date': date(2024, 1, 5), 'portfolio_value': 1_100_000.0,
             'cash': 100_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
            {'date': date(2024, 1, 10), 'portfolio_value': 950_000.0,
             'cash': 50_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 18.0},
            {'date': date(2024, 1, 15), 'portfolio_value': 1_200_000.0,
             'cash': 200_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
            {'date': date(2024, 1, 20), 'portfolio_value': 1_000_000.0,
             'cash': 0.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
            {'date': date(2024, 1, 31), 'portfolio_value': 1_050_000.0,
             'cash': 50_000.0, 'position_code': 'ETF_A', 'shares': 50_000, 'price': 20.0},
        ]
        summary = compute_summary(Portfolio(1_000_000), snapshots)
        # Peak = 1,200,000 (snapshot 4); trough after peak = 1,000,000 (snapshot 5)
        # Max DD = (1,200,000 - 1,000,000) / 1,200,000 * 100 = 16.666...%
        assert summary['max_drawdown_pct'] == pytest.approx(16.67, abs=0.01)

    def test_empty_snapshots_returns_empty_dict(self):
        """Empty snapshots list → empty dict."""
        summary = compute_summary(Portfolio(1_000_000), [])
        assert summary == {}

    def test_single_snapshot_zero_return_zero_drawdown(self):
        """Single snapshot → 0 % return and 0 % drawdown."""
        snapshots = [
            {'date': date(2024, 1, 2), 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': '', 'shares': 0, 'price': 0.0},
        ]
        summary = compute_summary(Portfolio(1_000_000), snapshots)
        assert summary['total_return_pct'] == 0.0
        assert summary['max_drawdown_pct'] == 0.0


# =====================================================================================
# Tests: run_backtest integration (CSV output)
# =====================================================================================

class TestRunBacktestCSVOutput:
    """run_backtest creates CSV output when output_dir is provided."""

    @patch('trading_calendar.load_trading_calendar')
    def test_integration_creates_csv_files(self, mock_cal, tmp_path):
        """Full backtest run with output_dir creates both CSV files."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        etf_data = {
            'ETF_A': _make_etf_df(dates, [10.0 + i * 0.5 for i in range(25)]),
            'ETF_B': _make_etf_df(dates, [10.0 - i * 0.1 for i in range(25)]),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data, output_dir=str(tmp_path))

        portfolio = result['portfolio']
        net_worth_path = os.path.join(tmp_path, 'net_worth.csv')
        trades_path = os.path.join(tmp_path, 'trades.csv')
        assert os.path.isfile(net_worth_path), 'net_worth.csv should exist'
        assert os.path.isfile(trades_path), 'trades.csv should exist'

        # Portfolio should have traded — holding the rising ETF
        assert 'ETF_A' in portfolio.positions
        assert portfolio.positions['ETF_A'].quantity > 0

    @patch('trading_calendar.load_trading_calendar')
    def test_integration_trade_log_length(self, mock_cal, tmp_path):
        """Trade log length matches expected number of rebalances."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A up first, then ETF_B up → triggers a switch
        a_prices = [10.0 + i * 0.5 for i in range(12)] + [10.0 - i * 0.3 for i in range(13)]
        b_prices = [10.0 - i * 0.2 for i in range(12)] + [10.0 + i * 0.4 for i in range(13)]

        assert len(a_prices) == 25
        assert len(b_prices) == 25

        etf_data = {
            'ETF_A': _make_etf_df(dates, a_prices),
            'ETF_B': _make_etf_df(dates, b_prices),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            run_backtest(config, etf_data=etf_data, output_dir=str(tmp_path))

        # Read trade log CSV to verify it has entries
        with open(os.path.join(tmp_path, 'trades.csv'), newline='') as f:
            rows = list(csv.reader(f))
        # header + at least 2 trades (initial BUY + SWITCH)
        assert len(rows) >= 3, f'Expected ≥3 rows (header + trades), got {len(rows)}'
