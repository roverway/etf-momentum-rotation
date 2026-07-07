"""Tests for backtest_engine module — run_backtest function.

TDD: tests written before implementation.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from config import BacktestConfig
from backtest_engine import run_backtest


# =====================================================================================
# Helpers
# =====================================================================================

def _make_dates(start: str = '2024-01-02', periods: int = 25) -> list[date]:
    """Generate a list of consecutive business dates."""
    return pd.bdate_range(start=start, periods=periods, freq='B').date.tolist()


def _make_etf_df(dates: list[date], prices: list[float]) -> pd.DataFrame:
    """Build a mock ETF DataFrame with ``date`` and ``close`` columns."""
    return pd.DataFrame({'date': dates, 'close': prices})


# =====================================================================================
# Tests: run_backtest
# =====================================================================================

class TestRunBacktest:
    """Core backtest engine — verifies the trading loop logic."""

    @patch('trading_calendar.load_trading_calendar')
    def test_short_backtest_runs_normally(self, mock_cal):
        """Basic flow: backtest runs and returns a Portfolio."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        etf_data = {
            'ETF_A': _make_etf_df(dates,
                                   [10.0 + i * 0.5 for i in range(25)]),   # ↑ rising
            'ETF_B': _make_etf_df(dates,
                                   [10.0 - i * 0.1 for i in range(25)]),   # ↓ falling
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # Should have traded — cash decreased
        assert portfolio.cash < config.initial_cash
        # Should hold the rising ETF
        assert 'ETF_A' in portfolio.positions
        assert portfolio.positions['ETF_A'].quantity > 0

    @patch('trading_calendar.load_trading_calendar')
    def test_first_eligible_day_buys_best_etf(self, mock_cal):
        """On the first day with enough data, the best ETF is bought."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A goes up, ETF_B goes down
        etf_data = {
            'ETF_A': _make_etf_df(dates,
                                   [10.0 + i * 0.3 for i in range(25)]),
            'ETF_B': _make_etf_df(dates,
                                   [10.0 - i * 0.2 for i in range(25)]),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        assert 'ETF_A' in portfolio.positions
        assert portfolio.positions['ETF_A'].quantity > 0
        # Only ETF_A should be held
        assert len(portfolio.positions) == 1

    @patch('trading_calendar.load_trading_calendar')
    def test_hold_when_signal_unchanged(self, mock_cal):
        """Signal same as current position → no additional trades."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A always outperforms ETF_B → signal never changes
        etf_data = {
            'ETF_A': _make_etf_df(dates,
                                   [10.0 + i * 0.5 for i in range(25)]),
            'ETF_B': _make_etf_df(dates,
                                   [10.0 + i * 0.1 for i in range(25)]),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # Only ETF_A should be held
        assert len(portfolio.positions) == 1
        assert 'ETF_A' in portfolio.positions
        # Quantity shouldn't change after initial buy: we bought once, held
        # (no sell hence no second buy)
        expected_qty = int(1_000_000 // 11.0)   # close on dates[2] = 10.0 + 2*0.5
        assert portfolio.positions['ETF_A'].quantity == expected_qty

    @patch('trading_calendar.load_trading_calendar')
    def test_signal_change_triggers_rebalance(self, mock_cal):
        """Signal from ETF_A → ETF_B: sell A, buy B."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A: up first 5 days, then down
        # ETF_B: down first 5 days, then up
        a_prices = ([10.0, 10.3, 10.6, 10.9, 11.2, 11.5] +
                    [11.3, 11.1, 10.9, 10.7, 10.5, 10.3,
                     10.1, 9.9, 9.7, 9.5, 9.3, 9.1,
                     8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7])
        b_prices = ([10.0, 9.7, 9.4, 9.1, 8.8, 8.5] +
                    [8.7, 8.9, 9.1, 9.3, 9.5, 9.7,
                     9.9, 10.1, 10.3, 10.5, 10.7, 10.9,
                     11.1, 11.3, 11.5, 11.7, 11.9, 12.1, 12.3])

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

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # By the end, ETF_B should be the better performer
        # We should have switched from ETF_A to ETF_B at some point
        assert 'ETF_B' in portfolio.positions
        # Should have sold A (can't hold both with this rebalancing logic)
        assert 'ETF_A' not in portfolio.positions

    @patch('trading_calendar.load_trading_calendar')
    def test_all_negative_sells_to_cash(self, mock_cal):
        """All ETFs down → signal is None → sell to cash (empty positions)."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A: up first 6 days (to trigger initial buy), then down
        # ETF_B: always down (worse than A initially)
        a_prices = ([10.0, 10.3, 10.6, 10.9, 11.2, 11.5] +
                    [11.3, 11.1, 10.9, 10.7, 10.5, 10.3,
                     10.1, 9.9, 9.7, 9.5, 9.3, 9.1,
                     8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7])
        b_prices = [10.0 - i * 0.2 for i in range(25)]   # always down

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

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # Position should be sold (all ETFs down → None signal)
        assert len(portfolio.positions) == 0

    @patch('trading_calendar.load_trading_calendar')
    def test_skip_insufficient_data(self, mock_cal):
        """Less data than CHECK_RANGE → no trades occur."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        etf_data = {
            'ETF_A': _make_etf_df(dates[:2], [10.0, 10.5]),   # only 2 rows
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        # CHECK_RANGE is 22 (default) — data is only 2 rows
        portfolio = run_backtest(config, etf_data=etf_data)

        # No trades should happen
        assert len(portfolio.positions) == 0
        assert portfolio.cash == config.initial_cash

    @patch('trading_calendar.load_trading_calendar')
    def test_empty_etf_data_no_trades(self, mock_cal):
        """Empty etf_data dict → no trades, cash unchanged."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data={})

        assert len(portfolio.positions) == 0
        assert portfolio.cash == config.initial_cash

    @patch('trading_calendar.load_trading_calendar')
    def test_config_date_range_filters_calendar(self, mock_cal):
        """Only dates within [start_date, end_date] are processed."""
        all_dates = _make_dates(periods=40)  # Jan 2 ~ Feb 26
        mock_cal.return_value = all_dates

        etf_data = {
            'ETF_A': _make_etf_df(all_dates,
                                   [10.0 + i * 0.3 for i in range(40)]),
            'ETF_B': _make_etf_df(all_dates,
                                   [10.0 - i * 0.1 for i in range(40)]),
        }
        # Narrow window later in the period
        config = BacktestConfig(
            start_date='2024-01-22', end_date='2024-01-31',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # Should have traded (at least one position)
        assert len(portfolio.positions) > 0

    @patch('trading_calendar.load_trading_calendar')
    def test_no_trades_outside_date_range(self, mock_cal):
        """Dates outside config range are never processed."""
        all_dates = _make_dates(start='2023-01-01', periods=500)
        mock_cal.return_value = all_dates

        etf_data = {
            'ETF_A': _make_etf_df(all_dates[:25],
                                   [10.0 + i * 0.5 for i in range(25)]),
        }
        # Range before any data exists
        config = BacktestConfig(
            start_date='2010-01-01', end_date='2010-12-31',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3):
            portfolio = run_backtest(config, etf_data=etf_data)

        # No data matches this range → no trades
        assert len(portfolio.positions) == 0
        assert portfolio.cash == config.initial_cash
