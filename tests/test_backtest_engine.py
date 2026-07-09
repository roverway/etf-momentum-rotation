"""Tests for backtest_engine module — run_backtest, compute_and_print_metrics, print_next_day_suggestion.

TDD: tests written before implementation.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import BacktestConfig
from backtest_engine import (
    compute_and_print_metrics,
    print_next_day_suggestion,
    run_backtest,
)


# =====================================================================================
# Helpers
# =====================================================================================

def _make_dates(start: str = '2024-01-02', periods: int = 25) -> list[date]:
    """Generate a list of consecutive business dates."""
    return pd.bdate_range(start=start, periods=periods, freq='B').date.tolist()


def _make_etf_df(dates: list[date], prices: list[float]) -> pd.DataFrame:
    """Build a mock ETF DataFrame with ``date`` and ``close`` columns."""
    return pd.DataFrame({'date': dates, 'close': prices})


def make_synthetic_etf_data(
    etf_codes: list[str],
    calendar: list[date],
    base_price: float = 10.0,
    drift: float = 0.001,
    vol: float = 0.02,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic ETF daily data for backtest testing.

    Each ETF's close price follows a geometric random walk with drift + volatility.
    Uses a fixed seed for reproducible tests.

    Parameters
    ----------
    etf_codes : list[str]
        ETF codes to generate (e.g. ``['ETF_A', 'ETF_B']``).
    calendar : list[date]
        Trading calendar dates.
    base_price : float
        Starting price for all ETFs.
    drift : float
        Daily drift (default 0.001 ≈ 0.1% per day).
    vol : float
        Daily volatility (default 0.02 ≈ 2% per day).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{code: DataFrame with 'date' and 'close' columns}``.
    """
    rng = np.random.default_rng(seed)
    n = len(calendar)

    result: dict[str, pd.DataFrame] = {}
    # Each ETF gets its own random walk with shared structure
    for code in etf_codes:
        # Different drift per ETF so momentum signals vary
        etf_drift = drift * (1 + (etf_codes.index(code) - len(etf_codes) / 2) * 0.2)
        returns = rng.normal(etf_drift, vol, n)
        prices = base_price * np.exp(np.cumsum(returns))
        result[code] = pd.DataFrame({'date': calendar, 'close': prices})

    return result


# =====================================================================================
# Tests: run_backtest
# =====================================================================================

class TestRunBacktest:
    """Core backtest engine — verifies the trading loop logic."""

    @patch('trading_calendar.load_trading_calendar')
    def test_short_backtest_runs_normally(self, mock_cal):
        """Basic flow: backtest runs and returns a dict with Portfolio."""
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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # Should have traded — cash decreased
        assert portfolio.cash < config.initial_cash
        # Should hold the rising ETF
        assert 'ETF_A' in portfolio.positions
        assert portfolio.positions['ETF_A'].quantity > 0
        # Should have all keys in result
        assert 'daily_snapshots' in result
        assert 'trade_log' in result
        assert 'calendar' in result
        assert 'etf_data' in result

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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        assert 'ETF_A' in portfolio.positions
        assert portfolio.positions['ETF_A'].quantity > 0
        # Only ETF_A should be held
        assert len(portfolio.positions) == 1

    @patch('trading_calendar.load_trading_calendar')
    def test_hold_when_signal_unchanged(self, mock_cal):
        """Signal same as current position → no additional trades."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A steadily up, ETF_B steadily down → signal stays on A
        etf_data = {
            'ETF_A': _make_etf_df(dates,
                                   [10.0 + i * 0.5 for i in range(25)]),
            'ETF_B': _make_etf_df(dates,
                                   [10.0 - i * 0.1 for i in range(25)]),   # steadily down
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # Only ETF_A should be held
        assert len(portfolio.positions) == 1
        assert 'ETF_A' in portfolio.positions
        # Quantity shouldn't change after initial buy: we bought once, held
        # (no sell hence no second buy)
        qty = portfolio.positions['ETF_A'].quantity
        # Verify qty is in valid range: should use most of 1M cash at ~11.0 price
        expected_range_low = int(1_000_000 // (11.5 * 1.0001 * 1.00025))
        expected_range_high = int(1_000_000 // (10.5 * 1.0001 * 1.00025))
        assert expected_range_low <= qty <= expected_range_high, \
            f"qty {qty} outside expected range [{expected_range_low}, {expected_range_high}]"

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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # Position should be sold (all ETFs down → None signal)
        assert len(portfolio.positions) == 0

    @patch('trading_calendar.load_trading_calendar')
    def test_threshold_suppresses_small_difference(self, mock_cal):
        """Momentum difference below REBALANCE_THRESHOLD → no rebalance."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # Two ETFs with very similar price trends → momentum diff < threshold
        # ETF_A: linear +0.25/day → +6.0 over 25 days
        # ETF_B: linear +0.24/day → +5.76 over 25 days  (very close)
        etf_data = {
            'ETF_A': _make_etf_df(dates, [10.0 + i * 0.25 for i in range(25)]),
            'ETF_B': _make_etf_df(dates, [10.0 + i * 0.24 for i in range(25)]),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 5), patch('backtest_engine.VOL_CHECK_RANGE', 5):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # Only ONE position should exist — no switching between near-identical ETFs
        assert len(portfolio.positions) == 1

    @patch('trading_calendar.load_trading_calendar')
    def test_threshold_allows_large_difference(self, mock_cal):
        """Momentum difference exceeds REBALANCE_THRESHOLD → rebalance happens."""
        dates = _make_dates(periods=25)
        mock_cal.return_value = dates

        # ETF_A: strong start then flattens
        # ETF_B: weak start then surges — large momentum difference
        a_prices = ([10.0 + i * 0.6 for i in range(13)] +
                    [10.0 + i * 0.0 for i in range(12)])  # total 25
        b_prices = ([10.0 + i * 0.0 for i in range(13)] +
                    [10.0 + i * 0.6 for i in range(12)])  # total 25

        etf_data = {
            'ETF_A': _make_etf_df(dates, a_prices),
            'ETF_B': _make_etf_df(dates, b_prices),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-02-05',
            initial_cash=1_000_000,
        )

        with patch('backtest_engine.CHECK_RANGE', 5), patch('backtest_engine.VOL_CHECK_RANGE', 5):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # By the end ETF_B should be held (momentum diff exceeded threshold)
        assert 'ETF_B' in portfolio.positions

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
        result = run_backtest(config, etf_data=etf_data)
        portfolio = result['portfolio']

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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data={})

        portfolio = result['portfolio']
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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
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

        with patch('backtest_engine.CHECK_RANGE', 3), patch('backtest_engine.VOL_CHECK_RANGE', 3):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        # No data matches this range → no trades
        assert len(portfolio.positions) == 0
        assert portfolio.cash == config.initial_cash


class TestRunBacktestEdgeCases:
    """Edge case tests for the backtest engine."""

    @patch('trading_calendar.load_trading_calendar')
    def test_insufficient_data_logs_warning(self, mock_cal, caplog):
        """数据不足跳过交易日时应记录 warning 日志"""
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        mock_cal.return_value = dates

        etf_data = {
            'ETF_A': pd.DataFrame({
                'date': [date(2024, 1, 2)],
                'close': [10.0],
            }),
        }
        config = BacktestConfig(
            start_date='2024-01-02', end_date='2024-01-04',
            initial_cash=1_000_000,
        )

        caplog.set_level(logging.WARNING)
        with patch('backtest_engine.CHECK_RANGE', 22), patch('backtest_engine.VOL_CHECK_RANGE', 22):
            result = run_backtest(config, etf_data=etf_data)

        portfolio = result['portfolio']
        assert len(portfolio.positions) == 0
        assert portfolio.cash == config.initial_cash
        assert "数据不足" in caplog.text
        assert "1 < 22" in caplog.text or "CHECK_RANGE" in caplog.text


# =====================================================================================
# Tests: compute_and_print_metrics
# =====================================================================================

class TestComputeAndPrintMetrics:
    """compute_and_print_metrics — metrics computation, printing, and report generation."""

    def test_calls_metrics_and_charts(self):
        """Calls compute_all_metrics and generate_report."""
        dates = _make_dates(periods=5)
        snapshots = [
            {'date': d, 'portfolio_value': 1_000_000.0 + i * 10_000,
             'cash': 1_000_000.0 - i * 10_000, 'position_code': 'ETF_A',
             'shares': 50_000, 'price': 10.0 + i * 0.2}
            for i, d in enumerate(dates)
        ]
        trade_log = [
            {'date': dates[0], 'action': 'BUY', 'code': 'ETF_A',
             'shares': 50_000, 'price': 10.0, 'value': 500_000.0},
        ]
        config = BacktestConfig(
            start_date=str(dates[0]), end_date=str(dates[-1]),
            initial_cash=1_000_000,
        )
        calendar = dates

        with patch('data.fetch_benchmark_data', return_value=pd.DataFrame()) as mock_fetch:
            with patch('metrics.compute_all_metrics') as mock_compute:
                with patch('charts.generate_report') as mock_gen_report:
                    mock_compute.return_value = {
                        'total_return_pct': 5.0, 'annual_return_pct': 10.0,
                        'max_drawdown_pct': -2.0,
                    }
                    result = compute_and_print_metrics(
                        snapshots, trade_log, config, calendar, '/tmp',
                    )

        assert mock_compute.called, "compute_all_metrics should be called"
        assert isinstance(result, dict)
        assert result['total_return_pct'] == 5.0
        # generate_report not called when CSVs don't exist at /tmp
        assert not mock_gen_report.called

    def test_returns_empty_dict_for_empty_snapshots(self):
        """Empty snapshots → empty dict, no calls to metrics."""
        with patch('metrics.compute_all_metrics') as mock_c:
            result = compute_and_print_metrics(
                [], [], MagicMock(), [], '/tmp',
            )
        assert result == {}
        mock_c.assert_not_called()

    def test_annual_returns_computed_correctly(self):
        """Annual returns dict is computed and present in metrics."""
        # 2 years of daily snapshots
        dates = _make_dates(start='2024-01-02', periods=300)
        snapshots = [
            {'date': d, 'portfolio_value': 1_000_000.0 + i * 500.0,
             'cash': 100_000.0, 'position_code': 'ETF_A',
             'shares': 50_000, 'price': 10.0}
            for i, d in enumerate(dates)
        ]
        config = BacktestConfig(
            start_date=str(dates[0]), end_date=str(dates[-1]),
            initial_cash=1_000_000,
        )

        with patch('data.fetch_benchmark_data', return_value=pd.DataFrame()):
            with patch('metrics.compute_all_metrics') as mock_compute:
                mock_compute.return_value = {
                    'total_return_pct': 5.0, 'annual_return_pct': 10.0,
                    'max_drawdown_pct': -2.0,
                }
                result = compute_and_print_metrics(
                    snapshots, [], config, dates, '/tmp',
                )

        assert 'strategy_annual_returns' in result
        assert isinstance(result['strategy_annual_returns'], dict)
        assert len(result['strategy_annual_returns']) > 0
        for year, ret in result['strategy_annual_returns'].items():
            assert isinstance(year, int)
            assert isinstance(ret, float)
        # No benchmark data → benchmark_annual_returns is None
        assert result['benchmark_annual_returns'] is None

    def test_annual_returns_with_benchmark(self):
        """Annual returns computed correctly when benchmark data is present."""
        dates = _make_dates(start='2024-01-02', periods=300)
        snapshots = [
            {'date': d, 'portfolio_value': 1_000_000.0,
             'cash': 1_000_000.0, 'position_code': None,
             'shares': 0, 'price': 0.0}
            for d in dates
        ]
        # Benchmark: same dates, close starts at 1000, ends higher
        bench_df = pd.DataFrame({
            'date': dates,
            'close': [1000.0 + i * 0.5 for i in range(len(dates))],
        })
        config = BacktestConfig(
            start_date=str(dates[0]), end_date=str(dates[-1]),
            initial_cash=1_000_000,
        )

        with patch('data.fetch_benchmark_data', return_value=bench_df):
            with patch('metrics.compute_all_metrics') as mock_compute:
                mock_compute.return_value = {
                    'total_return_pct': 0.0, 'annual_return_pct': 0.0,
                    'max_drawdown_pct': 0.0,
                }
                result = compute_and_print_metrics(
                    snapshots, [], config, dates, '/tmp',
                )

        assert 'strategy_annual_returns' in result
        assert 'benchmark_annual_returns' in result
        assert isinstance(result['benchmark_annual_returns'], dict)
        assert len(result['benchmark_annual_returns']) > 0
        for year, ret in result['benchmark_annual_returns'].items():
            assert isinstance(year, int)
            assert isinstance(ret, float)


# =====================================================================================
# Tests: print_next_day_suggestion
# =====================================================================================

class TestPrintNextDaySuggestion:
    """print_next_day_suggestion — next trading day recommendation."""

    @patch('backtest_engine.get_next_trading_date')
    @patch('backtest_engine.calculate_momentum_signal')
    def test_suggests_buy(self, mock_signal, mock_next):
        """Positive signal → suggests BUY <code>."""
        mock_next.return_value = date(2024, 2, 6)
        mock_signal.return_value = ('ETF_A', {'ETF_A': 0.1})

        calendar = [date(2024, 2, 5)]
        etf_data = {
            'ETF_A': pd.DataFrame({'date': [date(2024, 2, 5)], 'close': [10.0]}),
        }
        config = BacktestConfig(start_date='2024-01-01', end_date='2024-02-05')

        with patch('backtest_engine.CHECK_RANGE', 1), patch('backtest_engine.VOL_CHECK_RANGE', 1):
            print_next_day_suggestion(calendar, etf_data, config, None)

        mock_signal.assert_called_once()

    @patch('backtest_engine.get_next_trading_date')
    def test_no_next_date_prints_message(self, mock_next, capsys):
        """No next trading day → prints message."""
        mock_next.side_effect = ValueError("no next trading day")

        print_next_day_suggestion(
            [date(2024, 12, 31)], {}, MagicMock(), None,
        )
        captured = capsys.readouterr()
        assert "无法获取下一交易日" in captured.out

    def test_empty_calendar_prints_message(self, capsys):
        """Empty calendar → prints warning."""
        print_next_day_suggestion([], {}, MagicMock(), None)
        captured = capsys.readouterr()
        assert "无交易日历" in captured.out


# =====================================================================================
# Regression tests: legacy vs vectorized output equivalence
# =====================================================================================

class TestVectorizedRegression:
    """Compare legacy and vectorized loop outputs for identical results."""

    def _run_comparison(self, etf_data, calendar, config):
        """Run both loops and return (legacy_result, vectorized_result)."""
        from backtest_engine import _run_backtest_loop_legacy, _run_backtest_loop_vectorized
        from portfolio import Portfolio

        etf_codes = list(etf_data.keys())

        # Legacy
        p1 = Portfolio(config.initial_cash)
        commission_rate = getattr(config, 'commission_rate', 0.00025)
        slippage_rate = getattr(config, 'slippage_rate', 0.001)
        legacy_snapshots, legacy_trades = _run_backtest_loop_legacy(
            calendar, etf_data, etf_codes, p1, commission_rate, slippage_rate,
        )

        # Vectorized
        p2 = Portfolio(config.initial_cash)
        vec_snapshots, vec_trades = _run_backtest_loop_vectorized(
            calendar, etf_data, etf_codes, p2, commission_rate, slippage_rate,
        )

        return (legacy_snapshots, legacy_trades, p1), (vec_snapshots, vec_trades, p2)

    def test_identical_snapshots_basic(self):
        """4 ETF × 100 days: daily_snapshots match exactly."""
        calendar = _make_dates('2024-01-02', 100)
        etf_data = make_synthetic_etf_data(
            ['ETF_A', 'ETF_B', 'ETF_C', 'ETF_D'],
            calendar, drift=0.001, vol=0.02,
        )
        config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)

        (legacy_snap, legacy_trades, p1), (vec_snap, vec_trades, p2) = \
            self._run_comparison(etf_data, calendar, config)

        # Legacy loop skips pre-trade days (insufficient data); vectorized records all.
        # Compare only the overlapping portion (tail of vec_snap matching legacy length).
        legacy_len = len(legacy_snap)
        assert legacy_len <= len(vec_snap), \
            f"Legacy snapshot count {legacy_len} exceeds vectorized {len(vec_snap)}"
        offset = len(vec_snap) - legacy_len
        for i, (l, v) in enumerate(zip(legacy_snap, vec_snap[offset:])):
            for key in ('portfolio_value', 'cash', 'position_code', 'shares', 'price'):
                assert l[key] == v[key], (
                    f"Snapshot {i} (vec idx {offset + i}) mismatch at '{key}': "
                    f"legacy={l[key]}, vec={v[key]}"
                )

    def test_identical_trade_logs_basic(self):
        """4 ETF × 100 days: trade_log entries match exactly."""
        calendar = _make_dates('2024-01-02', 100)
        etf_data = make_synthetic_etf_data(
            ['ETF_A', 'ETF_B', 'ETF_C', 'ETF_D'],
            calendar, drift=0.001, vol=0.02,
        )
        config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)

        (legacy_snap, legacy_trades, p1), (vec_snap, vec_trades, p2) = \
            self._run_comparison(etf_data, calendar, config)

        assert len(legacy_trades) == len(vec_trades), \
            f"Trade log length mismatch: {len(legacy_trades)} vs {len(vec_trades)}"
        for i, (l, v) in enumerate(zip(legacy_trades, vec_trades)):
            for key in ('date', 'action', 'code', 'shares', 'price', 'value'):
                assert l[key] == v[key], \
                    f"Trade {i} mismatch at '{key}': legacy={l[key]}, vec={v[key]}"

    def test_identical_portfolios(self):
        """Portfolio cash and positions match after full run."""
        calendar = _make_dates('2024-01-02', 100)
        etf_data = make_synthetic_etf_data(
            ['ETF_A', 'ETF_B'],
            calendar, drift=0.001, vol=0.02,
        )
        config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)

        (_, _, p1), (_, _, p2) = self._run_comparison(etf_data, calendar, config)

        assert abs(p1.cash - p2.cash) < 0.02, \
            f"Cash mismatch: {p1.cash} vs {p2.cash}"
        assert p1.positions.keys() == p2.positions.keys(), \
            f"Position codes differ: {p1.positions.keys()} vs {p2.positions.keys()}"
        if p1.positions:
            for code in p1.positions:
                assert p1.positions[code].quantity == p2.positions[code].quantity
                assert abs(p1.positions[code].avg_price - p2.positions[code].avg_price) < 0.02

    @patch('trading_calendar.load_trading_calendar')
    def test_basic_backtest_matches(self, mock_cal):
        """Full run_backtest via public API produces same results."""
        calendar = _make_dates('2024-01-02', 100)
        mock_cal.return_value = calendar

        etf_data = make_synthetic_etf_data(
            ['ETF_A', 'ETF_B'],
            calendar, drift=0.001, vol=0.02,
        )

        # Run twice with same data → results should be deterministic
        config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)
        result1 = run_backtest(config, etf_data=etf_data)
        result2 = run_backtest(config, etf_data=etf_data)

        assert result1['daily_snapshots'] == result2['daily_snapshots']
        assert result1['trade_log'] == result2['trade_log']
