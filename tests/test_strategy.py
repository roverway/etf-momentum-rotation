"""Tests for strategy module — momentum signal, trade logging, logger setup.

TDD: tests written before implementation.
"""

from __future__ import annotations

import datetime
import logging

import pandas as pd
import pytest

from strategy import calculate_momentum_signal, log_trade_signal, setup_logger


# =====================================================================================
# Helpers
# =====================================================================================

def _make_etf_df(prices: list[float],
                 start_date: str = '2024-01-02') -> pd.DataFrame:
    """Build a mock ETF DataFrame with *prices* mapped across consecutive business days.

    Columns: ``date`` (datetime.date), ``close`` (float).
    """
    dates = pd.bdate_range(start=start_date, periods=len(prices), freq='B')
    return pd.DataFrame({
        'date': [d.date() for d in dates],
        'close': prices,
    })


def _returns(df: pd.DataFrame) -> float:
    """Compute the simple return over the full DataFrame's close prices."""
    return (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]


# =====================================================================================
# Tests: calculate_momentum_signal
# =====================================================================================

class TestCalculateMomentumSignal:
    """Core momentum calculation — must match original RiceQuant logic 1:1."""

    # -- normal multi-ETF -------------------------------------------------------

    def test_selects_etf_with_highest_positive_return(self):
        """Multi-ETF: pick the one with highest positive momentum."""
        n = 22
        data = {
            '513100.XSHG': _make_etf_df([10.0 + i * 0.5 for i in range(n)]),   # ↑  +105%
            '159915.XSHE': _make_etf_df([10.0 + i * 0.1 for i in range(n)]),   # ↑   +21%
            '518880.XSHG': _make_etf_df([10.0 - i * 0.1 for i in range(n)]),   # ↓   -21%
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == '513100.XSHG'

    def test_selects_correct_etf_with_varying_positive_returns(self):
        """Multi-ETF: highest positive wins even when others are also positive."""
        n = 22
        data = {
            'ETF_A': _make_etf_df([5.0 + i * 0.2 for i in range(n)]),   #  +84%
            'ETF_B': _make_etf_df([5.0 + i * 0.5 for i in range(n)]),   # +210% ← winner
            'ETF_C': _make_etf_df([5.0 + i * 0.3 for i in range(n)]),   # +126%
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == 'ETF_B'

    # -- all negative / no positive --------------------------------------------

    def test_all_negative_returns_returns_none(self):
        """All ETFs down → None."""
        n = 22
        data = {
            'ETF_A': _make_etf_df([10.0 - i * 0.1 for i in range(n)]),
            'ETF_B': _make_etf_df([10.0 - i * 0.2 for i in range(n)]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None

    def test_best_return_is_zero_returns_none(self):
        """Best return is exactly 0 → None (not > 0)."""
        n = 22
        # Constant price → return = 0
        data = {
            'ETF_A': _make_etf_df([10.0] * n),
            'ETF_B': _make_etf_df([10.0 - i * 0.1 for i in range(n)]),  # down
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None

    # -- single ETF (Series path in RQ) ----------------------------------------

    def test_single_etf_positive_return(self):
        """Single ETF with positive return → returns its code."""
        n = 22
        data = {
            '513100.XSHG': _make_etf_df([10.0 + i * 0.3 for i in range(n)]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == '513100.XSHG'

    def test_single_etf_negative_return_returns_none(self):
        """Single ETF with negative return → None."""
        n = 22
        data = {
            'X': _make_etf_df([10.0 - i * 0.2 for i in range(n)]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None

    # -- insufficient data -----------------------------------------------------

    def test_insufficient_data_returns_none(self):
        """DataFrame has fewer rows than check_range → None."""
        data = {
            'A': _make_etf_df([10.0, 11.0, 12.0, 13.0, 14.0]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None

    def test_insufficient_data_multiple_etfs_returns_none(self):
        """Multiple ETFs, all with < check_range rows → None."""
        data = {
            'A': _make_etf_df([10.0, 11.0, 12.0]),
            'B': _make_etf_df([20.0, 21.0, 22.0, 23.0, 24.0]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None

    def test_some_etfs_insufficient_data_ignored(self):
        """One ETF has < check_range data → it's skipped; others still compared."""
        n = 22
        data = {
            'SHORT': _make_etf_df([10.0, 11.0, 12.0]),                      # too short
            'LONG_A': _make_etf_df([10.0 + i * 0.5 for i in range(n)]),      # ↑ winner
            'LONG_B': _make_etf_df([10.0 + i * 0.1 for i in range(n)]),      # ↑ lower
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == 'LONG_A'

    # -- deterministic tie-breaking --------------------------------------------

    def test_tie_goes_to_alphabetically_first(self):
        """Identical returns → alphabetically first ETF code wins (deterministic)."""
        n = 22
        prices_a = [10.0 + i * 0.5 for i in range(n)]
        prices_b = [10.0 + i * 0.5 for i in range(n)]   # same!
        data = {
            'ETF_B': _make_etf_df(prices_b),
            'ETF_A': _make_etf_df(prices_a),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == 'ETF_A'  # A before B alphabetically

    # -- base_date filtering ---------------------------------------------------

    def test_base_date_limits_data_window(self):
        """base_date restricts how much data is considered."""
        n = 40  # lots of data
        data = {
            'ETF_A': _make_etf_df([10.0 + i * 0.3 for i in range(n)]),
        }
        # base_date early enough that only ~10 rows are available (< check_range)
        result = calculate_momentum_signal(
            data, check_range=22,
            base_date=datetime.date(2024, 1, 16),
        )
        assert result is None  # not enough data before base_date

    def test_base_date_equal_to_latest_trading_day(self):
        """base_date == date in data → works like no filter (data ≤ base_date)."""
        n = 22
        data = {
            'A': _make_etf_df([10.0 + i * 0.5 for i in range(n)]),
        }
        # Last date in the mock data (from bdate_range(start='2024-01-02', periods=22))
        # Should include all 22 rows
        result = calculate_momentum_signal(
            data, check_range=22,
            base_date=datetime.date(2024, 1, 31),
        )
        assert result == 'A'

    def test_base_date_none_uses_latest_data(self):
        """No base_date → uses the latest available data."""
        n = 22
        data = {
            'A': _make_etf_df([10.0 + i * 0.5 for i in range(n)]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == 'A'

    # -- empty / edge cases ----------------------------------------------------

    def test_empty_dict_returns_none(self):
        """Empty etf_data_dict → None."""
        result = calculate_momentum_signal({}, check_range=22)
        assert result is None

    def test_etf_with_nan_close_handled(self):
        """NaN in close prices → those ETFs are excluded via dropna()."""
        n = 22
        data = {
            'GOOD': _make_etf_df([10.0 + i * 0.5 for i in range(n)]),
            'BAD':  _make_etf_df([float('nan') if i == 0 else 10.0 + i * 0.1
                                  for i in range(n)]),
        }
        result = calculate_momentum_signal(data, check_range=22)
        # BAD's first price is NaN → return becomes NaN → dropna removes it
        # GOOD alone has +105% > 0
        assert result == 'GOOD'

    # -- return value type -----------------------------------------------------

    def test_returns_none_when_no_positive_momentum(self):
        """Explicit None return when best etf return is not > 0."""
        n = 22
        data = {
            'A': _make_etf_df([10.0 - i * 0.01 for i in range(n)]),  # slightly down
            'B': _make_etf_df([10.0 + i * 0.01 for i in range(n)]),  # slightly up
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result == 'B'  # B is positive, so return B

    def test_returns_none_when_best_is_exactly_zero(self):
        """If the highest return is exactly 0, return None (must be > 0)."""
        n = 22
        # Two ETFs, both flat → return = 0 for both
        data = {
            'A': _make_etf_df([10.0] * n),
            'B': _make_etf_df([10.0] * n),
        }
        result = calculate_momentum_signal(data, check_range=22)
        assert result is None


# =====================================================================================
# Tests: log_trade_signal
# =====================================================================================

class TestLogTradeSignal:
    """Four code paths for trade action messages."""

    def test_hold(self, caplog):
        """Current == target → HOLD."""
        caplog.set_level(logging.INFO)
        log_trade_signal('513100.XSHG', '513100.XSHG', logging.getLogger('test'))
        assert 'HOLD' in caplog.text

    def test_buy_from_cash(self, caplog):
        """Current is None, target is set → BUY."""
        caplog.set_level(logging.INFO)
        log_trade_signal(None, '513100.XSHG', logging.getLogger('test'))
        assert 'BUY' in caplog.text

    def test_sell_to_cash(self, caplog):
        """Current is set, target is None → SELL TO CASH."""
        caplog.set_level(logging.INFO)
        log_trade_signal('513100.XSHG', None, logging.getLogger('test'))
        assert 'SELL TO CASH' in caplog.text

    def test_switch(self, caplog):
        """Both set but different → SWITCH."""
        caplog.set_level(logging.INFO)
        log_trade_signal('513100.XSHG', '159915.XSHE', logging.getLogger('test'))
        assert 'SWITCH' in caplog.text

    def test_output_contains_both_current_and_target(self, caplog):
        """Log message includes current position and target."""
        caplog.set_level(logging.INFO)
        log_trade_signal('OLD_ETF', 'NEW_ETF', logging.getLogger('test'))
        assert 'OLD_ETF' in caplog.text
        assert 'NEW_ETF' in caplog.text


# =====================================================================================
# Tests: setup_logger
# =====================================================================================

class TestSetupLogger:
    """Logger factory returns a configured logger."""

    def test_returns_logger_with_correct_name(self):
        logger = setup_logger('test_strat')
        assert logger.name == 'test_strat'
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        logger = setup_logger('test_handler')
        assert len(logger.handlers) >= 1

    def test_logger_info_level(self):
        logger = setup_logger('test_level')
        assert logger.level == logging.INFO

    def test_default_name(self):
        logger = setup_logger()
        assert logger.name == 'strategy'
