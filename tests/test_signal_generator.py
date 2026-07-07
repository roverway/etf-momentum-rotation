"""Tests for signal_generator — real trading signal generator.

Uses mock data to avoid real API calls.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from signal_generator import generate_signal, print_signal


# =====================================================================================
# Helpers
# =====================================================================================

def _mock_calendar() -> list[datetime.date]:
    """Return 60 consecutive business days (enough for check_range * 2 with margin)."""
    return sorted(
        pd.bdate_range('2024-01-02', periods=60, freq='B').date.tolist()
    )


def _make_etf_df(
    dates: list[datetime.date],
    prices: list[float],
) -> pd.DataFrame:
    """Build a minimal mock ETF DataFrame with ``date`` and ``close`` columns."""
    return pd.DataFrame({'date': dates, 'close': prices})


# =====================================================================================
# Tests: generate_signal
# =====================================================================================

class TestGenerateSignal:
    """generate_signal() — structure, content, edge cases."""

    # -- structure ----------------------------------------------------------------

    def test_returns_correct_structure(self):
        """Result dict contains all five expected keys with correct types."""
        dates = _mock_calendar()
        n = len(dates)
        data_dict = {
            '513100.XSHG': _make_etf_df(dates, [10.0 + i * 0.5 for i in range(n)]),
            '159915.XSHE': _make_etf_df(dates, [10.0 - i * 0.1 for i in range(n)]),
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(
                etf_codes=['513100.XSHG', '159915.XSHE'],
                check_range=5,
            )

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            'signal_date', 'lookback_start', 'lookback_end', 'returns', 'target',
        }
        assert isinstance(result['signal_date'], datetime.date)
        assert isinstance(result['lookback_start'], datetime.date)
        assert isinstance(result['lookback_end'], datetime.date)
        assert isinstance(result['returns'], dict)
        assert result['signal_date'] == dates[-1]
        assert result['lookback_end'] == dates[-1]

    def test_returns_contains_all_etf_codes(self):
        """returns dict has an entry per input ETF code."""
        dates = _mock_calendar()
        etf_codes = ['513100.XSHG', '159915.XSHE', '518880.XSHG']
        n = len(dates)
        data_dict = {
            code: _make_etf_df(dates, [10.0 + i * 0.2 for i in range(n)])
            for code in etf_codes
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(etf_codes=etf_codes, check_range=5)

        for code in etf_codes:
            assert code in result['returns']

    # -- target selection --------------------------------------------------------

    def test_selects_etf_with_highest_momentum(self):
        """Multi-ETF: target is the one with highest return."""
        dates = _mock_calendar()
        n = len(dates)
        data_dict = {
            'ETF_A': _make_etf_df(dates, [10.0 + i * 0.5 for i in range(n)]),   # ↑ strongly
            'ETF_B': _make_etf_df(dates, [10.0 - i * 0.1 for i in range(n)]),   # ↓
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(etf_codes=['ETF_A', 'ETF_B'], check_range=5)

        assert result['target'] == 'ETF_A'

    def test_all_negative_returns_none_target(self):
        """All ETFs down → target is None."""
        dates = _mock_calendar()
        n = len(dates)
        # Slopes keep prices positive throughout 60 periods
        data_dict = {
            'ETF_A': _make_etf_df(dates, [10.0 - i * 0.03 for i in range(n)]),
            'ETF_B': _make_etf_df(dates, [10.0 - i * 0.05 for i in range(n)]),
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(etf_codes=['ETF_A', 'ETF_B'], check_range=5)

        assert result['target'] is None

    # -- single ETF --------------------------------------------------------------

    def test_single_etf_positive(self):
        """Single ETF with positive momentum → target is its code."""
        dates = _mock_calendar()
        n = len(dates)
        data_dict = {
            '513100.XSHG': _make_etf_df(dates, [10.0 + i * 0.3 for i in range(n)]),
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(etf_codes=['513100.XSHG'], check_range=5)

        assert result['target'] == '513100.XSHG'

    def test_single_etf_negative_target_none(self):
        """Single ETF with negative returns → target is None."""
        dates = _mock_calendar()
        n = len(dates)
        data_dict = {
            'X': _make_etf_df(dates, [10.0 - i * 0.05 for i in range(n)]),
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict),
        ):
            result = generate_signal(etf_codes=['X'], check_range=5)

        assert result['target'] is None

    # -- default etf_codes -------------------------------------------------------

    def test_default_uses_etf_pool(self):
        """When etf_codes is None, loads data for ETF_POOL codes."""
        from config import ETF_POOL

        dates = _mock_calendar()
        n = len(dates)
        data_dict = {
            code: _make_etf_df(dates, [10.0 + i * 0.2 for i in range(n)])
            for code in ETF_POOL
        }
        with (
            patch('signal_generator.load_trading_calendar', return_value=dates),
            patch('trading_calendar.load_trading_calendar', return_value=dates),
            patch('signal_generator.load_all_etf_data', return_value=data_dict) as mock_load,
        ):
            result = generate_signal(check_range=5)

        mock_load.assert_called_once_with(ETF_POOL)
        assert result['target'] is not None  # all rising → some target


# =====================================================================================
# Tests: print_signal
# =====================================================================================

class TestPrintSignal:
    """print_signal() — output format."""

    def test_output_contains_target_line(self, capsys):
        """Output includes 'TARGET: <code>'."""
        result = {
            'signal_date': datetime.date(2024, 6, 14),
            'lookback_start': datetime.date(2024, 5, 16),
            'lookback_end': datetime.date(2024, 6, 14),
            'returns': {'513100.XSHG': 5.23, '159915.XSHE': -1.05},
            'target': '513100.XSHG',
        }
        print_signal(result)
        captured = capsys.readouterr()
        assert 'TARGET: 513100.XSHG' in captured.out

    def test_target_none_shows_kongcang(self, capsys):
        """When target is None, output shows 'TARGET: 空仓'."""
        result = {
            'signal_date': datetime.date(2024, 6, 14),
            'lookback_start': datetime.date(2024, 5, 16),
            'lookback_end': datetime.date(2024, 6, 14),
            'returns': {'513100.XSHG': -2.1},
            'target': None,
        }
        print_signal(result)
        captured = capsys.readouterr()
        assert 'TARGET: 空仓' in captured.out

    def test_positive_return_shows_plus_sign(self, capsys):
        """Positive return values are prefixed with '+'."""
        result = {
            'signal_date': datetime.date(2024, 6, 14),
            'lookback_start': datetime.date(2024, 5, 16),
            'lookback_end': datetime.date(2024, 6, 14),
            'returns': {'ETF_A': 3.5, 'ETF_B': -2.0},
            'target': 'ETF_A',
        }
        print_signal(result)
        captured = capsys.readouterr()
        assert '+3.50%' in captured.out
        assert '-2.00%' in captured.out

    def test_output_contains_signal_date_and_lookback(self, capsys):
        """Output includes the signal date and lookback period."""
        result = {
            'signal_date': datetime.date(2024, 6, 14),
            'lookback_start': datetime.date(2024, 5, 16),
            'lookback_end': datetime.date(2024, 6, 14),
            'returns': {},
            'target': None,
        }
        print_signal(result)
        captured = capsys.readouterr()
        assert '2024-06-14' in captured.out
        assert '2024-05-16' in captured.out
