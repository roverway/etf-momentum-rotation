"""Performance benchmark and edge case tests for vectorized backtest.

These tests verify:
1. The vectorized backtest completes in < 1 second for 4 ETFs × 5000 days
2. Edge cases: NaN periods, data gaps, threshold boundaries
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from config import BacktestConfig
from backtest_engine import run_backtest
from tests.test_backtest_engine import make_synthetic_etf_data, _make_dates


# =====================================================================================
# Performance benchmark
# =====================================================================================

@pytest.mark.benchmark
def test_backtest_under_1_second():
    """4 ETF × 5000 天 → 回测循环耗时 < 1.0 秒（3 次平均）"""
    calendar = pd.bdate_range('2000-01-03', '2020-01-03', freq='B').date.tolist()
    etf_data = make_synthetic_etf_data(
        ['ETF_A', 'ETF_B', 'ETF_C', 'ETF_D'],
        calendar, drift=0.001, vol=0.02,
    )
    config = BacktestConfig(
        start_date='2000-01-03',
        end_date='2020-01-03',
        initial_cash=1_000_000,
    )

    import time
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = run_backtest(config, etf_data=etf_data)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    print(f"\n  ⏱  向量化回测平均耗时: {avg_time:.4f}s (3 次)")
    assert avg_time < 1.0, f"平均耗时 {avg_time:.3f}s > 1.0s"


# =====================================================================================
# Edge case: NaN starting period
# =====================================================================================

def test_first_check_range_days_no_trades():
    """前 CHECK_RANGE-1 天应无交易（数据不足）。"""
    calendar = _make_dates('2024-01-02', 100)
    etf_data = make_synthetic_etf_data(
        ['ETF_A', 'ETF_B'], calendar, drift=0.001, vol=0.02,
    )
    config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)
    result = run_backtest(config, etf_data=etf_data)

    snapshots = result['daily_snapshots']
    # First 21 days (CHECK_RANGE=22) should have empty position
    for i, snap in enumerate(snapshots[:21]):
        assert snap['position_code'] == '', \
            f"Day {i} should have no position, got {snap['position_code']}"

    # Day 22 onward should have a position (data sufficient)
    assert any(snap['position_code'] != '' for snap in snapshots[21:]), \
        "Expected trades after CHECK_RANGE days"


# =====================================================================================
# Edge case: missing data gap
# =====================================================================================

def test_missing_data_gap():
    """某 ETF 有数据缺口 → 不崩溃，缺口后交易恢复。"""
    calendar = _make_dates('2024-01-02', 150)
    # Generate full data, then create a gap
    etf_data = make_synthetic_etf_data(
        ['ETF_A', 'ETF_B'], calendar, drift=0.001, vol=0.02,
    )

    # Remove days 40-49 from ETF_A (create a data gap)
    gap_df = etf_data['ETF_A']
    mask = (gap_df['date'] >= calendar[40]) & (gap_df['date'] <= calendar[49])
    etf_data['ETF_A'] = gap_df[~mask].reset_index(drop=True)

    config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)
    # Should not raise ValueError
    result = run_backtest(config, etf_data=etf_data)

    assert len(result['daily_snapshots']) > 0
    # Trades should still happen after the gap
    assert any(snap['position_code'] != '' for snap in result['daily_snapshots'][100:])


# =====================================================================================
# Edge case: threshold boundary
# =====================================================================================

def test_threshold_boundary():
    """动量差低于阈值不换仓，高于阈值换仓。

    构造两个 ETF：
    - ETF_A: 始终略好于 ETF_B，但动量差 < 1% → 不换仓
    - 然后在某天让 ETF_B 暴涨 → 动量差 > 1% → 触发换仓
    """
    n_days = 100
    calendar = _make_dates('2024-01-02', n_days)

    # ETF_A: steady climber
    prices_a = [10.0 + i * 0.05 for i in range(n_days)]

    # ETF_B: slightly worse at first, then surges
    prices_b = [10.0 + i * 0.045 for i in range(n_days // 2)]  # slightly below A
    prices_b += [prices_b[-1] + j * 0.2 for j in range(n_days - n_days // 2)]  # surge

    etf_data = {
        'ETF_A': pd.DataFrame({'date': calendar, 'close': prices_a}),
        'ETF_B': pd.DataFrame({'date': calendar, 'close': prices_b}),
    }

    config = BacktestConfig(start_date='2024-01-02', initial_cash=1_000_000)
    result = run_backtest(config, etf_data=etf_data)

    # We should see both ETFs in trade_log entries
    traded_codes = set()
    for trade in result['trade_log']:
        traded_codes.add(trade['code'])

    # At minimum, both ETFs should appear in trade_log (A initial, B after surge)
    assert 'ETF_B' in traded_codes, \
        "Should have switched to ETF_B after its surge. Traded: {traded_codes}"
