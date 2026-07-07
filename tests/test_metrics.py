"""Tests for metrics.py — performance metrics calculation.

Tests use synthetic pandas data (no real API calls). Edge cases
(empty, single data point, all-zeros) are covered.
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from metrics import (
    annualized_return,
    annualized_vol,
    beta_alpha,
    compute_all_metrics,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def daily_values() -> pd.Series:
    """5 个交易日净值序列: 1.0 → 1.03 (累计+3%)"""
    return pd.Series([1.0, 1.01, 1.02, 1.015, 1.03],
                     index=pd.to_datetime(['2024-01-02', '2024-01-03',
                                            '2024-01-04', '2024-01-05',
                                            '2024-01-08']))


@pytest.fixture
def daily_returns_from_values(daily_values) -> pd.Series:
    return daily_values.pct_change().dropna()


@pytest.fixture
def constant_values() -> pd.Series:
    """净值不变（无波动）"""
    return pd.Series([1.0, 1.0, 1.0, 1.0, 1.0],
                     index=pd.to_datetime(['2024-01-02', '2024-01-03',
                                            '2024-01-04', '2024-01-05',
                                            '2024-01-08']))


@pytest.fixture
def benchmark_values() -> pd.Series:
    """基准净值: 1.0 → 1.02 (累计+2%)"""
    return pd.Series([1.0, 1.005, 1.01, 1.015, 1.02],
                     index=pd.to_datetime(['2024-01-02', '2024-01-03',
                                            '2024-01-04', '2024-01-05',
                                            '2024-01-08']))


@pytest.fixture
def benchmark_returns(benchmark_values) -> pd.Series:
    return benchmark_values.pct_change().dropna()


# ======================================================================
# annualized_return
# ======================================================================

class TestAnnualizedReturn:
    def test_positive_return(self):
        r = annualized_return(50.0, 2.0)  # 累计50%, 2年
        assert r == pytest.approx(22.474, rel=1e-3)

    def test_zero_years(self):
        assert annualized_return(10.0, 0.0) == 0.0

    def test_negative_years(self):
        assert annualized_return(10.0, -1.0) == 0.0

    def test_zero_return(self):
        r = annualized_return(0.0, 2.0)
        assert r == 0.0

    def test_negative_return(self):
        r = annualized_return(-10.0, 1.0)
        assert r == pytest.approx(-10.0, rel=1e-10)


# ======================================================================
# annualized_vol
# ======================================================================

class TestAnnualizedVol:
    def test_positive_vol(self, daily_returns_from_values):
        r = annualized_vol(daily_returns_from_values)
        assert r > 0

    def test_zero_vol(self):
        sr = pd.Series([0.01, 0.01, 0.01])
        r = annualized_vol(sr)
        assert r == 0.0

    def test_single_element(self):
        sr = pd.Series([0.01])
        r = annualized_vol(sr)
        assert r == 0.0


# ======================================================================
# sharpe_ratio
# ======================================================================

class TestSharpeRatio:
    def test_positive_sharpe(self, daily_returns_from_values):
        r = sharpe_ratio(daily_returns_from_values)
        assert r > 0

    def test_zero_vol_returns_zero(self):
        sr = pd.Series([0.01, 0.01, 0.01])  # 标准差=0
        r = sharpe_ratio(sr)
        assert r == 0.0

    def test_with_risk_free(self, daily_returns_from_values):
        r = sharpe_ratio(daily_returns_from_values, risk_free_rate=0.02)
        assert isinstance(r, float)


# ======================================================================
# sortino_ratio
# ======================================================================

class TestSortinoRatio:
    def test_positive_sortino(self, daily_returns_from_values):
        r = sortino_ratio(daily_returns_from_values)
        assert r > 0

    def test_no_negative_returns(self):
        sr = pd.Series([0.01, 0.02, 0.03])
        r = sortino_ratio(sr)
        assert r == float('inf')  # 无下行风险，返回inf

    def test_all_negative(self):
        sr = pd.Series([-0.01, -0.02, -0.03])
        r = sortino_ratio(sr)
        assert r < 0


# ======================================================================
# max_drawdown
# ======================================================================

class TestMaxDrawdown:
    def test_simple_drawdown(self, daily_values):
        dd_pct, duration = max_drawdown(daily_values)
        assert dd_pct == pytest.approx(0.4902, rel=1e-2)  # (1.02-1.015)/1.02
        assert duration == 1

    def no_decline(self, constant_values):
        dd_pct, duration = max_drawdown(constant_values)
        assert dd_pct == 0.0

    def test_single_element(self):
        sr = pd.Series([1.0])
        dd_pct, duration = max_drawdown(sr)
        assert dd_pct == 0.0

    def test_monotonic_increase(self):
        sr = pd.Series([1.0, 1.1, 1.2, 1.3])
        dd_pct, duration = max_drawdown(sr)
        assert dd_pct == 0.0

    def test_monotonic_decrease(self):
        sr = pd.Series([1.3, 1.2, 1.1, 1.0])
        dd_pct, duration = max_drawdown(sr)
        assert dd_pct == pytest.approx(23.0769, rel=1e-2)
        assert duration == 3

    def test_multiple_peaks(self):
        """从峰值1.2跌到1.0，再从新峰值1.15跌到1.05"""
        sr = pd.Series([1.0, 1.2, 1.1, 1.0, 1.15, 1.05])
        dd_pct, duration = max_drawdown(sr)
        # 最大回撤发生在 1.2 → 1.0，回撤 16.67%
        assert dd_pct == pytest.approx(16.666, rel=1e-2)


# ======================================================================
# win_rate
# ======================================================================

class TestWinRate:
    def test_all_positive(self):
        sr = pd.Series([0.01, 0.02, 0.03])
        assert win_rate(sr, 'D') == 100.0

    def test_mixed(self):
        sr = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
        r = win_rate(sr, 'D')
        assert r == 60.0

    def test_all_negative(self):
        sr = pd.Series([-0.01, -0.02, -0.03])
        assert win_rate(sr, 'D') == 0.0

    def test_empty(self):
        sr = pd.Series([], dtype=float)
        assert win_rate(sr, 'D') == 0.0

    def test_monthly_win_rate(self):
        """月度胜率: 3个月中有2个月正收益"""
        dates = pd.to_datetime(['2024-01-02', '2024-02-01', '2024-03-01'])
        sr = pd.Series([0.01, 0.02, -0.01], index=dates)
        r = win_rate(sr, 'M')
        assert r == pytest.approx(66.666, rel=1e-2)


# ======================================================================
# beta_alpha
# ======================================================================

class TestBetaAlpha:
    def test_beta_equal_one(self, daily_returns_from_values, benchmark_returns):
        """策略与基准完全同步时 beta=1"""
        beta, alpha = beta_alpha(daily_returns_from_values, daily_returns_from_values)
        assert beta == pytest.approx(1.0, rel=1e-10)
        assert alpha == pytest.approx(0.0, rel=1e-10)

    def test_beta_half(self, daily_returns_from_values):
        """策略波动只有基准一半"""
        bench = daily_returns_from_values * 2
        beta, alpha = beta_alpha(daily_returns_from_values, bench)
        assert beta == pytest.approx(0.5, rel=1e-2)

    def test_zero_benchmark_var(self):
        """基准无波动时 beta=0"""
        strat = pd.Series([0.01, -0.01, 0.02])
        bench = pd.Series([0.0, 0.0, 0.0])
        beta, alpha = beta_alpha(strat, bench)
        assert beta == 0.0


# ======================================================================
# information_ratio
# ======================================================================

class TestInformationRatio:
    def test_positive_ir(self, daily_returns_from_values, benchmark_returns):
        r = information_ratio(daily_returns_from_values, benchmark_returns)
        assert isinstance(r, float)

    def test_tracking_error_zero(self):
        strat = pd.Series([0.01, 0.02, 0.03])
        bench = pd.Series([0.01, 0.02, 0.03])
        r = information_ratio(strat, bench)
        assert r == 0.0

    def test_one_element_returns_zero(self):
        r = information_ratio(pd.Series([0.01]), pd.Series([0.005]))
        assert r == 0.0


# ======================================================================
# compute_all_metrics (integration)
# ======================================================================

class TestComputeAllMetrics:
    def test_full_output(self, daily_values, benchmark_values):
        """返回字典包含所有预期的键"""
        r = compute_all_metrics(daily_values, benchmark_values)
        expected_keys = {
            'start_date', 'end_date', 'total_days', 'trading_years',
            'strategy_cumulative_return_pct', 'strategy_annualized_return_pct',
            'benchmark_cumulative_return_pct', 'benchmark_annualized_return_pct',
            'excess_return_pct',
            'annualized_volatility', 'downside_deviation',
            'max_drawdown_pct', 'max_drawdown_duration',
            'daily_win_rate', 'monthly_win_rate', 'quarterly_win_rate',
            'yearly_win_rate',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'alpha', 'beta', 'information_ratio', 'tracking_error',
            'total_trades', 'avg_holding_days',
        }
        assert set(r.keys()) == expected_keys

    def test_strategy_cumulative_return(self, daily_values, benchmark_values):
        r = compute_all_metrics(daily_values, benchmark_values)
        assert r['strategy_cumulative_return_pct'] == pytest.approx(3.0, rel=1e-2)

    def test_without_benchmark(self, daily_values):
        """无基准数据时，基准相关指标应为 0 或 None"""
        r = compute_all_metrics(daily_values)
        assert r['benchmark_cumulative_return_pct'] == 0.0
        assert r['excess_return_pct'] == 0.0

    def test_single_value(self):
        """单数据点边界情况"""
        sr = pd.Series([1.0], index=pd.to_datetime(['2024-01-02']))
        r = compute_all_metrics(sr)
        assert r['total_days'] == 1
        assert r['strategy_cumulative_return_pct'] == 0.0

    def test_empty_series(self):
        """空序列边界情况"""
        sr = pd.Series([], dtype=float)
        r = compute_all_metrics(sr)
        assert r['total_days'] == 0
        assert r['strategy_cumulative_return_pct'] == 0.0

    def test_risk_free_rate(self, daily_values, benchmark_values):
        r = compute_all_metrics(daily_values, benchmark_values, risk_free_rate=0.02)
        assert isinstance(r['sharpe_ratio'], float)

    def test_trade_stats_defaults(self, daily_values):
        r = compute_all_metrics(daily_values)
        assert r['total_trades'] == 0
        assert r['avg_holding_days'] == 0.0

    def test_trade_dates_provided(self, daily_values):
        """提供 trade_dates 列表时计算平均持仓天数"""
        r = compute_all_metrics(daily_values, trade_dates=[
            date(2024, 1, 2), date(2024, 1, 5),
        ])
        assert r['total_trades'] == 2
        assert r['avg_holding_days'] == 3.0

    @pytest.mark.parametrize("rf,expected_type", [
        (0.0, float),
        (0.03, float),
    ])
    def test_sharpe_sortino_types(self, rf, expected_type, daily_values):
        r = compute_all_metrics(daily_values, risk_free_rate=rf)
        assert isinstance(r['sharpe_ratio'], expected_type)
        assert isinstance(r['sortino_ratio'], expected_type)
