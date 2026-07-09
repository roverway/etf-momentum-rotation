"""Tests for config.py — ETF pool, BacktestConfig, and map_to_sina_code."""

from datetime import date

import pytest

from config import (
    ETF_POOL,
    CHECK_RANGE,
    REBALANCE_THRESHOLD,
    VOL_CHECK_RANGE,
    VOLATILITY_LAMBDA,
    BacktestConfig,
    map_to_sina_code,
)


class TestETFpool:
    """ETF_POOL 常量验证"""

    def test_contains_expected_etfs(self):
        assert '513100.XSHG' in ETF_POOL
        assert '159915.XSHE' in ETF_POOL
        assert '518880.XSHG' in ETF_POOL
        assert '512890.XSHG' in ETF_POOL

    def test_length(self):
        assert len(ETF_POOL) == 4

    def test_check_range(self):
        assert CHECK_RANGE == 22

    def test_rebalance_threshold(self):
        assert REBALANCE_THRESHOLD == 0.01

    def test_vol_check_range(self):
        assert VOL_CHECK_RANGE == 60

    def test_volatility_lambda(self):
        assert VOLATILITY_LAMBDA == 1.0

    def test_benchmark_code_constant(self):
        assert BacktestConfig().benchmark_code == '000300.XSHG'


class TestMapToSinaCode:
    """map_to_sina_code 映射函数验证"""

    def test_xshg_to_sh(self):
        assert map_to_sina_code('513100.XSHG') == 'sh513100'

    def test_xshe_to_sz(self):
        assert map_to_sina_code('159915.XSHE') == 'sz159915'

    def test_another_xshg(self):
        assert map_to_sina_code('518880.XSHG') == 'sh518880'

    def test_another_xshe(self):
        assert map_to_sina_code('512890.XSHG') == 'sh512890'


class TestBacktestConfig:
    """BacktestConfig 数据类验证"""

    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.start_date == '2014-01-01'
        assert cfg.end_date == ''
        assert cfg.initial_cash == 1_000_000
        assert cfg.commission_rate == 0.0002
        assert cfg.slippage_rate == 0.001
        assert cfg.cash_return_rate == 0.0
        assert cfg.benchmark_code == '000300.XSHG'

    def test_custom_initial_cash(self):
        cfg = BacktestConfig(initial_cash=500_000)
        assert cfg.initial_cash == 500_000

    def test_custom_start_date(self):
        cfg = BacktestConfig(start_date='2020-06-01')
        assert cfg.start_date == '2020-06-01'

    def test_custom_end_date(self):
        cfg = BacktestConfig(end_date='2024-12-31')
        assert cfg.end_date == '2024-12-31'

    def test_custom_commission_rate(self):
        cfg = BacktestConfig(commission_rate=0.001)
        assert cfg.commission_rate == 0.001

    def test_custom_slippage_rate(self):
        cfg = BacktestConfig(slippage_rate=0.0005)
        assert cfg.slippage_rate == 0.0005

    def test_custom_cash_return_rate(self):
        cfg = BacktestConfig(cash_return_rate=0.02)
        assert cfg.cash_return_rate == 0.02

    def test_custom_benchmark_code(self):
        cfg = BacktestConfig(benchmark_code='000688.XSHG')
        assert cfg.benchmark_code == '000688.XSHG'

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(BacktestConfig)
