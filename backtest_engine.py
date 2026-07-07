"""backtest_engine.py — 回测引擎骨架。

连接交易日历、ETF 数据、组合跟踪、策略逻辑四大模块，按逐个交易日
执行动量信号计算与调仓，返回最终的 ``Portfolio`` 对象。

用法
----
::

    from config import BacktestConfig
    portfolio = run_backtest(BacktestConfig(start_date='2024-01-01'))
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import ETF_POOL, CHECK_RANGE, BacktestConfig
from data import load_all_etf_data
from portfolio import Portfolio
from strategy import calculate_momentum_signal
from trading_calendar import load_trading_calendar


# =====================================================================================
# Internal helpers
# =====================================================================================

def _to_date(value: str | date) -> date:
    """Normalize a string or ``datetime.date`` to ``datetime.date``."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _get_close_price(
    etf_data: dict[str, pd.DataFrame],
    code: str,
    trade_date: date,
) -> float:
    """Look up the closing price of *code* on *trade_date*.

    Raises
    ------
    ValueError
        If *trade_date* is not found in the ETF's DataFrame.
    """
    df = etf_data[code]
    matches = df[df['date'] == trade_date]
    if matches.empty:
        raise ValueError(f"No data for {code} on {trade_date}")
    return float(matches.iloc[0]['close'])


# =====================================================================================
# Public API
# =====================================================================================

def run_backtest(
    config,
    etf_data: Optional[dict[str, pd.DataFrame]] = None,
) -> Portfolio:
    """执行完整回测。

    Parameters
    ----------
    config : BacktestConfig
        回测配置，至少包含 ``start_date``、``end_date``、``initial_cash``。
        可选地包含 ``etf_codes``（没有时使用 ``config.ETF_POOL``）。
    etf_data : dict[str, pd.DataFrame], optional
        预加载的 ETF 数据（code → DataFrame with ``date``, ``close``）。
        为 *None* 时自动调用 ``load_all_etf_data()``。

    Returns
    -------
    Portfolio
        回测结束后的投资组合，包含最终持仓、现金、已实现盈亏等信息。
    """
    # ── 1. 交易日历 ────────────────────────────────────────────────────
    calendar: list[date] = load_trading_calendar()

    # ── 2. 过滤日期范围 ────────────────────────────────────────────────
    start = _to_date(config.start_date)
    end = _to_date(config.end_date) if config.end_date else date.today()
    calendar = [d for d in calendar if start <= d <= end]

    if not calendar:
        return Portfolio(config.initial_cash)

    # ── 3. 加载 ETF 数据 ──────────────────────────────────────────────
    etf_codes = getattr(config, 'etf_codes', ETF_POOL)
    if etf_data is None:
        etf_data = load_all_etf_data(etf_codes)

    if not etf_data:
        return Portfolio(config.initial_cash)

    # ── 4. 创建投资组合 ───────────────────────────────────────────────
    portfolio = Portfolio(config.initial_cash)

    # ── 5. 逐日回测 ───────────────────────────────────────────────────
    for trade_date in calendar:
        # 5a. 过滤当前日期之前的 ETF 数据
        etf_data_dict: dict[str, pd.DataFrame] = {}
        for code, df in etf_data.items():
            sub = df[df['date'] <= trade_date]
            if not sub.empty:
                etf_data_dict[code] = sub

        if not etf_data_dict:
            continue

        # 5b. 可用历史数据 < CHECK_RANGE → 跳过
        max_rows = max(len(sub) for sub in etf_data_dict.values())
        if max_rows < CHECK_RANGE:
            continue

        # 5c. 计算动量信号
        target: str | None = calculate_momentum_signal(
            etf_data_dict, CHECK_RANGE, trade_date,
        )

        # 5d. 当前持仓
        current_holding: str | None = (
            next(iter(portfolio.positions)) if portfolio.positions else None
        )

        # 5e. 调仓
        if target is not None and target != current_holding:
            # 卖出旧持仓
            if current_holding:
                sell_price = _get_close_price(
                    etf_data, current_holding, trade_date,
                )
                portfolio.sell(
                    current_holding,
                    portfolio.positions[current_holding].quantity,
                    sell_price,
                )
            # 买入新目标
            buy_price = _get_close_price(etf_data, target, trade_date)
            qty = int(portfolio.cash // buy_price)
            if qty > 0:
                portfolio.buy(target, qty, buy_price)

        elif target is None and current_holding:
            # 全跌信号 → 卖出清仓
            sell_price = _get_close_price(
                etf_data, current_holding, trade_date,
            )
            portfolio.sell(
                current_holding,
                portfolio.positions[current_holding].quantity,
                sell_price,
            )

        # 5f. 更新持仓市价（仅用于估值，不影响调仓）
        for code in list(portfolio.positions.keys()):
            try:
                price = _get_close_price(etf_data, code, trade_date)
                portfolio.update_price(code, price)
            except ValueError:
                pass

    return portfolio
