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

import csv
import logging
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import ETF_POOL, CHECK_RANGE, BacktestConfig
from data import load_all_etf_data
from portfolio import Portfolio
from strategy import calculate_momentum_signal
from trading_calendar import load_trading_calendar

logger = logging.getLogger(__name__)


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
# CSV output helpers
# =====================================================================================

_NET_WORTH_FIELDS = ['date', 'portfolio_value', 'cash', 'position_code', 'shares', 'price']
_TRADE_LOG_FIELDS = ['date', 'action', 'code', 'shares', 'price', 'value']


def save_results(
    portfolio: Portfolio,
    daily_snapshots: list[dict],
    trade_log: list[dict],
    output_dir: str = 'backtest_results/',
) -> None:
    """写入每日净值和交易记录到 CSV 文件。

    Parameters
    ----------
    portfolio : Portfolio
        回测结束后的投资组合（用于汇总日志）。
    daily_snapshots : list[dict]
        每日快照列表，每个 dict 包含 *date*, *portfolio_value*, *cash*,
        *position_code*, *shares*, *price*。
    trade_log : list[dict]
        交易记录列表，每个 dict 包含 *date*, *action*, *code*, *shares*,
        *price*, *value*。
    output_dir : str
        输出目录（默认 ``'backtest_results/'``），不存在时自动创建。
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── net_worth.csv ──────────────────────────────────────────────────────
    net_worth_path = os.path.join(output_dir, 'net_worth.csv')
    with open(net_worth_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_NET_WORTH_FIELDS)
        writer.writeheader()
        for snap in daily_snapshots:
            # Convert date to iso string for CSV
            row = dict(snap)
            if isinstance(row.get('date'), date):
                row['date'] = row['date'].isoformat()
            writer.writerow(row)

    # ── trades.csv ─────────────────────────────────────────────────────────
    trades_path = os.path.join(output_dir, 'trades.csv')
    with open(trades_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_LOG_FIELDS)
        writer.writeheader()
        for trade in trade_log:
            row = dict(trade)
            if isinstance(row.get('date'), date):
                row['date'] = row['date'].isoformat()
            writer.writerow(row)

    logger.info("回测结果已保存: net_worth.csv (%d 行), trades.csv (%d 行)",
                len(daily_snapshots), len(trade_log))


def compute_summary(
    portfolio: Portfolio,
    daily_snapshots: list[dict],
) -> dict:
    """计算总收益率、最大回撤等汇总指标。

    Parameters
    ----------
    portfolio : Portfolio
        回测结束后的投资组合。
    daily_snapshots : list[dict]
        每日快照列表，至少包含 *portfolio_value*。

    Returns
    -------
    dict
        包含 *initial_value*, *final_value*, *total_return*,
        *total_return_pct*, *max_drawdown_pct*, *total_pnl* 的字典。
        当 *daily_snapshots* 为空时返回空字典。
    """
    if not daily_snapshots:
        return {}

    initial_value = daily_snapshots[0]['portfolio_value']
    final_value = daily_snapshots[-1]['portfolio_value']
    total_return = final_value - initial_value
    total_return_pct = round(
        (total_return / initial_value) * 100 if initial_value else 0.0, 2,
    )

    # ── 最大回撤 (peak-to-trough) ──────────────────────────────────────
    peak = initial_value
    max_drawdown_pct = 0.0
    for snap in daily_snapshots:
        v = snap['portfolio_value']
        if v > peak:
            peak = v
        dd = ((peak - v) / peak * 100) if peak else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd
    max_drawdown_pct = round(max_drawdown_pct, 2)

    return {
        'initial_value': round(initial_value, 2),
        'final_value': round(final_value, 2),
        'total_return': round(total_return, 2),
        'total_return_pct': total_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'total_pnl': round(portfolio.total_pnl, 2),
    }


# =====================================================================================
# Public API
# =====================================================================================

def run_backtest(
    config,
    etf_data: Optional[dict[str, pd.DataFrame]] = None,
    output_dir: Optional[str] = None,
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
    commission_rate = getattr(config, 'commission_rate', 0.00025)
    slippage_rate = getattr(config, 'slippage_rate', 0.0001)

    # ── 5. 逐日回测 ───────────────────────────────────────────────────
    daily_snapshots: list[dict] = []
    trade_log: list[dict] = []

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
            logger.warning(
                "数据不足: 最大行数 %d < %d (CHECK_RANGE)，跳过 %s",
                max_rows, CHECK_RANGE, trade_date,
            )
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
                sell_qty = portfolio.positions[current_holding].quantity
                portfolio.sell(
                    current_holding, sell_qty, sell_price,
                    commission_rate=commission_rate,
                    slippage_rate=slippage_rate,
                )
                sell_effective_price = sell_price * (1 - slippage_rate)
                trade_log.append({
                    'date': trade_date,
                    'action': 'SELL',
                    'code': current_holding,
                    'shares': sell_qty,
                    'price': round(sell_price, 4),
                    'value': round(sell_qty * sell_effective_price, 2),
                })
            # 买入新目标（qty 需预留滑点+佣金空间）
            buy_price = _get_close_price(etf_data, target, trade_date)
            effective_buy_price = buy_price * (1 + slippage_rate)
            qty = int(portfolio.cash // (effective_buy_price * (1 + commission_rate)))
            if qty > 0:
                portfolio.buy(
                    target, qty, buy_price,
                    commission_rate=commission_rate,
                    slippage_rate=slippage_rate,
                )
                trade_log.append({
                    'date': trade_date,
                    'action': 'BUY',
                    'code': target,
                    'shares': qty,
                    'price': round(buy_price, 4),
                    'value': round(qty * effective_buy_price, 2),
                })

        elif target is None and current_holding:
            # 全跌信号 → 卖出清仓
            sell_price = _get_close_price(
                etf_data, current_holding, trade_date,
            )
            sell_qty = portfolio.positions[current_holding].quantity
            portfolio.sell(
                current_holding, sell_qty, sell_price,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )
            sell_effective_price = sell_price * (1 - slippage_rate)
            trade_log.append({
                'date': trade_date,
                'action': 'SELL',
                'code': current_holding,
                'shares': sell_qty,
                'price': round(sell_price, 4),
                'value': round(sell_qty * sell_effective_price, 2),
            })

        # 5f. 更新持仓市价（仅用于估值，不影响调仓）
        for code in list(portfolio.positions.keys()):
            try:
                price = _get_close_price(etf_data, code, trade_date)
                portfolio.update_price(code, price)
            except ValueError:
                pass

        # 5g. 每日净值快照
        snap_holding: str | None = (
            next(iter(portfolio.positions)) if portfolio.positions else None
        )
        snap_pos = portfolio.positions.get(snap_holding) if snap_holding else None
        daily_snapshots.append({
            'date': trade_date,
            'portfolio_value': round(portfolio.total_value, 2),
            'cash': round(portfolio.cash, 2),
            'position_code': snap_holding or '',
            'shares': snap_pos.quantity if snap_pos else 0,
            'price': round(snap_pos.current_price, 2) if snap_pos and snap_pos.current_price is not None else 0.0,
        })

    # ── 6. 汇总指标 & CSV 输出 ──────────────────────────────────────────
    summary = compute_summary(portfolio, daily_snapshots)
    if summary:
        logger.info(
            "回测汇总: 总收益率 %.2f%%, 最大回撤 %.2f%%, 总收益 %.2f, 最终资产 %.2f",
            summary['total_return_pct'],
            summary['max_drawdown_pct'],
            summary['total_return'],
            summary['final_value'],
        )

    if output_dir is not None:
        save_results(portfolio, daily_snapshots, trade_log, output_dir=output_dir)

    return portfolio
