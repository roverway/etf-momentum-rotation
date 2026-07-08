"""signal_generator.py — 实盘信号生成器。

提供:
  - generate_signal: 主入口，加载交易日历和 ETF 数据 → 计算动量 → 输出信号。
  - print_signal: 格式化打印信号结果到控制台。
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from config import ETF_POOL
from data import load_all_etf_data
from strategy import calculate_momentum_signal
from trading_calendar import get_previous_trading_date, load_trading_calendar


# =====================================================================================
# Console output
# =====================================================================================

def print_signal(result: dict) -> None:
    """打印信号到控制台。

    Parameters
    ----------
    result : dict
        由 ``generate_signal()`` 返回的结果字典。
    """
    print(f"信号生成: {result['signal_date']}")
    print(f"回看期: {result['lookback_start']} ~ {result['lookback_end']}")
    for code, ret in result['returns'].items():
        arrow = "+" if ret > 0 else ""
        print(f"  {code}: {arrow}{ret:.2f}%")
    target = result['target'] or "空仓"
    print(f"TARGET: {target}")


# =====================================================================================
# Signal generator
# =====================================================================================

def generate_signal(
    etf_codes: Optional[list[str]] = None,
    check_range: int = 22,
) -> dict:
    """生成实盘信号。

    流程:
      1. 加载 ETF 数据 → 从实际收盘价确定信号日期；
      2. 根据信号日期计算数据起始日，筛选足够历史；
      3. 调用 ``calculate_momentum_signal()`` 确定目标 ETF；
      4. 打印格式化输出。

    Parameters
    ----------
    etf_codes : list[str] or None, optional
        待评测的 ETF 代码列表。为 *None* 时使用 ``config.ETF_POOL``。
    check_range : int, optional
        动量计算的回看天数（交易日），默认 22（≈ 1 个月）。

    Returns
    -------
    dict
        {
            'signal_date'    : datetime.date,   # 信号生成日（最后交易日）
            'lookback_start'  : datetime.date,   # 动量窗口起始日
            'lookback_end'    : datetime.date,   # 动量窗口结束日（同 signal_date）
            'returns'         : {code: pct},    # 各 ETF 在窗口内的收益率（%）
            'target'          : str or None,    # 目标 ETF 代码，空仓时为 None
        }
    """
    if etf_codes is None:
        etf_codes = ETF_POOL

    # ── 1. 加载 ETF 数据 → 从实际收盘价确定信号日期 ────────────────────
    all_data = load_all_etf_data(etf_codes)

    # 从各 ETF 最近有收盘价数据的日期中取最大值
    latest_dates: list[date] = []
    for code, df in all_data.items():
        valid = df[df['close'].notna()]
        if not valid.empty:
            latest_dates.append(valid['date'].max())
    if not latest_dates:
        print("⚠ 无可用 ETF 数据")
        return {
            'signal_date': None,
            'lookback_start': None,
            'lookback_end': None,
            'returns': {},
            'target': None,
        }
    signal_date = max(latest_dates)

    # ── 2. 加载足够的历史数据 ─────────────────────────────────────────────
    # 加载 check_range * 2 天的数据，确保 even after alignment 仍有足够行数
    data_start = get_previous_trading_date(signal_date, n=check_range * 2)

    filtered: dict[str, pd.DataFrame] = {}
    for code, df in all_data.items():
        sub = df[(df['date'] >= data_start) & (df['date'] <= signal_date)]
        if not sub.empty:
            filtered[code] = sub

    # 计算各 ETF 在动量窗口内的收益率
    lookback_start = get_previous_trading_date(signal_date, n=check_range - 1)
    returns: dict[str, float] = {}
    for code, df in filtered.items():
        window = df[df['date'] >= lookback_start]
        if len(window) >= 2:
            first = float(window['close'].iloc[0])
            last = float(window['close'].iloc[-1])
            returns[code] = round(((last - first) / first) * 100, 2)

    # ── 3. 计算动量信号 ───────────────────────────────────────────────────
    target = calculate_momentum_signal(filtered, check_range)

    result = {
        'signal_date': signal_date,
        'lookback_start': lookback_start,
        'lookback_end': signal_date,
        'returns': returns,
        'target': target,
    }

    # ── 4. 打印 ───────────────────────────────────────────────────────────
    print_signal(result)
    return result
