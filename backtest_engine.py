"""backtest_engine.py — 回测引擎骨架。

连接交易日历、ETF 数据、组合跟踪、策略逻辑四大模块，按逐个交易日
执行动量信号计算与调仓，返回包含 Portfolio + 快照 + 日志的字典。

用法
----
::

    from config import BacktestConfig
    result = run_backtest(BacktestConfig(start_date='2024-01-01'))
    portfolio = result['portfolio']
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
from trading_calendar import get_next_trading_date, load_trading_calendar

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

def _empty_result(config) -> dict:
    """Return an empty result dict (no trades possible)."""
    portfolio = Portfolio(config.initial_cash)
    return {
        'portfolio': portfolio,
        'daily_snapshots': [],
        'trade_log': [],
        'calendar': [],
        'etf_data': {},
    }


def run_backtest(
    config,
    etf_data: Optional[dict[str, pd.DataFrame]] = None,
    output_dir: Optional[str] = None,
) -> dict:
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
    dict
        {
            'portfolio': Portfolio,          # 最终投资组合
            'daily_snapshots': list[dict],   # 每日净值快照
            'trade_log': list[dict],         # 交易记录
            'calendar': list[date],          # 交易日历
            'etf_data': dict[str, pd.DataFrame],  # ETF 数据引用
        }
    """
    # ── 1. 交易日历 ────────────────────────────────────────────────────
    calendar: list[date] = load_trading_calendar()

    # ── 2. 过滤日期范围 ────────────────────────────────────────────────
    start = _to_date(config.start_date)
    end = _to_date(config.end_date) if config.end_date else date.today()
    calendar = [d for d in calendar if start <= d <= end]

    if not calendar:
        return _empty_result(config)

    # ── 3. 加载 ETF 数据 ──────────────────────────────────────────────
    etf_codes = getattr(config, 'etf_codes', ETF_POOL)
    if etf_data is None:
        etf_data = load_all_etf_data(etf_codes)

    if not etf_data:
        return _empty_result(config)

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

    return {
        'portfolio': portfolio,
        'daily_snapshots': daily_snapshots,
        'trade_log': trade_log,
        'calendar': calendar,
        'etf_data': etf_data,
    }


# =====================================================================================
# Metrics, reports & next-day suggestion
# =====================================================================================

def compute_and_print_metrics(
    daily_snapshots: list[dict],
    trade_log: list[dict],
    config,
    calendar: list[date],
    output_dir: str,
) -> dict:
    """计算全套业绩指标、打印到终端、生成 HTML 报告。

    Parameters
    ----------
    daily_snapshots : list[dict]
        每日净值快照（来自 ``run_backtest`` 返回值）。
    trade_log : list[dict]
        交易记录列表。
    config : BacktestConfig
        回测配置。
    calendar : list[date]
        交易日历。
    output_dir : str
        CSV 文件所在目录 / 报告输出目录。

    Returns
    -------
    dict
        完整指标字典（与 ``metrics.compute_all_metrics`` 返回值相同）。
    """
    import data as data_module
    import metrics as metrics_module
    from charts import generate_report

    if not daily_snapshots:
        print("⚠ 无回测数据，无法计算指标。")
        return {}

    # ── 1. 转换 daily_snapshots → pd.Series ──────────────────────────────
    dates = pd.to_datetime([s['date'] for s in daily_snapshots])
    values = [s['portfolio_value'] for s in daily_snapshots]
    portfolio_series = pd.Series(values, index=dates).sort_index()

    # ── 2. 加载基准数据 ─────────────────────────────────────────────────
    benchmark_df = data_module.fetch_benchmark_data(config.benchmark_code)
    benchmark_series = None
    if benchmark_df is not None and not benchmark_df.empty:
        bench_dates = pd.to_datetime(benchmark_df['date'])
        bench_close = benchmark_df['close'].astype(float)
        # 归一化基准到策略起始日
        bench_series_raw = pd.Series(bench_close.values, index=bench_dates).sort_index()
        # 裁剪到回测区间内
        bench_series_raw = bench_series_raw[
            (bench_series_raw.index >= portfolio_series.index[0]) &
            (bench_series_raw.index <= portfolio_series.index[-1])
        ]
        if len(bench_series_raw) > 1:
            # 归一化到起始净值 = 策略起始净值
            first_nav = portfolio_series.iloc[0]
            benchmark_series = bench_series_raw / bench_series_raw.iloc[0] * first_nav

    # ── 3. 调用 compute_all_metrics ─────────────────────────────────────
    metrics = metrics_module.compute_all_metrics(
        portfolio_series,
        benchmark_values=benchmark_series,
        trade_dates=calendar,
        risk_free_rate=0.0,
    )

    # Augment metrics with basic info
    metrics['start_date'] = str(calendar[0]) if calendar else ''
    metrics['end_date'] = str(calendar[-1]) if calendar else ''
    metrics['total_days'] = len(calendar)
    metrics['initial_capital'] = config.initial_cash
    metrics['final_value'] = portfolio_series.iloc[-1] if len(portfolio_series) > 0 else 0

    # Trade stats from trade_log
    metrics['total_trades'] = len(trade_log)
    if len(trade_log) > 1:
        # Average holding days: compute how many days between buy and corresponding sell
        buy_dates = {}
        holding_durations = []
        for trade in trade_log:
            tdate = trade['date'] if isinstance(trade['date'], date) else date.fromisoformat(str(trade['date']))
            if trade['action'] == 'BUY':
                buy_dates[trade['code']] = tdate
            elif trade['action'] == 'SELL' and trade['code'] in buy_dates:
                diff = (tdate - buy_dates[trade['code']]).days
                holding_durations.append(diff)
                del buy_dates[trade['code']]
        if holding_durations:
            metrics['avg_holding_days'] = round(sum(holding_durations) / len(holding_durations))
        else:
            metrics['avg_holding_days'] = 0
    else:
        metrics['avg_holding_days'] = 0

    # ── 年度收益率计算 ──
    daily_returns = portfolio_series.pct_change().dropna()
    strat_annual = daily_returns.groupby(daily_returns.index.year).apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    metrics['strategy_annual_returns'] = strat_annual.to_dict()

    if benchmark_series is not None:
        bench_daily_returns = benchmark_series.pct_change().dropna()
        bench_annual = bench_daily_returns.groupby(bench_daily_returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        metrics['benchmark_annual_returns'] = bench_annual.to_dict()
    else:
        metrics['benchmark_annual_returns'] = None

    # ── 4. 打印格式化指标到终端 ─────────────────────────────────────────
    _print_metrics(metrics)

    # ── 5. 调用 charts.generate_report() 生成 HTML ──────────────────────
    net_worth_csv = os.path.join(output_dir, 'net_worth.csv')
    trades_csv = os.path.join(output_dir, 'trades.csv')
    report_path = os.path.join(output_dir, 'report.html')
    if os.path.isfile(net_worth_csv) and os.path.isfile(trades_csv):
        try:
            generate_report(net_worth_csv, trades_csv, metrics, report_path)
        except Exception as e:
            logger.warning("生成报告失败: %s", e)
    else:
        logger.info("CSV 文件不存在，跳过 HTML 报告生成（仅 backtest_results 目录有 CSV 时可用）")

    return metrics


def _print_metrics(metrics: dict) -> None:
    """打印格式化指标到终端。"""
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"   ETF动量轮动策略 - 回测报告")
    print(f"{sep}\n")

    # 基础信息
    print("【基础信息】")
    _print_line("回测区间", f"{metrics.get('start_date', '')} ~ {metrics.get('end_date', '')}")
    _print_line("交易日数", f"{metrics.get('total_days', 0)} 天")
    _print_line("起始资金", f"{metrics.get('initial_capital', 0):,.2f}")
    _print_line("最终资产", f"{metrics.get('final_value', 0):,.2f}")
    print()

    # 收益
    print("【收益】")
    _print_line("策略累计收益率", f"{metrics.get('strategy_cumulative_return_pct', 0):+.2f}%")
    _print_line("策略年化收益率", f"{metrics.get('strategy_annualized_return_pct', 0):+.2f}%")
    bcr = metrics.get('benchmark_cumulative_return_pct')
    if bcr is not None:
        _print_line("基准累计收益率", f"{bcr:+.2f}%")
    bar = metrics.get('benchmark_annualized_return_pct')
    if bar is not None:
        _print_line("基准年化收益率", f"{bar:+.2f}%")
    er = metrics.get('excess_return_pct')
    if er is not None:
        _print_line("超额收益", f"{er:+.2f}%")
    print()

    # 风险
    print("【风险】")
    _print_line("年化波动率", f"{metrics.get('annualized_volatility', 0):.2f}%")
    _print_line("年化下行波动率", f"{metrics.get('downside_deviation', 0):.2f}%")
    mdd = metrics.get('max_drawdown_pct', 0)
    _print_line("最大回撤", f"-{mdd:.2f}%" if mdd else "0.00%")
    _print_line("最大回撤持续", f"{metrics.get('max_drawdown_duration', 0)} 天")
    print()

    # 胜率（战胜基准）
    print("【胜率（战胜基准）】")
    _print_line("日胜率", f"{metrics.get('daily_win_rate', 0):.2f}%")
    _print_line("月胜率", f"{metrics.get('monthly_win_rate', 0):.2f}%")
    _print_line("季度胜率", f"{metrics.get('quarterly_win_rate', 0):.2f}%")
    _print_line("年胜率", f"{metrics.get('yearly_win_rate', 0):.2f}%")
    print()
    
    # 胜率（策略正收益）
    print("【胜率（策略正收益）】")
    _print_line("日胜率", f"{metrics.get('daily_win_rate_abs', 0):.2f}%")
    _print_line("月胜率", f"{metrics.get('monthly_win_rate_abs', 0):.2f}%")
    _print_line("季度胜率", f"{metrics.get('quarterly_win_rate_abs', 0):.2f}%")
    _print_line("年胜率", f"{metrics.get('yearly_win_rate_abs', 0):.2f}%")
    print()

    # 风险调整收益
    print("【风险调整收益】")
    _print_line("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
    _print_line("索提诺比率", f"{metrics.get('sortino_ratio', 0):.2f}")
    _print_line("卡玛比率", f"{metrics.get('calmar_ratio', 0):.2f}")
    print()

    # 相对基准
    alpha = metrics.get('alpha')
    beta = metrics.get('beta')
    if alpha is not None or beta is not None:
        print("【相对基准】")
        if alpha is not None:
            _print_line("阿尔法", f"{alpha:+.2f}%")
        if beta is not None:
            _print_line("贝塔", f"{beta:.2f}")
        ir = metrics.get('information_ratio')
        if ir is not None:
            _print_line("信息比率", f"{ir:.2f}")
        te = metrics.get('tracking_error')
        if te is not None:
            _print_line("跟踪误差", f"{te:.2f}%")
        print()

    # 交易统计
    print("【交易统计】")
    _print_line("总交易次数", f"{metrics.get('total_trades', 0)}")
    _print_line("平均持仓天数", f"{metrics.get('avg_holding_days', 0)} 天")
    print()

    # 年度收益率
    strat_annual = metrics.get('strategy_annual_returns')
    bench_annual = metrics.get('benchmark_annual_returns')
    if strat_annual:
        print("【年度收益率】")
        _print_line("年份", "策略收益率    基准收益率    超额收益")
        for year in sorted(strat_annual.keys()):
            s_ret = strat_annual[year]
            b_ret = bench_annual.get(year, 0.0) if bench_annual else 0.0
            excess = s_ret - b_ret
            line = f"{year}    {s_ret:+8.2f}%    {b_ret:+8.2f}%    {excess:+8.2f}%"
            visual_pad = 6  # align "年份" label with first year
            print(f"  {' ' * visual_pad}{line}")
        print()

    print(f"{sep}\n")


def _print_line(label: str, value: str) -> None:
    """打印一行指标。"""
    # 中文宽度约 2 个英文字符，计算对齐
    visual_len = sum(2 if ord(c) > 127 else 1 for c in label)
    pad = max(2, 24 - visual_len)
    print(f"  {label}: {' ' * pad}{value}")


def print_next_day_suggestion(
    calendar: list[date],
    etf_data: dict[str, pd.DataFrame],
    config,
    last_holding: str | None,
) -> None:
    """计算下一交易日的操作建议并打印。

    Parameters
    ----------
    calendar : list[date]
        回测使用的交易日历。
    etf_data : dict[str, pd.DataFrame]
        ETF 数据（code → DataFrame with ``date``, ``close``）。
    config : BacktestConfig
        回测配置（用于获取 ``CHECK_RANGE`` 等参数）。
    last_holding : str or None
        回测结束时的持仓代码，None 表示空仓。
    """
    if not calendar:
        print("⚠ 无交易日历，无法计算下一日建议。")
        return

    try:
        next_date = get_next_trading_date(calendar[-1])
    except ValueError:
        print("⚠ 无法获取下一交易日（已到最后交易日）。")
        return

    # Filter data up to next_date
    filtered: dict[str, pd.DataFrame] = {}
    for code, df in etf_data.items():
        sub = df[df['date'] <= next_date]
        if not sub.empty:
            filtered[code] = sub

    if not filtered:
        suggestion = "数据不足，无法计算信号"
    else:
        check_range = getattr(config, 'check_range', CHECK_RANGE)
        max_rows = max(len(sub) for sub in filtered.values())
        if max_rows < check_range:
            suggestion = f"数据不足（{max_rows} 行 < {check_range}），无法计算信号"
        else:
            target = calculate_momentum_signal(filtered, check_range, base_date=next_date)
            if target is not None:
                suggestion = f"BUY {target}"
            else:
                suggestion = "空仓 (HOLD)"

    print(f"\n{'=' * 15} 下一交易日操作建议 {'=' * 15}")
    print(f"日期: {next_date}")
    print(f"建议操作: {suggestion}")
    print(f"{'=' * 52}\n")
