"""metrics.py — 业绩评价指标计算。"""

import math
from datetime import date

import numpy as np
import pandas as pd


# =====================================================================================
# 基础工具函数
# =====================================================================================

def annualized_return(cumulative_return_pct: float, years: float) -> float:
    """年化收益率 = (1 + 累计收益率)^(1/年数) - 1

    Parameters
    ----------
    cumulative_return_pct : float
        累计收益率（%），如 50 表示 50%。
    years : float
        年数，如 2.5。

    Returns
    -------
    float
        年化收益率（%）。years <= 0 时返回 0。
    """
    if years <= 0:
        return 0.0
    return ((1 + cumulative_return_pct / 100) ** (1 / years) - 1) * 100


def annualized_vol(daily_returns: pd.Series) -> float:
    """年化波动率 = 日收益率标准差 * sqrt(252) * 100

    Parameters
    ----------
    daily_returns : pd.Series
        日收益率序列。

    Returns
    -------
    float
        年化波动率（%）。序列长度 < 2 时返回 0。
    """
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * math.sqrt(252) * 100)


# =====================================================================================
# 风险调整收益
# =====================================================================================

def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """夏普比率 = (年化收益 - 无风险利率) / 年化波动率

    年化收益由日收益率均值 * 252 估算。
    """
    if len(daily_returns) < 2:
        return 0.0
    ann_ret = float(daily_returns.mean() * 252)
    ann_vol = annualized_vol(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / (ann_vol / 100)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """索提诺比率 = (年化收益 - 无风险利率) / 年化下行波动率"""
    if len(daily_returns) < 2:
        return 0.0
    ann_ret = float(daily_returns.mean() * 252)
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        # 下行数据不足时无法计算有意义的下行波动率
        if ann_ret > risk_free_rate:
            return float('inf')
        return 0.0
    downside_vol = float(downside.std() * math.sqrt(252))
    if downside_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_vol


def calmar_ratio(annualized_return_pct: float, max_drawdown_pct: float) -> float:
    """卡玛比率 = 年化收益率 / 最大回撤

    年化收益率和最大回撤均以百分比形式输入（如 15 表示 15%）。
    """
    if max_drawdown_pct == 0:
        return 0.0
    return annualized_return_pct / max_drawdown_pct


# =====================================================================================
# 最大回撤
# =====================================================================================

def max_drawdown(portfolio_values: pd.Series) -> tuple[float, int]:
    """计算最大回撤及其持续天数。

    Parameters
    ----------
    portfolio_values : pd.Series
        策略每日净值序列。

    Returns
    -------
    tuple[float, int]
        (最大回撤百分比, 持续天数)。
    """
    if len(portfolio_values) == 0:
        return 0.0, 0

    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100

    max_dd = float(drawdown.min())

    if max_dd >= 0:
        return 0.0, 0

    # 找到最大回撤的起点和终点
    trough_idx = drawdown.idxmin()
    # 从开始到 trough_idx 之间的最大值位置
    peak_idx = portfolio_values[: portfolio_values.index.get_loc(trough_idx) + 1].idxmax()

    # 从 peak 到 trough 的交易日数
    if hasattr(peak_idx, 'date') and hasattr(trough_idx, 'date'):
        # datetime index 的情况
        peak_pos = portfolio_values.index.get_loc(peak_idx)
        trough_pos = portfolio_values.index.get_loc(trough_idx)
        duration = trough_pos - peak_pos
    else:
        try:
            peak_pos = list(portfolio_values.index).index(peak_idx)
            trough_pos = list(portfolio_values.index).index(trough_idx)
            duration = trough_pos - peak_pos
        except (ValueError, IndexError):
            duration = 0

    return abs(max_dd), duration


# =====================================================================================
# 贝塔 / 阿尔法
# =====================================================================================

def beta_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> tuple[float, float]:
    """计算贝塔和阿尔法。

    Parameters
    ----------
    strategy_returns : pd.Series
        策略日收益率。
    benchmark_returns : pd.Series
        基准日收益率。
    risk_free_rate : float, optional
        无风险利率（年化），默认 0。

    Returns
    -------
    tuple[float, float]
        (beta, alpha_annualized_pct)。
    """
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 0.0

    # 对齐索引
    common = strategy_returns.align(benchmark_returns, join='inner')
    s_ret = common[0]
    b_ret = common[1]

    if len(s_ret) < 2:
        return 0.0, 0.0

    # Use pandas cov/var (both ddof=1) for consistency
    cov = float(s_ret.cov(b_ret))
    var = float(b_ret.var())
    beta = cov / var if var > 0 else 0.0

    # 年化收益
    ann_strat = float(s_ret.mean() * 252)
    ann_bench = float(b_ret.mean() * 252)

    alpha = (ann_strat - risk_free_rate) - beta * (ann_bench - risk_free_rate)
    alpha_pct = alpha * 100

    return beta, alpha_pct


# =====================================================================================
# 信息比率 & 跟踪误差
# =====================================================================================

def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """信息比率 = 年化超额收益 / 年化跟踪误差"""
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    common = strategy_returns.align(benchmark_returns, join='inner')
    s_ret = common[0]
    b_ret = common[1]

    if len(s_ret) < 2:
        return 0.0

    excess = s_ret - b_ret
    tracking_error = float(excess.std() * math.sqrt(252))
    if tracking_error == 0:
        return 0.0
    annualized_excess = float(excess.mean() * 252)
    return annualized_excess / tracking_error


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """年化跟踪误差（%）"""
    if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    common = strategy_returns.align(benchmark_returns, join='inner')
    s_ret = common[0]
    b_ret = common[1]

    if len(s_ret) < 2:
        return 0.0

    excess = s_ret - b_ret
    return float(excess.std() * math.sqrt(252) * 100)


# =====================================================================================
# 胜率
# =====================================================================================

def win_rate(
    returns: pd.Series,
    freq: str = 'D',
    benchmark_returns: pd.Series | None = None,
) -> float:
    """统计胜率（策略战胜基准的比例）。

    Parameters
    ----------
    returns : pd.Series
        策略日收益率序列，index 应为 DatetimeIndex。
    freq : str, optional
        频率：'D'（日）, 'M'（月）, 'Q'（季）, 'Y'（年）。
    benchmark_returns : pd.Series, optional
        基准日收益率序列。提供时，胜率 = 策略超额收益 > 0 的比例（即战胜基准）。
        为 None 时，胜率 = 策略正收益的比例。

    Returns
    -------
    float
        胜率（%）。
    """
    if len(returns) == 0:
        return 0.0

    if freq != 'D':
        freq_map = {'M': 'ME', 'Q': 'QE', 'Y': 'YE'}
        offset = freq_map.get(freq, freq)
        strat_resampled = returns.resample(offset).apply(
            lambda x: (1 + x).prod() - 1,
        )

        if benchmark_returns is not None:
            # 分别复利计算策略和基准的周期收益，再相减得超额收益
            bench_resampled = benchmark_returns.resample(offset).apply(
                lambda x: (1 + x).prod() - 1,
            )
            combined = pd.concat(
                [strat_resampled, bench_resampled], axis=1, join='inner',
            )
            if len(combined) == 0:
                return 0.0
            excess = combined.iloc[:, 0] - combined.iloc[:, 1]
        else:
            excess = strat_resampled
    else:
        # 日胜率
        if benchmark_returns is not None:
            combined = pd.concat(
                [returns, benchmark_returns], axis=1, join='inner',
            )
            if len(combined) == 0:
                return 0.0
            excess = combined.iloc[:, 0] - combined.iloc[:, 1]
        else:
            excess = returns

    wins = (excess > 0).sum()
    total = len(excess)
    return float(wins / total * 100) if total > 0 else 0.0


# =====================================================================================
# 整合计算
# =====================================================================================

def compute_all_metrics(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series | None = None,
    trade_dates: list | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """计算全套业绩评价指标并返回字典。

    Parameters
    ----------
    portfolio_values : pd.Series
        策略每日净值序列，index 应为 DatetimeIndex。
    benchmark_values : pd.Series, optional
        基准每日净值序列（可选）。
    trade_dates : list, optional
        交易日列表（用于计算平均持仓天数）。
    risk_free_rate : float, optional
        无风险利率（年化），默认 0。

    Returns
    -------
    dict
        包含全套指标结果。
    """
    result: dict = {}

    # ── 基本信息 ──────────────────────────────────────────────────────
    if len(portfolio_values) == 0:
        return _empty_result()

    dates = portfolio_values.index
    result['start_date'] = dates[0] if hasattr(dates, 'dtype') else str(dates[0])
    result['end_date'] = dates[-1] if hasattr(dates, 'dtype') else str(dates[-1])
    result['total_days'] = len(portfolio_values)

    # 交易日年数
    trading_years = max(len(portfolio_values) / 252, 0)
    result['trading_years'] = trading_years

    # ── 策略收益 ──────────────────────────────────────────────────────
    cumulative_return_pct = float(
        (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    )
    result['strategy_cumulative_return_pct'] = cumulative_return_pct
    result['strategy_annualized_return_pct'] = annualized_return(
        cumulative_return_pct, trading_years,
    )

    # ── 日收益率序列 ────────────────────────────────────────────────
    daily_returns = portfolio_values.pct_change().dropna()

    # ── 基准收益 ──────────────────────────────────────────────────────
    if benchmark_values is not None and len(benchmark_values) > 0:
        bench_cumulative = float(
            (benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1) * 100
        )
        result['benchmark_cumulative_return_pct'] = bench_cumulative
        result['benchmark_annualized_return_pct'] = annualized_return(
            bench_cumulative, trading_years,
        )
        result['excess_return_pct'] = cumulative_return_pct - bench_cumulative
        bench_daily_returns = benchmark_values.pct_change().dropna()
    else:
        result['benchmark_cumulative_return_pct'] = 0.0
        result['benchmark_annualized_return_pct'] = 0.0
        result['excess_return_pct'] = 0.0
        bench_daily_returns = None

    # ── 风险 ──────────────────────────────────────────────────────────
    result['annualized_volatility'] = annualized_vol(daily_returns)

    # 下行波动率
    downside = daily_returns[daily_returns < 0]
    if len(downside) >= 2:
        result['downside_deviation'] = float(downside.std() * math.sqrt(252) * 100)
    else:
        result['downside_deviation'] = 0.0

    # 最大回撤
    dd_pct, dd_duration = max_drawdown(portfolio_values)
    result['max_drawdown_pct'] = dd_pct
    result['max_drawdown_duration'] = dd_duration

    # ── 胜率（基于超额收益：战胜基准） ──────────────────────────────
    if len(daily_returns) > 0:
        result['daily_win_rate'] = win_rate(daily_returns, 'D', benchmark_returns=bench_daily_returns)
        if isinstance(daily_returns.index, pd.DatetimeIndex):
            result['monthly_win_rate'] = win_rate(daily_returns, 'M', benchmark_returns=bench_daily_returns)
            result['quarterly_win_rate'] = win_rate(daily_returns, 'Q', benchmark_returns=bench_daily_returns)
            result['yearly_win_rate'] = win_rate(daily_returns, 'Y', benchmark_returns=bench_daily_returns)
        else:
            result['monthly_win_rate'] = 0.0
            result['quarterly_win_rate'] = 0.0
            result['yearly_win_rate'] = 0.0
    else:
        result['daily_win_rate'] = 0.0
        result['monthly_win_rate'] = 0.0
        result['quarterly_win_rate'] = 0.0
        result['yearly_win_rate'] = 0.0

    # ── 胜率（基于策略正收益） ────────────────────────────────────
    if len(daily_returns) > 0:
        result['daily_win_rate_abs'] = win_rate(daily_returns, 'D')
        if isinstance(daily_returns.index, pd.DatetimeIndex):
            result['monthly_win_rate_abs'] = win_rate(daily_returns, 'M')
            result['quarterly_win_rate_abs'] = win_rate(daily_returns, 'Q')
            result['yearly_win_rate_abs'] = win_rate(daily_returns, 'Y')
        else:
            result['monthly_win_rate_abs'] = 0.0
            result['quarterly_win_rate_abs'] = 0.0
            result['yearly_win_rate_abs'] = 0.0
    else:
        result['daily_win_rate_abs'] = 0.0
        result['monthly_win_rate_abs'] = 0.0
        result['quarterly_win_rate_abs'] = 0.0
        result['yearly_win_rate_abs'] = 0.0

    # ── 风险调整收益 ──────────────────────────────────────────────────
    result['sharpe_ratio'] = sharpe_ratio(daily_returns, risk_free_rate)
    result['sortino_ratio'] = sortino_ratio(daily_returns, risk_free_rate)
    result['calmar_ratio'] = calmar_ratio(
        result['strategy_annualized_return_pct'],
        result['max_drawdown_pct'],
    )

    # ── 相对基准 ──────────────────────────────────────────────────────
    if (
        benchmark_values is not None
        and len(benchmark_values) > 0
        and len(daily_returns) >= 2
    ):
        bench_returns = benchmark_values.pct_change().dropna()
        beta, alpha = beta_alpha(daily_returns, bench_returns, risk_free_rate)
        result['beta'] = beta
        result['alpha'] = alpha
        result['information_ratio'] = information_ratio(
            daily_returns, bench_returns,
        )
        result['tracking_error'] = tracking_error(daily_returns, bench_returns)
    else:
        result['beta'] = 0.0
        result['alpha'] = 0.0
        result['information_ratio'] = 0.0
        result['tracking_error'] = 0.0

    # ── 交易统计 ──────────────────────────────────────────────────────
    if trade_dates is not None and len(trade_dates) > 0:
        result['total_trades'] = len(trade_dates)
        if len(trade_dates) >= 2:
            # 排序后计算平均间隔天数
            sorted_dates = sorted(trade_dates)
            gaps = [
                (sorted_dates[i] - sorted_dates[i - 1]).days
                for i in range(1, len(sorted_dates))
            ]
            result['avg_holding_days'] = sum(gaps) / len(gaps)
        else:
            result['avg_holding_days'] = 0.0
    else:
        result['total_trades'] = 0
        result['avg_holding_days'] = 0.0

    return result


def _empty_result() -> dict:
    """返回空数据时的指标字典。"""
    return {
        'start_date': None,
        'end_date': None,
        'total_days': 0,
        'trading_years': 0.0,
        'strategy_cumulative_return_pct': 0.0,
        'strategy_annualized_return_pct': 0.0,
        'benchmark_cumulative_return_pct': 0.0,
        'benchmark_annualized_return_pct': 0.0,
        'excess_return_pct': 0.0,
        'annualized_volatility': 0.0,
        'downside_deviation': 0.0,
        'max_drawdown_pct': 0.0,
        'max_drawdown_duration': 0,
        'daily_win_rate': 0.0,
        'monthly_win_rate': 0.0,
        'quarterly_win_rate': 0.0,
        'yearly_win_rate': 0.0,
        'daily_win_rate_abs': 0.0,
        'monthly_win_rate_abs': 0.0,
        'quarterly_win_rate_abs': 0.0,
        'yearly_win_rate_abs': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'calmar_ratio': 0.0,
        'beta': 0.0,
        'alpha': 0.0,
        'information_ratio': 0.0,
        'tracking_error': 0.0,
        'total_trades': 0,
        'avg_holding_days': 0.0,
    }
