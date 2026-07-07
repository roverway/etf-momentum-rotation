"""Plotly 图表模块 — 净值曲线、水下回撤、持仓比例堆叠图。"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TEMPLATE = 'plotly_dark'


def _compute_drawdown(portfolio_values: pd.Series) -> pd.Series:
    """计算每日回撤百分比。

    Parameters
    ----------
    portfolio_values : pd.Series
        每日组合净值序列。

    Returns
    -------
    pd.Series
        回撤百分比（负值表示从峰值下跌的幅度）。
    """
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100
    return drawdown


def _prepare_df(net_worth_df: pd.DataFrame) -> pd.DataFrame:
    """Convert date column to datetime and sort."""
    df = net_worth_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _apply_theme(fig: go.Figure, **kwargs: Any) -> go.Figure:
    """Apply common layout settings."""
    pio.templates.default = _TEMPLATE
    fig.update_layout(
        hovermode='x unified',
        **kwargs,
    )
    return fig


def _write_html(fig: go.Figure, path: str) -> None:
    """写入 HTML，确保以 ``<!DOCTYPE html>`` 开头。"""
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    if not html.lstrip().startswith('<!DOCTYPE html>'):
        html = '<!DOCTYPE html>\n' + html
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_net_worth(net_worth_df: pd.DataFrame, path: str) -> str:
    """净值曲线图。

    Parameters
    ----------
    net_worth_df : pd.DataFrame
        包含 ``date``, ``portfolio_value`` 列的 DataFrame。
    path : str
        输出 HTML 路径。

    Returns
    -------
    str
        输出文件路径（与 *path* 相同）。
    """
    df = _prepare_df(net_worth_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='rgb(0, 200, 100)', width=2),
    ))

    _apply_theme(
        fig,
        title='Net Worth',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (CNY)',
    )
    _write_html(fig, path)
    return path


def plot_drawdown(net_worth_df: pd.DataFrame, path: str) -> str:
    """水下回撤图。

    Parameters
    ----------
    net_worth_df : pd.DataFrame
        包含 ``date``, ``portfolio_value`` 列的 DataFrame。
    path : str
        输出 HTML 路径。

    Returns
    -------
    str
        输出文件路径。
    """
    df = _prepare_df(net_worth_df)
    drawdown = _compute_drawdown(df['portfolio_value'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=drawdown,
        fill='tozeroy',
        mode='lines',
        name='Drawdown',
        line=dict(color='rgb(255, 80, 80)', width=1.5),
    ))

    _apply_theme(
        fig,
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        yaxis=dict(ticksuffix='%'),
    )
    _write_html(fig, path)
    return path


def plot_allocation(
    trades_df: pd.DataFrame,
    net_worth_df: pd.DataFrame,
    path: str,
) -> str:
    """持仓比例堆叠图。

    从 trades 重建每日持仓比例，用堆叠面积图展示。

    Parameters
    ----------
    trades_df : pd.DataFrame
        交易记录 DataFrame（提供各交易日的持仓变更）。
    net_worth_df : pd.DataFrame
        包含 ``date``, ``portfolio_value``, ``cash``, ``position_code``,
        ``shares``, ``price`` 列的 DataFrame。
    path : str
        输出 HTML 路径。

    Returns
    -------
    str
        输出文件路径。
    """
    df = _prepare_df(net_worth_df)

    total = df['portfolio_value'].values
    cash_pct = df['cash'].values / total * 100

    # 所有出现过的持仓代码
    codes = sorted(
        df.loc[df['position_code'] != '', 'position_code'].unique().tolist()
    )

    # 为每个代码计算每日持仓比例
    alloc: dict[str, pd.Series] = {'cash': cash_pct}
    for code in codes:
        mask = (df['position_code'] == code).values
        pos_val = df['shares'].values * df['price'].values
        alloc[code] = pd.Series(
            data=pos_val / total * 100,
            index=df.index,
        )
        # 不在该代码的日期设为 0
        alloc[code][~mask] = 0.0

    dates = df['date']

    fig = go.Figure()

    # Cash 层（放在最底）
    fig.add_trace(go.Scatter(
        x=dates,
        y=alloc['cash'],
        mode='none',
        stackgroup='one',
        name='Cash',
        fillcolor='rgba(100, 149, 237, 0.6)',
    ))

    # 每个代码一层
    for code in codes:
        fig.add_trace(go.Scatter(
            x=dates,
            y=alloc[code],
            mode='none',
            stackgroup='one',
            name=code,
        ))

    _apply_theme(
        fig,
        title='Portfolio Allocation',
        xaxis_title='Date',
        yaxis_title='Allocation (%)',
        yaxis=dict(ticksuffix='%', range=[0, 105]),
    )
    _write_html(fig, path)
    return path


# ---------------------------------------------------------------------------
# One-shot generation
# ---------------------------------------------------------------------------

def generate_all(
    net_worth_csv: str,
    trades_csv: str,
    output_dir: str,
) -> list[str]:
    """生成全部 3 个图表。

    Parameters
    ----------
    net_worth_csv : str
        net_worth.csv 文件路径。
    trades_csv : str
        trades.csv 文件路径。
    output_dir : str
        输出目录（自动创建）。

    Returns
    -------
    list[str]
        [净値曲线路径, 回撤图路径, 持仓比例图路径]。
    """
    os.makedirs(output_dir, exist_ok=True)

    net_worth_df = pd.read_csv(net_worth_csv, parse_dates=['date'])
    trades_df = pd.read_csv(trades_csv, parse_dates=['date'])

    paths = [
        plot_net_worth(net_worth_df, os.path.join(output_dir, 'net_worth.html')),
        plot_drawdown(net_worth_df, os.path.join(output_dir, 'drawdown.html')),
        plot_allocation(trades_df, net_worth_df, os.path.join(output_dir, 'allocation.html')),
    ]
    return paths
