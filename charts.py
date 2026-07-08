"""Plotly 图表模块 — 完整回测报告（单文件 HTML）。

提供:
  - generate_report: 生成单文件 HTML 报告（含业绩指标表格 + 净值曲线 + 回撤 + 持仓堆叠）
  - generate_all: 向后兼容的旧入口（内部调用 generate_report）
  - plot_net_worth / plot_drawdown / plot_allocation: 单图入口（保留兼容）
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# 指标定义: (metrics_key, 中文标签, format_string)
# 按显示顺序排列
# ---------------------------------------------------------------------------

_METRIC_DEFS: list[tuple[str, str, str]] = [
    # ── 基础信息 ──
    ('start_date', '开始日期', '{}'),
    ('end_date', '结束日期', '{}'),
    ('total_days', '回测天数', '{}天'),
    ('trading_years', '交易年限', '{:.2f}年'),
    # ── 收益类 ──
    ('strategy_cumulative_return_pct', '策略累计收益率', '{:+.2f}%'),
    ('strategy_annualized_return_pct', '策略年化收益率', '{:+.2f}%'),
    ('benchmark_cumulative_return_pct', '基准累计收益率', '{:+.2f}%'),
    ('benchmark_annualized_return_pct', '基准年化收益率', '{:+.2f}%'),
    ('excess_return_pct', '超额收益率', '{:+.2f}%'),
    # ── 风险类 ──
    ('annualized_volatility', '年化波动率', '{:.2f}%'),
    ('downside_deviation', '下行波动率', '{:.2f}%'),
    ('max_drawdown_pct', '最大回撤', '-{:.2f}%'),
    ('max_drawdown_duration', '最大回撤持续天数', '{}天'),
    # ── 胜率类（战胜基准） ──
    ('daily_win_rate', '日胜率(超基准)', '{:.2f}%'),
    ('monthly_win_rate', '月胜率(超基准)', '{:.2f}%'),
    ('quarterly_win_rate', '季胜率(超基准)', '{:.2f}%'),
    ('yearly_win_rate', '年胜率(超基准)', '{:.2f}%'),
    # ── 胜率类（策略正收益） ──
    ('daily_win_rate_abs', '日胜率(正收益)', '{:.2f}%'),
    ('monthly_win_rate_abs', '月胜率(正收益)', '{:.2f}%'),
    ('quarterly_win_rate_abs', '季胜率(正收益)', '{:.2f}%'),
    ('yearly_win_rate_abs', '年胜率(正收益)', '{:.2f}%'),
    # ── 风险调整收益 ──
    ('sharpe_ratio', '夏普比率', '{:.2f}'),
    ('sortino_ratio', '索提诺比率', '{:.2f}'),
    ('calmar_ratio', '卡尔玛比率', '{:.2f}'),
    # ── 相对基准 ──
    ('alpha', 'Alpha', '{:.2f}'),
    ('beta', 'Beta', '{:.2f}'),
    ('information_ratio', '信息比率', '{:.2f}'),
    ('tracking_error', '跟踪误差', '{:.2f}%'),
    # ── 交易统计 ──
    ('total_trades', '总交易次数', '{}'),
    ('avg_holding_days', '平均持仓天数', '{:.1f}天'),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TEMPLATE = 'plotly_white'


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
    """Convert date column to datetime and sort.

    ！！！不过滤任何行，确保完整区间。！！！
    """
    df = net_worth_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # 丢弃 NaT 行（无法解析的日期），但保留所有有效行
    df = df.dropna(subset=['date'])
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


def _write_html(fig: go.Figure, path: str, metrics_html: str = "") -> None:
    """写入包含指标卡片和 Plotly 图表的完整 HTML 报告。"""
    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETF动量轮动策略 - 回测报告</title>
<style>
body {{ margin:0; padding:0; background:#f5f7fa; font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }}
{_METRICS_CSS}
</style>
</head>
<body>
{metrics_html}
{plotly_html}
</body>
</html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Metrics HTML cards
# ---------------------------------------------------------------------------

# 类别分组定义: (类别名, [metric_key, ...])
_METRIC_CATEGORIES: list[tuple[str, list[str]]] = [
    ('基础信息', ['start_date', 'end_date', 'total_days', 'trading_years']),
    ('收益', [
        'strategy_cumulative_return_pct', 'strategy_annualized_return_pct',
        'benchmark_cumulative_return_pct', 'benchmark_annualized_return_pct',
        'excess_return_pct',
    ]),
    ('风险', [
        'annualized_volatility', 'downside_deviation',
        'max_drawdown_pct', 'max_drawdown_duration',
    ]),
    ('胜率', [
        'daily_win_rate', 'monthly_win_rate', 'quarterly_win_rate',
        'yearly_win_rate', 'daily_win_rate_abs', 'monthly_win_rate_abs',
        'quarterly_win_rate_abs', 'yearly_win_rate_abs',
    ]),
    ('风险调整收益', ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']),
    ('相对基准', ['alpha', 'beta', 'information_ratio', 'tracking_error']),
    ('交易统计', ['total_trades', 'avg_holding_days']),
]

_METRICS_CSS = """
.metrics-container {
  max-width: 1200px;
  margin: 0 auto 30px;
  padding: 20px 24px 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}
.metric-group {
  background: #ffffff;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 12px 16px 16px;
  flex: 1 1 200px;
  min-width: 180px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.group-title {
  font-size: 14px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 8px 0;
  padding-bottom: 6px;
  border-bottom: 2px solid #409eff;
}
.metric-table {
  width: 100%;
  border-collapse: collapse;
}
.metric-table tr {
  border-bottom: 1px solid #f0f0f0;
}
.metric-table tr:last-child {
  border-bottom: none;
}
.metric-table td {
  padding: 4px 0;
  font-size: 13px;
  line-height: 1.6;
}
.metric-table td.label {
  color: #606266;
}
.metric-table td.value {
  text-align: right;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: #303133;
}
.metric-table td.value.positive {
  color: #67c23a;
}
.metric-table td.value.negative {
  color: #f56c6c;
}
"""


def _build_metrics_html(metrics: dict) -> str:
    """构建业绩指标卡片 HTML（仅 body 内容，不含 CSS）。

    Parameters
    ----------
    metrics : dict
        由 metrics.compute_all_metrics() 返回的指标字典。

    Returns
    -------
    str
        指标卡片组的 HTML 字符串（<div class="metrics-container">...</div>）。
    """
    # ── 1. 构建 key -> (label, formatted_value) 查找表 ──
    metric_map: dict[str, tuple[str, str]] = {}
    for key, label, fmt in _METRIC_DEFS:
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            try:
                formatted = fmt.format(val)
            except (ValueError, TypeError):
                formatted = str(val)
            metric_map[key] = (label, formatted)

    # ── 2. 构建每个类别的 HTML ──
    parts: list[str] = []
    for cat_name, cat_keys in _METRIC_CATEGORIES:
        rows: list[str] = []
        for k in cat_keys:
            if k in metric_map:
                label, value = metric_map[k]
                val_class = ''
                if value.startswith('+'):
                    val_class = ' positive'
                elif value.startswith('-'):
                    val_class = ' negative'
                rows.append(
                    f'          <tr>'
                    f'<td class="label">{label}</td>'
                    f'<td class="value{val_class}">{value}</td>'
                    f'</tr>'
                )
        if rows:
            parts.append('        <div class="metric-group">')
            parts.append(f'          <h3 class="group-title">{cat_name}</h3>')
            parts.append('          <table class="metric-table">')
            parts.extend(rows)
            parts.append('          </table>')
            parts.append('        </div>')

    if not parts:
        return ''

    parts.insert(0, '      <div class="metrics-container">')
    parts.append('      </div>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Report Figure (subplots)
# ---------------------------------------------------------------------------

def _build_report_figure(
    net_worth_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: dict,
    benchmark_series: pd.Series | None = None,
) -> go.Figure:
    """构建组合页面 Figure — 3 行子图（净值 + 回撤 + 持仓）。

    业绩指标改用 HTML 卡片（由 _build_metrics_html 生成），不再内嵌为 subplot。

    Parameters
    ----------
    net_worth_df : pd.DataFrame
        包含 ``date``, ``portfolio_value``, ``cash``, ``position_code``,
        ``shares``, ``price`` 列的 DataFrame。
    trades_df : pd.DataFrame
        交易记录 DataFrame（暂仅占位，持仓数据从 net_worth_df 提取）。
    metrics : dict
        业绩指标字典。
    benchmark_series : pd.Series | None
        基准指数净值序列（DatetimeIndex），可选。

    Returns
    -------
    go.Figure
        包含 3 行子图的组合 Figure。
    """
    df = _prepare_df(net_worth_df)

    # ── subplot 布局: 3 rows ──
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.38, 0.30, 0.26],
        subplot_titles=('净值曲线', '回撤曲线', '持仓比例'),
        vertical_spacing=0.10,
        specs=[
            [{'type': 'scatter'}],
            [{'type': 'scatter'}],
            [{'type': 'scatter'}],
        ],
    )

    # ── Row 1: 净值曲线 ──
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['portfolio_value'],
            mode='lines',
            name='策略净值',
            line=dict(color='#2ecc71', width=2),
        ),
        row=1, col=1,
    )

    # 基准曲线（作为独立参数传入）
    if benchmark_series is not None and not benchmark_series.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_series.index,
                y=benchmark_series.values,
                mode='lines',
                name='沪深300',
                line=dict(color='#FF8C00', width=1.5, dash='dash'),
            ),
            row=1, col=1,
        )

    # ── Row 2: 回撤曲线 ──
    drawdown = _compute_drawdown(df['portfolio_value'])
    min_dd = drawdown.min()
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            name='回撤',
            line=dict(color='#e74c3c', width=1.5),
        ),
        row=2, col=1,
    )

    # ── Row 3: 持仓比例堆叠图 ──
    total = df['portfolio_value'].values
    cash_pct = df['cash'].values / total * 100
    pos_codes = df['position_code'].fillna('')
    codes = sorted(pos_codes.loc[pos_codes != ''].unique().tolist())

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=cash_pct,
            mode='none',
            stackgroup='one',
            name='Cash',
            fillcolor='rgba(100, 149, 237, 0.6)',
        ),
        row=3, col=1,
    )

    for code in codes:
        mask = (df['position_code'] == code).values
        pos_val = df['shares'].values * df['price'].values
        alloc = pd.Series(data=pos_val / total * 100, index=df.index)
        alloc[~mask] = 0.0
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=alloc,
                mode='none',
                stackgroup='one',
                name=code,
            ),
            row=3, col=1,
        )

    # ── 全局布局 ──
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text='ETF动量轮动策略 - 回测报告',
            x=0.5,
            xanchor='center',
            font=dict(size=20),
        ),
        height=1200,
        hovermode='x unified',
        font=dict(family='Arial, sans-serif', size=12),
        legend=dict(
            orientation='h',
            y=1.04,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=9),
        ),
        margin=dict(l=80, r=40, t=160, b=60),
    )

    # ── 轴属性 ──
    fig.update_xaxes(title='日期', row=1, col=1)
    fig.update_yaxes(title='组合净值 (CNY)', row=1, col=1)

    fig.update_xaxes(title='日期', row=2, col=1)
    fig.update_yaxes(title='回撤 (%)', ticksuffix='%', range=[min_dd - 3, 3], row=2, col=1)

    fig.update_xaxes(title='日期', row=3, col=1)
    fig.update_yaxes(title='持仓比例 (%)', ticksuffix='%', range=[0, 105], row=3, col=1)

    # ── 区间选择器 (所有 x 轴) ──
    selector_buttons = [
        dict(count=1, label='1m', step='month', stepmode='backward'),
        dict(count=6, label='6m', step='month', stepmode='backward'),
        dict(count=1, label='1y', step='year', stepmode='backward'),
        dict(count=3, label='3y', step='year', stepmode='backward'),
        dict(step='all', label='全部'),
    ]
    selector_style = dict(bgcolor='#f0f0f0', activecolor='#c8d8e8')
    for r in (1, 2, 3):
        fig.update_xaxes(rangeselector=dict(buttons=selector_buttons, **selector_style), row=r, col=1)

    # ── 同步 x 轴缩放（不启用 rangeslider，因其微缩图会与回撤子图重叠）──
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)

    return fig


# ---------------------------------------------------------------------------
# Public plot functions (single-chart, backward compatible)
# ---------------------------------------------------------------------------

def plot_net_worth(net_worth_df: pd.DataFrame, path: str) -> str:
    """净值曲线图。（保留向后兼容）

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
    """水下回撤图。（保留向后兼容）

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
    """持仓比例堆叠图。（保留向后兼容）

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

def generate_report(
    net_worth_csv: str,
    trades_csv: str,
    metrics: dict,
    output_path: str,
    benchmark_series: pd.Series | None = None,
) -> str:
    """生成完整业绩报告 HTML（单文件）。

    页面结构:
    1. 业绩指标卡片组（HTML，按类别分组）
    2. 净值曲线（含策略净值 + 可选基准指数）
    3. 回撤曲线
    4. 持仓比例堆叠图

    Parameters
    ----------
    net_worth_csv : str
        net_worth.csv 文件路径。
    trades_csv : str
        trades.csv 文件路径。
    metrics : dict
        业绩指标字典（由 ``metrics.compute_all_metrics()`` 返回）。
    output_path : str
        输出 HTML 文件路径。
    benchmark_series : pd.Series | None
        基准指数净值序列（DatetimeIndex），可选。

    Returns
    -------
    str
        ``output_path``（与传入值相同）。
    """
    net_worth_df = pd.read_csv(net_worth_csv)
    trades_df = pd.read_csv(trades_csv)

    metrics_html = _build_metrics_html(metrics)
    fig = _build_report_figure(
        net_worth_df, trades_df, metrics,
        benchmark_series=benchmark_series,
    )
    _write_html(fig, output_path, metrics_html=metrics_html)
    return output_path


def generate_all(
    net_worth_csv: str,
    trades_csv: str,
    output_dir: str,
    metrics: dict | None = None,
) -> list[str]:
    """生成完整报告（单文件）。

    向后兼容包装器：内部调用 ``generate_report``，返回包含单个
    文件路径的列表。

    Parameters
    ----------
    net_worth_csv : str
        net_worth.csv 文件路径。
    trades_csv : str
        trades.csv 文件路径。
    output_dir : str
        输出目录（自动创建）。
    metrics : dict | None
        业绩指标字典（可选，不提供时传空 dict）。

    Returns
    -------
    list[str]
        只含一个元素：[\"report.html\" 的完整路径]。
    """
    if metrics is None:
        metrics = {}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'report.html')
    generate_report(net_worth_csv, trades_csv, metrics, output_path)
    return [output_path]
