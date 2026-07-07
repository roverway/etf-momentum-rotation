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


def _write_html(fig: go.Figure, path: str) -> None:
    """写入 HTML，确保以 ``<!DOCTYPE html>`` 开头。"""
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    if not html.lstrip().startswith('<!DOCTYPE html>'):
        html = '<!DOCTYPE html>\n' + html
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Metrics Table
# ---------------------------------------------------------------------------

# ── Category groupings for multi-column table layout ──
# Each tuple: (display_name, [metric_key, ...])
_CATEGORIES: list[tuple[str, list[str]]] = [
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

# Column groups: arrange categories into 3 side-by-side blocks (6 cols total)
_COLUMN_GROUPS: list[list[tuple[str, list[str]]]] = [
    [_CATEGORIES[0], _CATEGORIES[1]],   # 基础信息 + 收益
    [_CATEGORIES[2], _CATEGORIES[3]],   # 风险 + 胜率
    [_CATEGORIES[4], _CATEGORIES[5], _CATEGORIES[6]],  # 风险调整收益 + 相对基准 + 交易统计
]


def _metrics_table(metrics: dict) -> go.Table:
    """构建多列分组业绩指标表格。

    将指标按类别分组为3列并排显示，每列由 (label, value) 对组成，
    类别名称作为行内标题。
    布局: 6 列 → (指标|值) × 3 组。

    Parameters
    ----------
    metrics : dict
        由 metrics.compute_all_metrics() 返回的指标字典。

    Returns
    -------
    go.Table
        Plotly Table 对象，适合放入 subplot 的第一行。
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

    # ── 2. 为每组构建行条目 ──
    # 每条: ('header', cat_name) 或 ('metric', label, value) 或 ('empty',)
    group_rows: list[list[tuple]] = [[], [], []]
    for g_idx, group in enumerate(_COLUMN_GROUPS):
        for cat_name, cat_keys in group:
            available = [metric_map[k] for k in cat_keys if k in metric_map]
            if not available:
                continue
            group_rows[g_idx].append(('header', cat_name))
            for label, value in available:
                group_rows[g_idx].append(('metric', label, value))

    max_rows = max(len(r) for r in group_rows) if any(group_rows) else 0
    for rows in group_rows:
        while len(rows) < max_rows:
            rows.append(('empty',))

    # ── 3. 构建 per-column cell arrays ──
    col_values: list[list[str]] = [[], [], [], [], [], []]
    col_fills: list[list[str]] = [[], [], [], [], [], []]
    row_norm = 0  # normalized row counter for alternating shading

    for row_i in range(max_rows):
        any_metric = any(g[row_i][0] == 'metric' for g in group_rows if row_i < len(g))
        if any_metric:
            row_norm += 1

        for g_idx in range(3):
            c_label = g_idx * 2
            c_value = g_idx * 2 + 1
            entry = group_rows[g_idx][row_i]

            if entry[0] == 'header':
                col_values[c_label].append(f'<b>{entry[1]}</b>')
                col_values[c_value].append('')
                col_fills[c_label].append('#e8edf3')  # subtle blue-gray header
                col_fills[c_value].append('#e8edf3')
            elif entry[0] == 'metric':
                col_values[c_label].append(entry[1])
                col_values[c_value].append(entry[2])
                bg = '#fafafa' if (row_norm % 2 == 0) else '#ffffff'
                col_fills[c_label].append(bg)
                col_fills[c_value].append(bg)
            else:  # empty
                col_values[c_label].append('')
                col_values[c_value].append('')
                col_fills[c_label].append('#ffffff')
                col_fills[c_value].append('#ffffff')

    return go.Table(
        header=dict(
            values=['<b>指标</b>', '<b>值</b>'] * 3,
            font=dict(size=12, color='#333333'),
            fill_color='#f0f0f0',
            align='left',
            height=28,
        ),
        cells=dict(
            values=col_values,
            font=dict(size=11, color=['#333333', '#006600'] * 3),
            fill_color=col_fills,
            align='left',
            height=22,
            line=dict(color='#e0e0e0', width=1),
        ),
        columnwidth=[125, 90] * 3,
    )


# ---------------------------------------------------------------------------
# Report Figure (subplots)
# ---------------------------------------------------------------------------

def _build_report_figure(
    net_worth_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    metrics: dict,
) -> go.Figure:
    """构建组合页面 Figure。

    使用 make_subplots:
    - rows=4, cols=1 (指标表格 + 净值 + 回撤 + 持仓)
    - row_heights=[0.15, 0.35, 0.25, 0.25]

    Parameters
    ----------
    net_worth_df : pd.DataFrame
        包含 ``date``, ``portfolio_value``, ``cash``, ``position_code``,
        ``shares``, ``price`` 列的 DataFrame。
    trades_df : pd.DataFrame
        交易记录 DataFrame（暂仅占位，持仓数据从 net_worth_df 提取）。
    metrics : dict
        业绩指标字典。

    Returns
    -------
    go.Figure
        包含全部子图的组合 Figure。
    """
    # ── 1. 准备数据 ──────────────────────────────────────────────────────
    df = _prepare_df(net_worth_df)

    # ── 2. 创建 subplot 布局 ─────────────────────────────────────────────
    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.18, 0.34, 0.24, 0.24],
        subplot_titles=('', '', '', ''),
        vertical_spacing=0.07,
        specs=[
            [{'type': 'domain'}],
            [{'type': 'scatter'}],
            [{'type': 'scatter'}],
            [{'type': 'scatter'}],
        ],
    )

    # ── 3. Row 1: 业绩指标表格 ────────────────────────────────────────────
    table = _metrics_table(metrics)
    fig.add_trace(table, row=1, col=1)

    # ── 4. Row 2: 净值曲线 ────────────────────────────────────────────────
    # 策略净值
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['portfolio_value'],
            mode='lines',
            name='策略净值',
            line=dict(color='rgb(0, 200, 100)', width=2),
        ),
        row=2,
        col=1,
    )

    # 基准曲线（如果有 benchmark_value 列）
    if 'benchmark_value' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['benchmark_value'],
                mode='lines',
                name='基准',
                line=dict(color='rgb(255, 180, 50)', width=1.5, dash='dash'),
            ),
            row=2,
            col=1,
        )

    # ── 5. Row 3: 回撤曲线 ────────────────────────────────────────────────
    drawdown = _compute_drawdown(df['portfolio_value'])
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            name='回撤',
            line=dict(color='rgb(255, 80, 80)', width=1.5),
        ),
        row=3,
        col=1,
    )

    # ── 6. Row 4: 持仓比例堆叠图 ──────────────────────────────────────────
    total = df['portfolio_value'].values
    cash_pct = df['cash'].values / total * 100

    # 所有出现过的持仓代码（填充 NaN → 空字符串，避免排序时 str vs float 错误）
    pos_codes = df['position_code'].fillna('')
    codes = sorted(
        pos_codes.loc[pos_codes != ''].unique().tolist()
    )

    # Cash 层
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=cash_pct,
            mode='none',
            stackgroup='one',
            name='Cash',
            fillcolor='rgba(100, 149, 237, 0.6)',
        ),
        row=4,
        col=1,
    )

    # 每个代码一层
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
            row=4,
            col=1,
        )

    # ── 7. 布局 ────────────────────────────────────────────────────────────
    _apply_theme(
        fig,
        title=dict(
            text='ETF动量轮动策略 - 回测报告',
            x=0.5,
            xanchor='center',
            font=dict(size=20),
        ),
        height=1400,
        # 移除表格的轴线（Row 1 是 domain 类型，不需要轴）
        xaxis1=dict(visible=False),
        yaxis1=dict(visible=False),
        # 净值
        xaxis2=dict(title='日期'),
        yaxis2=dict(title='组合净值 (CNY)'),
        # 回撤
        xaxis3=dict(title='日期'),
        yaxis3=dict(title='回撤 (%)', ticksuffix='%'),
        # 持仓
        xaxis4=dict(title='日期'),
        yaxis4=dict(title='持仓比例 (%)', ticksuffix='%', range=[0, 105]),
        # 图例（位置）
        legend=dict(
            orientation='h',
            y=1.02,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
        ),
        margin=dict(l=80, r=40, t=120, b=60),
    )

    # 添加子图标题（用 annotation 替代 subplot_titles 以精细控制位置）
    subtitle_y = [0.94, 0.72, 0.46, 0.22]
    subtitle_text = ['业绩指标', '净值曲线', '回撤曲线', '持仓比例']
    for i, (sy, st) in enumerate(zip(subtitle_y, subtitle_text), start=1):
        fig.add_annotation(
            x=0.5,
            y=sy,
            xref='paper',
            yref='paper',
            text=f'<b>{st}</b>',
            showarrow=False,
            font=dict(size=14),
            xanchor='center',
            yanchor='bottom',
        )

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
) -> str:
    """生成完整业绩报告 HTML（单文件）。

    页面结构:
    1. 标题: "ETF动量轮动策略 - 回测报告"
    2. 业绩指标表格 (plotly table)
    3. 净值曲线 (subplot 2)
    4. 回撤曲线 (subplot 3)
    5. 持仓比例堆叠图 (subplot 4)

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

    Returns
    -------
    str
        ``output_path``（与传入值相同）。
    """
    net_worth_df = pd.read_csv(net_worth_csv)
    trades_df = pd.read_csv(trades_csv)

    # 确保 parse_dates 在读取层面就被处理，保留所有行
    # （不传 parse_dates，由 _prepare_df 统一处理）

    fig = _build_report_figure(net_worth_df, trades_df, metrics)
    _write_html(fig, output_path)
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
