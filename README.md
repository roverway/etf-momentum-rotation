# ETF动量轮动策略

基于AKShare免费Sina API的ETF动量轮动策略回测系统。

## 策略说明

从ETF池中选取过去22个交易日涨幅最强的ETF持有，全部下跌则空仓（持有现金）。
每月末调仓，使用收盘价成交。

## 架构

- config.py — ETF池、回测参数、代码映射
- data.py — AKShare数据获取 + CSV缓存
- trading_calendar.py — A股交易日历
- portfolio.py — 组合跟踪（仓位、资金、PnL）
- strategy.py — 动量计算 + 信号生成
- backtest_engine.py — 回测引擎（逐日循环 + CSV输出）
- charts.py — Plotly图表（净值/回撤/持仓）
- signal_generator.py — 实盘信号生成器
- main.py — CLI入口

## 快速开始

### 环境准备

```bash
cd /path/to/trade
uv venv
uv pip install -r requirements.txt
```

### 运行回测

```bash
uv run python3 main.py backtest --start 2019-01-18 --end 2026-07-07
```

### 生成实盘信号

```bash
uv run python3 main.py signal
```

### 运行测试

```bash
uv run pytest tests/ -v
```

## 扩展ETF池

只需在 `config.py` 的 `ETF_POOL` 列表中添加新代码：

```python
ETF_POOL = [
    '513100.XSHG',  # 纳指ETF
    '159915.XSHE',  # 创业板ETF
    '518880.XSHG',  # 黄金ETF
    '512890.XSHG',  # 红利低波ETF
    # '159999.XSHE',  # ← 加这一行即可
]
```

代码映射、数据获取、回测引擎自动适配。

## 输出文件

运行回测后在 `backtest_results/` 目录生成：

- net_worth.csv — 每日净值快照
- trades.csv — 交易记录
- net_worth.html — 净值曲线图
- drawdown.html — 回撤图
- allocation.html — 持仓比例图

## 回测参数

- 起始日期: 可配置（--start）
- 初始资金: 100万（--cash）
- 佣金: 万2.5
- 滑点: 0.01%
- 空仓收益: 0%

## 关键决策

- 纯Python实现，不兼容米筐API
- 使用uv管理虚拟环境
- 数据缓存到local CSV（避免重复下载）
- 代码映射自动识别 .XSHG（上海）/ .XSHE（深圳）
