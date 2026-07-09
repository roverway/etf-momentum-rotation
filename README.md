# ETF动量轮动策略

基于AKShare免费Sina API的ETF动量轮动策略回测系统。

## 策略说明

从ETF池中选取过去22个交易日**风险调整后动量（Sharpe式）**最强的ETF持有，全部下跌则空仓（持有现金）。

### 动量计算

动量基于 **收盘价（close）** 计算，采用带敏感度因子的波动率归一化动量（Sharpe 式）：

```
动量 = (最新收盘价 − 22天前收盘价) ÷ 22天前收盘价 ÷ 年化波动率 ^ λ
```

其中年化波动率 = 窗口内日收益率标准差 × √252。

相比传统涨幅算法，这种计算方式确保选出的是**"涨得稳"**的ETF，而不是**"涨得疯"**的高波动品种。一只稳步爬升的ETF即使涨幅绝对值较低，也可能在风险调整后排名优于暴涨暴跌的标的。

#### 波动率回看期

动量涨幅仍使用 22 个交易日（`CHECK_RANGE`），但波动率使用独立的更长周期（`VOL_CHECK_RANGE`，默认 **60 个交易日 ≈ 3 个月**）计算。

长周期波动率代表资产的**固有波动基因**，而非短期噪音。即使 ETF 近期波动剧烈，若其长期波动率稳定，也不会被错误惩罚。

#### 敏感度因子 λ

`VOLATILITY_LAMBDA`（默认 **1.0**）控制波动率对动量的惩罚力度：

| λ | 效果 | 适用场景 |
|---|------|---------|
| 1.0 | 完全风险调整（原行为） | 追求稳定的风险调整收益 |
| 0.5 | 中间态——保留对龙头的敏感度 | 平衡稳定性和爆发力 |
| 0.0 | 退化为原始涨幅策略 | 纯追逐趋势，不惩罚波动 |

公式：`动量 = 涨幅 / (波动率 ^ λ)`。可在 `config.py` 中调整 `VOL_CHECK_RANGE` 和 `VOLATILITY_LAMBDA` 参数。

### 调仓阈值滤波

为防止频繁换仓导致利润被佣金和滑点吃光，引入了可配置的**缓冲阈值**：

```
只有当 新标的动量 − 当前持仓动量 > REBALANCE_THRESHOLD (默认1%) 时才触发换仓
否则继续持有不动
```

可在 `config.py` 中调整 `REBALANCE_THRESHOLD` 参数。

### 调仓时机

每个交易日收盘后，以当日收盘价成交。

每日循环流程：
1. 进入交易日 → 2. 用截至当日的数据计算全部ETF动量分 → 3. 对比当前持仓各标的动量，判断是否超过阈值 → 4. 若需换仓则取当日收盘价执行买卖 → 5. 记录成交

## 架构

- config.py — ETF池、回测参数、代码映射
- data.py — AKShare数据获取 + CSV缓存
- trading_calendar.py — A股交易日历
- portfolio.py — 组合跟踪（仓位、资金、PnL）
- strategy.py — 动量计算 + 信号生成（含向量化信号矩阵 compute_all_momentum_signals）
- backtest_engine.py — 向量化回测引擎（Pandas rolling + shift 加速，CSV输出 + 业绩指标 + 下一日建议）
- metrics.py — 绩效指标计算（夏普/索提诺/回撤/胜率等）
- charts.py — Plotly图表 → 单HTML报告（业绩表格 + 净值/回撤/持仓子图）
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
- report.html — 完整回测报告（含业绩指标表格、净值曲线、回撤图、持仓比例，4行子图合并为单HTML）

## 回测参数

回测参数通过 `BacktestConfig` 数据类或 CLI 参数配置，默认值如下：

- 起始日期: 可配置（--start），默认 2014-01-01
- 结束日期: 可配置（--end），默认今天
- 初始资金: 100万（--cash）
- 佣金: 万2（可配置，config.py commission_rate）
- 滑点: 无（可配置，config.py slippage_rate）
- 现金收益率: 0%（年化，可配置，config.py cash_return_rate）
- 基准: 沪深300指数（可配置，config.py benchmark_code）

## 特色功能

- **业绩指标输出**: 回测完成后自动计算并打印全套绩效指标
  - 累计/年化收益率（策略+基准）
  - 日/月/季/年胜率
  - Alpha/Beta/Sharpe/Sortino/信息比率
  - 年化波动率/下行波动率/跟踪误差
  - 最大回撤及持续时间
- **年度收益率明细**: 每年策略 vs 基准 vs 超额收益对比表
- **次日操作建议**: 回测结束后自动输出下一交易日信号
- **配置化参数**: BacktestConfig 支持自定义佣金/滑点/现金收益率/基准代码
- **波动率参数独立可配**: VOL_CHECK_RANGE（波动率回看期）、VOLATILITY_LAMBDA（波动率敏感度）
- **向量化回测加速**: 基于 Pandas rolling / shift 矩阵运算替代逐日 for 循环，5000 交易日回测耗时 < 1 秒

## 关键决策

- 纯Python实现，不兼容米筐API
- 每日调仓（向量化回测：信号矩阵预计算 + 简化持仓模拟循环，5000天 < 1秒）
- 绩效指标通过 metrics.py 独立模块计算
- 图表统一输出为单HTML report.html
- 使用uv管理虚拟环境
- 数据缓存到local CSV（避免重复下载）
- 代码映射自动识别 .XSHG（上海）/ .XSHE（深圳）
