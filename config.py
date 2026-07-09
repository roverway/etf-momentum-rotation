"""策略配置模块 — ETF池、回测参数、交易所代码映射。"""

from dataclasses import dataclass, field

# =====================================================================================
# ETF 池 & 回测常数
# =====================================================================================

ETF_POOL = [
    "513100.XSHG",  # 纳指ETF
    "159915.XSHE",  # 创业板ETF
    "518880.XSHG",  # 黄金ETF
    "512890.XSHG",  # 红利低波ETF
]

CHECK_RANGE = 22           # 交易日；22 ≈ 1 个月
REBALANCE_THRESHOLD = 0.01  # 1% 动量差异阈值，防止频繁换仓


# =====================================================================================
# 回测配置 (数据类)
# =====================================================================================


@dataclass
class BacktestConfig:
    start_date: str = "2014-01-01"
    end_date: str = ""  # 空字符串 ≡ 今天
    initial_cash: float = 1_000_000
    commission_rate: float = 0.0002  # 万2
    slippage_rate: float = 0.001  # 千分之一
    cash_return_rate: float = 0.0  # 现金收益率（年化）
    benchmark_code: str = "000300.XSHG"  # 基准代码，沪深300


# =====================================================================================
# 新浪财经代码映射
# =====================================================================================

_SUFFIX_MAP = {
    ".XSHG": "sh",
    ".XSHE": "sz",
}


def map_to_sina_code(code: str) -> str:
    """将标准 ETF 代码（如 ``513100.XSHG``）转为新浪财经格式（如 ``sh513100``）。

    Parameters
    ----------
    code : str
        带后缀的 ETF 代码，如 ``'513100.XSHG'`` 或 ``'159915.XSHE'``。

    Returns
    -------
    str
        新浪前缀+纯数字，如 ``'sh513100'``。
    """
    for suffix, prefix in _SUFFIX_MAP.items():
        if code.endswith(suffix):
            return prefix + code[: -len(suffix)]
    raise ValueError(f"未知交易所后缀: {code}")
