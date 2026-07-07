"""data.py — ETF 数据获取 + CSV 缓存模块。

提供:
  - fetch_single_etf: 通过 AKShare 获取单只 ETF 日线数据。
  - load_all_etf_data: 批量加载 ETF 数据，带本地 CSV 缓存。
"""

from __future__ import annotations

import logging
import os

import akshare as ak
import pandas as pd

from config import BENCHMARK_CODE, map_to_sina_code

logger = logging.getLogger(__name__)


def fetch_single_etf(sina_code: str) -> pd.DataFrame:
    """从 AKShare 获取单只 ETF 日线数据（全量历史）。

    Parameters
    ----------
    sina_code : str
        新浪格式代码，如 ``'sh513100'``。

    Returns
    -------
    pd.DataFrame
        包含列: date, open, high, low, close, volume, amount。
        网络失败时返回空 DataFrame。
    """
    try:
        df = ak.fund_etf_hist_sina(symbol=sina_code)
        return df
    except Exception as e:
        logger.warning("⚠ 下载 %s 失败: %s", sina_code, e)
        return pd.DataFrame()


def load_all_etf_data(
    etf_codes: list[str],
    cache_dir: str = "data_cache/",
) -> dict[str, pd.DataFrame]:
    """加载所有 ETF 数据（从缓存或 AKShare 下载后缓存）。

    对每个 ETF 代码：
      1. 通过 ``map_to_sina_code()`` 转为新浪格式；
      2. 检查 ``{cache_dir}/{sina_code}.csv`` 是否存在；
      3. 缓存存在 → 读取 CSV；不存在 → 下载并保存为 CSV；
      4. 以原始代码为 key 存入结果字典。

    Parameters
    ----------
    etf_codes : list[str]
        RiceQuant 格式代码列表，如 ``['513100.XSHG', '159915.XSHE']``。
    cache_dir : str, optional
        缓存目录路径，默认 ``'data_cache/'``。

    Returns
    -------
    dict[str, pd.DataFrame]
        原始代码 → DataFrame 的映射。
    """
    os.makedirs(cache_dir, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}

    for code in etf_codes:
        sina_code = map_to_sina_code(code)
        cache_path = os.path.join(cache_dir, f"{sina_code}.csv")

        if os.path.isfile(cache_path):
            df = pd.read_csv(cache_path)
            # 将 CSV 中字符串格式的日期还原为 datetime.date
            df["date"] = pd.to_datetime(df["date"]).dt.date
        else:
            df = fetch_single_etf(sina_code)
            df.to_csv(cache_path, index=False)

        if 'volume' in df.columns:
            zero_vol = df[df['volume'] == 0]
            if not zero_vol.empty:
                dates_str = ", ".join(str(d) for d in zero_vol['date'].tolist())
                logger.warning("⚠ %s 存在零成交量日: %s", sina_code, dates_str)

        result[code] = df

    return result


def fetch_benchmark_data(
    benchmark_code: str = '000300.XSHG',
    cache_dir: str = "data_cache/",
) -> pd.DataFrame:
    """获取基准指数（沪深300）日线数据并缓存。

    使用 ``ak.stock_zh_index_daily(symbol='sh000300')`` 获取。
    返回 DataFrame 包含 date, close 列。

    Parameters
    ----------
    benchmark_code : str, optional
        基准代码，默认 ``'000300.XSHG'``（沪深300）。
    cache_dir : str, optional
        缓存目录路径，默认 ``'data_cache/'``。

    Returns
    -------
    pd.DataFrame
        包含 date（datetime.date）, close 列的 DataFrame。
        网络失败时返回空 DataFrame。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{benchmark_code}.csv")

    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df

    sina_code = map_to_sina_code(benchmark_code)
    try:
        df = ak.stock_zh_index_daily(symbol=sina_code)
        # 只保留 date 和 close 列
        result = df[['date', 'close']].copy()
        result['date'] = pd.to_datetime(result['date']).dt.date
        # 不足 100 行的数据不缓存（避免网络异常导致缓存截断）
        if len(result) >= 100:
            result.to_csv(cache_path, index=False)
        else:
            logger.warning("⚠ 基准 %s 数据行数不足(%d)，跳过缓存", benchmark_code, len(result))
        return result
    except Exception as e:
        logger.warning("⚠ 下载基准 %s 失败: %s", benchmark_code, e)
        return pd.DataFrame()
