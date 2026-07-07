"""Tests for data.py — ETF data fetching and CSV caching.

Tests use mocked AKShare (no real network calls). Caching is tested
via pytest's tmp_path for isolation.
"""

import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from data import fetch_single_etf, load_all_etf_data


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_ak_data() -> pd.DataFrame:
    """Realistic AKShare-style DataFrame (2 rows)."""
    return pd.DataFrame({
        'date': [date(2024, 1, 2), date(2024, 1, 3)],
        'open': [1.0, 1.02],
        'high': [1.05, 1.06],
        'low': [0.99, 1.01],
        'close': [1.02, 1.04],
        'volume': [1_000_000, 1_500_000],
        'amount': [1_020_000.0, 1_560_000.0],
    })


# ======================================================================
# fetch_single_etf
# ======================================================================

class TestFetchSingleETF:
    """fetch_single_etf 单元测试"""

    def test_returns_dataframe_with_expected_columns(self, mock_ak_data):
        """确保返回的 DataFrame 包含所有必需列"""
        with patch('data.ak.fund_etf_hist_sina', return_value=mock_ak_data):
            df = fetch_single_etf('sh513100')
            assert isinstance(df, pd.DataFrame)
            expected = {'date', 'open', 'high', 'low', 'close', 'volume', 'amount'}
            assert expected.issubset(set(df.columns))

    def test_calls_akshare_with_correct_symbol(self, mock_ak_data):
        """验证传入的 sina_code 正确传递给 AKShare"""
        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            mock_ak.return_value = mock_ak_data
            fetch_single_etf('sh513100')
            mock_ak.assert_called_once_with(
                symbol='sh513100', start_date=None, end_date=None
            )

    def test_empty_date_range_still_calls_akshare(self, mock_ak_data):
        """空日期范围应全量获取（传 None 给 AKShare）"""
        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            mock_ak.return_value = mock_ak_data
            fetch_single_etf('sz159915', start_date='', end_date='')
            mock_ak.assert_called_once_with(
                symbol='sz159915', start_date=None, end_date=None
            )

    def test_date_range_is_passed_none_when_empty(self, mock_ak_data):
        """确认空的 start_date/end_date 以 None 传给 AKShare"""
        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            mock_ak.return_value = mock_ak_data
            fetch_single_etf('sh518880', start_date='', end_date='')
            # symbol only — no start_date/end_date in kwargs
            call_kwargs = mock_ak.call_args[1]
            assert call_kwargs.get('start_date') is None
            assert call_kwargs.get('end_date') is None

    def test_date_range_is_passed_when_provided(self, mock_ak_data):
        """提供日期范围时应传给 AKShare"""
        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            mock_ak.return_value = mock_ak_data
            fetch_single_etf('sh512890', start_date='2024-01-01', end_date='2024-01-31')
            call_kwargs = mock_ak.call_args[1]
            assert call_kwargs.get('start_date') == '2024-01-01'
            assert call_kwargs.get('end_date') == '2024-01-31'

    def test_data_types(self, mock_ak_data):
        """验证返回数据的基本类型正确"""
        with patch('data.ak.fund_etf_hist_sina', return_value=mock_ak_data):
            df = fetch_single_etf('sh513100')
            assert df['date'].dtype == object  # datetime.date
            assert df['volume'].dtype in ('int64', 'int32')


# ======================================================================
# load_all_etf_data
# ======================================================================

class TestLoadAllETFData:
    """load_all_etf_data 单元测试"""

    def test_cache_miss_downloads_and_caches(self, tmp_path, mock_ak_data):
        """缓存未命中 → 下载 → 写入 CSV → 返回数据"""
        codes = ['513100.XSHG', '159915.XSHE']
        cache_dir = str(tmp_path / 'data_cache')

        with patch('data.ak.fund_etf_hist_sina', return_value=mock_ak_data) as mock_ak:
            result = load_all_etf_data(codes, cache_dir=cache_dir)

        # 应调用 AKShare 两次（两只 ETF 均未缓存）
        assert mock_ak.call_count == 2
        mock_ak.assert_any_call(
            symbol='sh513100', start_date=None, end_date=None
        )
        mock_ak.assert_any_call(
            symbol='sz159915', start_date=None, end_date=None
        )

        # CSV 文件应已写入
        assert os.path.isfile(os.path.join(cache_dir, 'sh513100.csv'))
        assert os.path.isfile(os.path.join(cache_dir, 'sz159915.csv'))

        # 返回 dict key 应为原始代码
        assert set(result.keys()) == set(codes)
        for code in codes:
            assert isinstance(result[code], pd.DataFrame)

    def test_cache_hit_skips_download(self, tmp_path, mock_ak_data):
        """缓存命中 → 读取 CSV → 不调 AKShare"""
        codes = ['513100.XSHG']
        cache_dir = str(tmp_path / 'data_cache')
        os.makedirs(cache_dir)

        # 预置缓存
        mock_ak_data.to_csv(os.path.join(cache_dir, 'sh513100.csv'), index=False)

        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            result = load_all_etf_data(codes, cache_dir=cache_dir)

        mock_ak.assert_not_called()
        assert '513100.XSHG' in result
        # 验证列数一致（date 列会被 parse 为 datetime.date）
        assert list(result['513100.XSHG'].columns) == list(mock_ak_data.columns)

    def test_auto_creates_cache_dir(self, tmp_path, mock_ak_data):
        """缓存目录不存在时自动创建"""
        codes = ['513100.XSHG']
        cache_dir = str(tmp_path / 'nonexistent' / 'deep' / 'cache')
        assert not os.path.exists(cache_dir)

        with patch('data.ak.fund_etf_hist_sina', return_value=mock_ak_data):
            result = load_all_etf_data(codes, cache_dir=cache_dir)

        assert os.path.isdir(cache_dir)
        assert '513100.XSHG' in result

    def test_mixed_hit_and_miss(self, tmp_path, mock_ak_data):
        """部分缓存命中 + 部分未命中 → 正确的混合行为"""
        codes = ['513100.XSHG', '159915.XSHE']
        cache_dir = str(tmp_path / 'data_cache')
        os.makedirs(cache_dir)

        # 预置 513100 的缓存
        mock_ak_data.to_csv(os.path.join(cache_dir, 'sh513100.csv'), index=False)

        with patch('data.ak.fund_etf_hist_sina', return_value=mock_ak_data) as mock_ak:
            result = load_all_etf_data(codes, cache_dir=cache_dir)

        # 只有未命中的 159915 会调 AKShare
        mock_ak.assert_called_once_with(
            symbol='sz159915', start_date=None, end_date=None
        )
        assert set(result.keys()) == set(codes)

    def test_cached_csv_date_column_parsed_correctly(self, tmp_path, mock_ak_data):
        """缓存 CSV 的 date 列在读取后仍应为 datetime.date"""
        codes = ['513100.XSHG']
        cache_dir = str(tmp_path / 'data_cache')
        os.makedirs(cache_dir)
        mock_ak_data.to_csv(os.path.join(cache_dir, 'sh513100.csv'), index=False)

        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            result = load_all_etf_data(codes, cache_dir=cache_dir)

        df = result['513100.XSHG']
        # date 列应为 datetime.date 类型
        assert all(isinstance(d, date) for d in df['date'])
        mock_ak.assert_not_called()

    def test_empty_code_list(self, tmp_path):
        """空代码列表应返回空字典"""
        cache_dir = str(tmp_path / 'cache')
        with patch('data.ak.fund_etf_hist_sina') as mock_ak:
            result = load_all_etf_data([], cache_dir=cache_dir)
        assert result == {}
        mock_ak.assert_not_called()
