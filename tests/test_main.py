"""Tests for main.py CLI entry point."""

from unittest.mock import MagicMock, patch
import sys

import pytest

from main import main


class TestMain:
    def test_no_args_shows_help(self):
        """无参数显示帮助并退出码1"""
        with patch.object(sys, 'argv', ['main.py']):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1

    def test_backtest_command_runs(self):
        """backtest子命令调用run_backtest"""
        mock_portfolio = MagicMock()
        mock_portfolio.total_value = 1_200_000.0
        mock_portfolio.total_pnl = 200_000.0
        mock_portfolio.positions = {}

        with patch('strategy.setup_logger') as mock_logger:
            with patch(
                'backtest_engine.run_backtest', return_value=mock_portfolio
            ) as mock_bt:
                with patch.object(
                    sys,
                    'argv',
                    [
                        'main.py',
                        'backtest',
                        '--start',
                        '2024-01-02',
                        '--end',
                        '2024-03-01',
                    ],
                ):
                    main()

        mock_bt.assert_called_once()
        call_config = mock_bt.call_args[0][0]
        assert call_config.start_date == '2024-01-02'
        assert call_config.end_date == '2024-03-01'
        assert call_config.initial_cash == 1_000_000

    def test_signal_command_runs(self):
        """signal子命令可运行"""
        with patch('strategy.setup_logger') as mock_logger:
            with patch.object(sys, 'argv', ['main.py', 'signal']):
                main()  # Should not raise

    def test_backtest_passes_start_end_to_config(self):
        """验证参数正确传递"""
        mock_portfolio = MagicMock()
        mock_portfolio.total_value = 1_200_000.0
        mock_portfolio.total_pnl = 200_000.0
        mock_portfolio.positions = {}

        with patch('strategy.setup_logger'):
            with patch(
                'backtest_engine.run_backtest', return_value=mock_portfolio
            ) as mock_bt:
                with patch.object(
                    sys,
                    'argv',
                    [
                        'main.py',
                        'backtest',
                        '--start',
                        '2019-01-18',
                        '--end',
                        '2024-12-31',
                        '--cash',
                        '500000',
                    ],
                ):
                    main()

        call_config = mock_bt.call_args[0][0]
        assert call_config.start_date == '2019-01-18'
        assert call_config.end_date == '2024-12-31'
        assert call_config.initial_cash == 500000
