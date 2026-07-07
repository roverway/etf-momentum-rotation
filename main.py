"""main.py — ETF动量轮动策略入口。"""

import argparse
import sys
from datetime import date

from config import BacktestConfig, ETF_POOL


def run_backtest_mode(start: str, end: str, initial_cash: float = 1_000_000):
    """运行回测模式"""
    from backtest_engine import run_backtest
    from strategy import setup_logger

    logger = setup_logger("trade")
    logger.info("回测启动: %s ~ %s, 本金 %.0f", start, end, initial_cash)

    config = BacktestConfig(
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
    )
    config.etf_codes = ETF_POOL  # duck-typing for backtest_engine.getattr

    portfolio = run_backtest(config, output_dir='backtest_results/')

    logger.info("回测完成")
    logger.info("最终资产: %.2f", portfolio.total_value)
    logger.info("总收益: %.2f", portfolio.total_pnl)
    logger.info(
        "持仓: %s",
        {code: pos.quantity for code, pos in portfolio.positions.items()},
    )


def run_signal_mode():
    """运行信号模式（占位，Task 11完善）"""
    from strategy import setup_logger

    logger = setup_logger("trade")
    logger.info("信号模式 - 待实现")


def main():
    parser = argparse.ArgumentParser(
        description="ETF动量轮动策略 - 回测与实盘信号",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # backtest 子命令
    bt_parser = subparsers.add_parser("backtest", help="运行回测")
    bt_parser.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    bt_parser.add_argument(
        "--end", default=str(date.today()), help="结束日期 YYYY-MM-DD"
    )
    bt_parser.add_argument("--cash", type=float, default=1_000_000, help="本金")

    # signal 子命令
    subparsers.add_parser("signal", help="生成实盘信号")

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest_mode(args.start, args.end, args.cash)
    elif args.command == "signal":
        run_signal_mode()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
