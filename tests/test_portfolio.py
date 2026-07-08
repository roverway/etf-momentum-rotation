"""Tests for portfolio module — Position & Portfolio classes."""

import pytest
from portfolio import Position, Portfolio


class TestPosition:
    """Position data class — basic structure tests."""

    def test_fields(self):
        p = Position(code='513100.XSHG', quantity=5000, avg_price=2.0)
        assert p.code == '513100.XSHG'
        assert p.quantity == 5000
        assert p.avg_price == 2.0

    def test_market_value(self):
        p = Position(code='513100.XSHG', quantity=5000, avg_price=2.0)
        assert p.market_value == 5000 * 2.0  # 10000.0

    def test_unrealized_pnl(self):
        p = Position(code='513100.XSHG', quantity=5000, avg_price=2.0)
        p.current_price = 2.5
        assert p.unrealized_pnl == 5000 * (2.5 - 2.0)  # 2500.0

    def test_unrealized_pnl_zero_when_no_current_price(self):
        p = Position(code='513100.XSHG', quantity=5000, avg_price=2.0)
        assert p.unrealized_pnl == 0.0


class TestPortfolioBuy:
    """Portfolio.buy() — cash deduction & position creation."""

    def test_buy_new_position_deducts_cash(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        # effective_price=2.002, cost=10010.0, comm=2.0, total=10012.0
        # cash = 1_000_000 - 10012.0 = 989988.0
        assert pf.cash == 989_988.0
        assert pf.positions['513100'].quantity == 5000

    def test_buy_new_position_sets_avg_price(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        assert pf.positions['513100'].avg_price == 2.0

    def test_buy_insufficient_cash_raises(self):
        pf = Portfolio(cash=100)
        with pytest.raises(ValueError, match='Insufficient cash'):
            pf.buy('513100', 5000, 2.0)

    def test_buy_zero_shares_no_op(self):
        """买入 0 股不应影响现金或仓位"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 0, 2.0)
        assert pf.cash == 1_000_000
        assert '513100' not in pf.positions


class TestPortfolioBuyMultiple:
    """Buying same ETF multiple times — weighted average price."""

    def test_two_buys_weighted_avg_price(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cost 10000, qty 5000, avg_price=2.0
        pf.buy('513100', 5000, 3.0)   # cost 15000, qty 5000, avg_price=3.0
        # avg_price = (5000*2.0 + 5000*3.0) / 10000 = 2.5
        assert pf.positions['513100'].avg_price == 2.5
        assert pf.positions['513100'].quantity == 10000

    def test_three_buys_weighted_avg_price(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 1000, 1.0)
        pf.buy('513100', 2000, 2.0)
        pf.buy('513100', 3000, 3.0)
        # total_qty=6000, total_cost=1000*1+2000*2+3000*3=14000
        # avg_price = 14000/6000 ≈ 2.333333...
        assert pf.positions['513100'].avg_price == pytest.approx(14_000 / 6000)
        assert pf.positions['513100'].quantity == 6000


class TestPortfolioSell:
    """Portfolio.sell() — cash increase, position decrease, realized PnL."""

    def test_sell_increases_cash(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)    # cash=989988.0
        pf.sell('513100', 2000, 2.5)   # net=4994.0 (price*0.999*2000 - comm)
        # cash = 989988.0 + 4994.0 = 994982.0
        assert pf.cash == 994_982.0

    def test_sell_returns_realized_pnl(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        realized = pf.sell('513100', 2000, 2.5)
        # realized = (2.5 - 2.0) * 2000 = 1000
        assert realized == 1000.0

    def test_sell_negative_realized_pnl(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        realized = pf.sell('513100', 1000, 1.5)
        assert realized == -500.0

    def test_sell_updates_quantity(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.sell('513100', 2000, 2.5)
        assert pf.positions['513100'].quantity == 3000

    def test_sell_all_removes_position(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.sell('513100', 5000, 2.5)
        assert '513100' not in pf.positions

    def test_sell_more_than_held_raises(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        with pytest.raises(ValueError, match='Insufficient'):
            pf.sell('513100', 6000, 2.5)

    def test_sell_nonexistent_position_raises(self):
        pf = Portfolio(cash=1_000_000)
        with pytest.raises(ValueError, match='not held'):
            pf.sell('513100', 1000, 2.5)

    def test_sell_zero_shares_no_op(self):
        """卖出 0 股不应影响现金、仓位或已实现盈亏"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        cash_before = pf.cash
        realized = pf.sell('513100', 0, 2.5)
        assert realized == 0.0
        assert pf.cash == cash_before
        assert pf.positions['513100'].quantity == 5000
        assert pf.total_pnl == 0.0


class TestPortfolioTotalValue:
    """Portfolio.total_value — cash + position market values."""

    def test_total_value_no_positions(self):
        pf = Portfolio(cash=100_000)
        assert pf.total_value == 100_000.0

    def test_total_value_with_positions(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        # cash=989988.0, mkt=5000*2.0=10000
        assert pf.total_value == 989_988.0 + 5000 * 2.0  # 999988.0

    def test_total_value_after_price_change(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.update_price('513100', 3.0)
        # cash=989988.0, position mkt=5000*3.0=15000
        assert pf.total_value == 1_004_988.0

    def test_update_price_nonexistent_ignored(self):
        pf = Portfolio(cash=100_000)
        pf.update_price('NONEXIST', 10.0)  # should not raise
        assert pf.total_value == 100_000.0


class TestPortfolioTotalPnl:
    """Portfolio.total_pnl — cumulative realized PnL tracking."""

    def test_total_pnl_starts_zero(self):
        pf = Portfolio(cash=1_000_000)
        assert pf.total_pnl == 0.0

    def test_total_pnl_accumulates_across_trades(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        r1 = pf.sell('513100', 2000, 2.5)  # realized=1000
        assert pf.total_pnl == 1000.0
        r2 = pf.sell('513100', 3000, 3.0)  # realized=(3-2)*3000=3000
        assert pf.total_pnl == 4000.0

    def test_total_pnl_with_multiple_etfs(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.buy('159915', 3000, 1.0)
        r1 = pf.sell('513100', 2000, 2.5)   # realized=1000
        r2 = pf.sell('159915', 1000, 1.5)   # realized=500
        assert pf.total_pnl == 1500.0

    def test_total_pnl_with_losses(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.sell('513100', 5000, 1.8)  # realized=-1000
        assert pf.total_pnl == pytest.approx(-1000.0)


class TestPortfolioMixedOperations:
    """Integration tests combining multiple operations."""

    def test_buy_then_sell_then_buy_again(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cash=989988.0
        pf.sell('513100', 5000, 2.5)  # cash=989988.0+12485.0=1,002,473.0

        pf.buy('513100', 3000, 3.0)   # cash=1,002,473.0-9010.8=993,462.2
        assert pf.cash == pytest.approx(993_462.2)
        assert pf.positions['513100'].quantity == 3000
        assert pf.total_pnl == 2500.0

    def test_scenario_trading_sequence(self):
        """Full scenario: buy ETF-A, buy ETF-B, sell part of A, update prices."""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cash=989988.0
        pf.buy('159915', 3000, 1.0)   # cash=989988.0-3003.6=986984.4
        pf.sell('513100', 2000, 2.5)  # cash=986984.4+4994.0=991978.4

        assert pf.positions['513100'].quantity == 3000
        assert pf.total_pnl == 1000.0

        pf.update_price('513100', 3.0)   # mkt=3000*3.0=9000
        pf.update_price('159915', 1.2)   # mkt=3000*1.2=3600
        # total_value = cash 991978.4 + 9000 + 3600 = 1,004,578.4
        assert pf.total_value == 1_004_578.4


# =====================================================================
# New tests: commission & slippage
# =====================================================================

class TestPortfolioCommission:
    """Commission deduction on buy/sell."""

    def test_buy_commission_deducted(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0.00025, slippage_rate=0)
        # cost = 5000 * 2.0 = 10000
        # commission = round(10000 * 0.00025, 2) = 2.5
        # cash = 1_000_000 - 10000 - 2.5 = 989997.5
        assert pf.cash == 989_997.5

    def test_sell_commission_deducted(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0, slippage_rate=0)
        pf.sell('513100', 2000, 2.5, commission_rate=0.00025, slippage_rate=0)
        # proceeds = 2000 * 2.5 = 5000
        # commission = round(5000 * 0.00025, 2) = 1.25
        # net = 5000 - 1.25 = 4998.75
        # cash = 990000 + 4998.75 = 994998.75
        assert pf.cash == 994_998.75

    def test_commission_rate_zero(self):
        """佣金率为0时行为不变"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0, slippage_rate=0)
        assert pf.cash == 990_000.0
        pf.sell('513100', 2000, 2.5, commission_rate=0, slippage_rate=0)
        assert pf.cash == 995_000.0

    def test_avg_price_uses_original_price(self):
        """avg_price 用原始报价，不是含滑点的价格"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0.00025, slippage_rate=0.0001)
        assert pf.positions['513100'].avg_price == 2.0


class TestPortfolioSlippage:
    """Slippage effect on buy/sell prices."""

    def test_buy_slippage_increases_cost(self):
        """验证买入成交价高于报价"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0, slippage_rate=0.0001)
        # effective_price = 2.0 * 1.0001 = 2.0002
        # cost = 5000 * 2.0002 = 10001.0
        # cash = 1_000_000 - 10001.0 = 989999.0
        assert pf.cash == 989_999.0

    def test_sell_slippage_decreases_proceeds(self):
        """验证卖出成交价低于报价"""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0, commission_rate=0, slippage_rate=0)
        pf.sell('513100', 2000, 2.5, commission_rate=0, slippage_rate=0.0001)
        # effective_price = 2.5 * 0.9999 = 2.49975
        # proceeds = 2000 * 2.49975 = 4999.5
        # cash = 990000 + 4999.5 = 994999.5
        assert pf.cash == 994_999.5
