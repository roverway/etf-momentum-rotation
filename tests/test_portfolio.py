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
        assert pf.cash == 990_000.0
        assert pf.positions['513100'].quantity == 5000

    def test_buy_new_position_sets_avg_price(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        assert pf.positions['513100'].avg_price == 2.0

    def test_buy_insufficient_cash_raises(self):
        pf = Portfolio(cash=100)
        with pytest.raises(ValueError, match='Insufficient cash'):
            pf.buy('513100', 5000, 2.0)


class TestPortfolioBuyMultiple:
    """Buying same ETF multiple times — weighted average price."""

    def test_two_buys_weighted_avg_price(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cost 10000, qty 5000
        pf.buy('513100', 5000, 3.0)   # cost 15000, qty 5000
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
        pf.buy('513100', 5000, 2.0)   # cash=990000
        pf.sell('513100', 2000, 2.5)   # cash += 2000*2.5 = 5000
        assert pf.cash == 995_000.0

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


class TestPortfolioTotalValue:
    """Portfolio.total_value — cash + position market values."""

    def test_total_value_no_positions(self):
        pf = Portfolio(cash=100_000)
        assert pf.total_value == 100_000.0

    def test_total_value_with_positions(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cost=10000, mkt=5000*2.0=10000
        assert pf.total_value == 1_000_000.0  # cash 990k + mkt 10k

    def test_total_value_after_price_change(self):
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)
        pf.update_price('513100', 3.0)
        # cash=990k, position mkt=5000*3.0=15000
        assert pf.total_value == 1_005_000.0

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
        pf.buy('513100', 5000, 2.0)   # cash=990k, qty=5000
        pf.sell('513100', 5000, 2.5)  # cash=990k+12.5k=1,002,500, realized=2500

        pf.buy('513100', 3000, 3.0)   # cash=1,002,500-9000=993,500, qty=3000
        assert pf.cash == 993_500.0
        assert pf.positions['513100'].quantity == 3000
        assert pf.total_pnl == 2500.0

    def test_scenario_trading_sequence(self):
        """Full scenario: buy ETF-A, buy ETF-B, sell part of A, update prices."""
        pf = Portfolio(cash=1_000_000)
        pf.buy('513100', 5000, 2.0)   # cash=990k
        pf.buy('159915', 3000, 1.0)   # cash=987k
        pf.sell('513100', 2000, 2.5)  # cash=987k+5k=992k, realized=1000

        assert pf.positions['513100'].quantity == 3000
        assert pf.total_pnl == 1000.0

        pf.update_price('513100', 3.0)   # mkt=3000*3.0=9000
        pf.update_price('159915', 1.2)   # mkt=3000*1.2=3600
        # total_value = cash 992k + 9000 + 3600 = 1,004,600
        assert pf.total_value == 1_004_600.0
