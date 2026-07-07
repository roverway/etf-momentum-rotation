"""Portfolio tracking module — Position & Portfolio classes.

Provides:
  - Position: dataclass holding ETF code, quantity, avg cost price, current price.
  - Portfolio: manages positions, cash, buy/sell/update_price operations,
    and tracks cumulative realized PnL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Position:
    """A single ETF position.

    Attributes:
        code: ETF code (e.g. '513100.XSHG').
        quantity: Number of shares held.
        avg_price: Average cost price per share (weighted).
        current_price: Latest market price (optional, for unrealized PnL).
    """

    code: str
    quantity: int = 0
    avg_price: float = 0.0
    current_price: float | None = None

    @property
    def market_value(self) -> float:
        """Current market value of this position."""
        price = self.current_price if self.current_price is not None else self.avg_price
        return self.quantity * price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss (mark-to-market)."""
        if self.current_price is None:
            return 0.0
        return (self.current_price - self.avg_price) * self.quantity


class Portfolio:
    """Tracks positions, cash, and cumulative realized PnL.

    Attributes:
        positions: dict of code -> Position.
        cash: Available cash balance.
        _total_pnl: Cumulative realized PnL across all trades.
    """

    def __init__(self, cash: float = 0.0) -> None:
        self.positions: Dict[str, Position] = {}
        self.cash: float = float(cash)
        self._total_pnl: float = 0.0

    @property
    def total_value(self) -> float:
        """Total portfolio value = cash + sum of position market values."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Cumulative realized PnL across all trades (sell returns)."""
        return self._total_pnl

    def buy(self, code: str, quantity: int, price: float) -> None:
        """Buy shares of an ETF.

        Deducts cost from cash. If the ETF is already held, updates the
        average cost price using weighted average.

        Args:
            code: ETF code.
            quantity: Number of shares to buy.
            price: Price per share.

        Raises:
            ValueError: If cash is insufficient.
        """
        cost = quantity * price
        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}"
            )

        self.cash -= cost

        if code in self.positions:
            pos = self.positions[code]
            total_cost = pos.avg_price * pos.quantity + cost
            total_qty = pos.quantity + quantity
            pos.avg_price = total_cost / total_qty
            pos.quantity = total_qty
        else:
            self.positions[code] = Position(
                code=code,
                quantity=quantity,
                avg_price=price,
                current_price=price,
            )

    def sell(self, code: str, quantity: int, price: float) -> float:
        """Sell shares of an ETF.

        Adds proceeds to cash, returns realized PnL, and removes the
        position if fully closed.

        Args:
            code: ETF code.
            quantity: Number of shares to sell.
            price: Price per share.

        Returns:
            Realized PnL: (price - avg_price) * quantity.

        Raises:
            ValueError: If the position does not exist or quantity exceeds
                shares held.
        """
        if code not in self.positions:
            raise ValueError(f"Position {code} not held")

        pos = self.positions[code]
        if quantity > pos.quantity:
            raise ValueError(
                f"Insufficient shares: want {quantity}, have {pos.quantity}"
            )

        realized = (price - pos.avg_price) * quantity
        self._total_pnl += realized

        proceeds = quantity * price
        self.cash += proceeds

        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.positions[code]

        return realized

    def update_price(self, code: str, price: float) -> None:
        """Update the current market price for a position.

        Affects total_value via position market_value. Silently ignores
        codes not in the portfolio.

        Args:
            code: ETF code.
            price: Latest market price.
        """
        if code in self.positions:
            self.positions[code].current_price = price
