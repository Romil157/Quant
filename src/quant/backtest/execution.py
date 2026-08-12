"""Execution simulator with transaction cost modeling."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quant.backtest.types import Fill, Order, OrderSide, OrderType


@dataclass
class ExecutionConfig:
    """Execution cost configuration."""
    commission_bps: float = 2.0      # Commission in basis points
    spread_bps: float = 1.0          # Bid-ask spread in basis points
    slippage_bps: float = 2.0        # Slippage in basis points
    market_impact_bps: float = 0.0   # Market impact in basis points (per 1% ADV)
    min_commission: float = 0.0      # Minimum commission per trade
    fill_probability: float = 1.0    # Probability of fill for limit orders (0-1)
    partial_fill_prob: float = 0.0   # Probability of partial fill


class ExecutionSimulator:
    """Simulates order execution with realistic costs."""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.rng = np.random.default_rng(42)

    def simulate_fill(
        self,
        order: Order,
        market_price: float,
        bid: float | None = None,
        ask: float | None = None,
        volume: float = 0.0,
        adv: float = 0.0,
    ) -> Fill | None:
        """
        Simulate order fill with transaction costs.

        Args:
            order: Order to fill
            market_price: Current market price (mid)
            bid: Current bid price
            ask: Current ask price
            volume: Current bar volume
            adv: Average daily volume for market impact

        Returns:
            Fill if order executes, None otherwise
        """
        # Determine fill probability
        if order.order_type == OrderType.LIMIT:
            if self.rng.random() > self.config.fill_probability:
                return None

            # Check if limit price is reached
            if (
                order.side == OrderSide.BUY
                and order.limit_price is not None
                and ask is not None
                and order.limit_price < ask
            ):
                return None
            if (
                order.side == OrderSide.SELL
                and order.limit_price is not None
                and bid is not None
                and order.limit_price > bid
            ):
                return None

        # Determine execution price
        if order.order_type == OrderType.MARKET:
            exec_price = self._get_market_execution_price(order, market_price, bid, ask)
        else:
            # Limit order fills at limit price (or better)
            exec_price = order.limit_price or market_price
            if order.side == OrderSide.BUY:
                exec_price = min(exec_price, ask or market_price)
            else:
                exec_price = max(exec_price, bid or market_price)

        # Apply slippage
        exec_price = self._apply_slippage(exec_price, order.side)

        # Apply market impact
        exec_price = self._apply_market_impact(exec_price, order.side, order.quantity, adv)

        # Determine fill quantity
        fill_qty = order.quantity
        if self.rng.random() < self.config.partial_fill_prob:
            fill_qty = order.quantity * self.rng.uniform(0.1, 0.9)

        # Calculate commission
        commission = self._calculate_commission(fill_qty, exec_price)

        return Fill(
            order_id=order.order_id or "",
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            timestamp=datetime.now(),
            commission=commission,
        )

    def _get_market_execution_price(
        self,
        order: Order,
        mid: float,
        bid: float | None,
        ask: float | None,
    ) -> float:
        """Get execution price for market order."""
        if order.side == OrderSide.BUY:
            return ask if ask is not None else mid * (1 + self.config.spread_bps / 20000)
        else:
            return bid if bid is not None else mid * (1 - self.config.spread_bps / 20000)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to execution price."""
        slippage = price * self.config.slippage_bps / 10000
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage

    def _apply_market_impact(
        self,
        price: float,
        side: OrderSide,
        quantity: float,
        adv: float,
    ) -> float:
        """Apply market impact based on order size relative to ADV."""
        if adv <= 0 or self.config.market_impact_bps <= 0:
            return price

        participation = quantity / adv
        impact = price * self.config.market_impact_bps / 10000 * participation

        if side == OrderSide.BUY:
            return price + impact
        else:
            return price - impact

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission."""
        notional = quantity * price
        commission = notional * self.config.commission_bps / 10000
        return max(commission, self.config.min_commission)
