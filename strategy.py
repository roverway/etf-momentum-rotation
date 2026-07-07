"""Strategy logic module — momentum signal, trade logging, logger setup.

Core function
-------------
``calculate_momentum_signal``
    Pure function that computes the highest-momentum ETF over a lookback window.
    Logic is a 1:1 translation of the original RiceQuant reference.

    Steps (matching RQ ``main.py:28-63``):
      1. Determine the end date: latest date across all ETFs (or *base_date* if given).
      2. Build a combined close-price DataFrame indexed by date (like RQ ``get_price``).
      3. Take the last ``check_range`` rows.
      4. Compute return per ETF: ``(close_last - close_first) / close_first``.
      5. Drop NaN returns.
      6. Pick the ETF with the highest return (tie → alphabetical for determinism).
      7. If the best return > 0 → return the ETF code; else → None.

Supporting
----------
``log_trade_signal`` — print human-readable Chinese trade instruction.
``setup_logger`` — factory for a timestamped console logger.
"""

from __future__ import annotations

import logging

import pandas as pd


# =====================================================================================
# Core momentum signal
# =====================================================================================

def calculate_momentum_signal(
    etf_data_dict: dict[str, pd.DataFrame],
    check_range: int,
    base_date: object = None,
) -> str | None:
    """Determine the target ETF with the strongest positive momentum.

    Parameters
    ----------
    etf_data_dict : dict[str, pd.DataFrame]
        ETF code → DataFrame with at least ``date`` (datetime.date) and ``close`` (float)
        columns, sorted ascending by date.
    check_range : int
        Number of most-recent trading days to evaluate (e.g. 22 ≈ 1 month).
    base_date : object, optional
        Reference date (``datetime.date`` or anything comparable).  Data points with
        ``date > base_date`` are excluded from consideration.  When *None*, the latest
        date across all ETFs is used.

    Returns
    -------
    str or None
        ETF code with the highest positive return, or *None* when no candidate has
        a return > 0 or when data is insufficient.

    Notes
    -----
    Logic mirrors the original RiceQuant implementation:

    .. code:: python

        end_date = get_previous_trading_date(base_date)
        start_date = get_previous_trading_date(end_date, n=check_range - 1)
        all_prices = get_price(codes, start=start_date, end=end_date, fields='close')
        # … then iloc[0] / iloc[-1] / dropna / max
    """
    if not etf_data_dict:
        return None

    # ── 1. Build combined close-price DataFrame (like RQ get_price) ──────────
    #   Aligns by date index (outer join), so each ETF gets a column and dates
    #   that some ETFs lack become NaN.
    close_series: dict[str, pd.Series] = {}
    for code, df in etf_data_dict.items():
        if base_date is not None:
            sub = df[df['date'] <= base_date]
        else:
            sub = df
        if sub.empty:
            continue
        close_series[code] = sub.set_index('date')['close']

    if not close_series:
        return None

    all_prices = pd.DataFrame(close_series)  # index=date, columns=etf_codes

    # ── 2. Take the most-recent check_range rows ────────────────────────────
    if len(all_prices) < check_range:
        return None
    all_prices = all_prices.tail(check_range)

    # ── 3. RQ logic: DataFrame path vs Series path ──────────────────────────
    valid_returns: dict[str, float] = {}

    if all_prices.shape[1] > 1:
        # ── Multi-ETF (DataFrame) path ──
        start_prices = all_prices.iloc[0]
        last_prices = all_prices.iloc[-1]
        etf_returns = (last_prices - start_prices) / start_prices
        valid_returns = etf_returns.dropna().to_dict()
    else:
        # ── Single-ETF (Series) path ──
        the_one = all_prices.columns[0]
        start_price = all_prices.iloc[0, 0]
        last_price = all_prices.iloc[-1, 0]
        if pd.notna(start_price) and pd.notna(last_price):
            valid_returns[the_one] = (last_price - start_price) / start_price

    if not valid_returns:
        return None

    # ── 4. Pick best ────────────────────────────────────────────────────────
    #   ``sorted(…)`` provides deterministic tie-breaking (alphabetical order).
    best_etf = max(sorted(valid_returns), key=valid_returns.get)

    if valid_returns[best_etf] > 0:
        return best_etf

    return None


# =====================================================================================
# Trade signal logging
# =====================================================================================

def log_trade_signal(
    current_position: str | None,
    next_day_target: str | None,
    logger: logging.Logger,
) -> None:
    """Print a human-readable Chinese trade instruction.

    Parameters
    ----------
    current_position : str or None
        ETF code currently held, or *None* for cash.
    next_day_target : str or None
        ETF code to trade into tomorrow, or *None* to go to cash.
    logger : logging.Logger
        Logger instance to emit the messages.

    Logic mirrors the original ``main.py:65-87``.
    """
    current_name = current_position or "空仓"
    target_name = next_day_target or "空仓"

    logger.info("  > 当前实盘持仓: 【%s】", current_name)
    logger.info("  > 次日交易目标: 【%s】", target_name)
    logger.info("-" * 40)

    if current_position == next_day_target:
        logger.info(
            "✅ 操作建议: 保持现有持仓【%s】不变。(HOLD)",
            current_name,
        )
    elif current_position is None and next_day_target is not None:
        logger.info(
            "🚀 操作建议: 从【空仓】买入【%s】。(BUY)",
            target_name,
        )
    elif current_position is not None and next_day_target is None:
        logger.info(
            "🛑 操作建议: 卖出【%s】，清仓观望。(SELL TO CASH)",
            current_name,
        )
    else:
        logger.info(
            "🔁 操作建议: 卖出【%s】，换仓买入【%s】。(SWITCH)",
            current_name,
            target_name,
        )


# =====================================================================================
# Logger factory
# =====================================================================================

def setup_logger(name: str = 'strategy') -> logging.Logger:
    """Return a configured ``logging.Logger`` with timestamp + level.

    Parameters
    ----------
    name : str
        Logger name (default ``'strategy'``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
