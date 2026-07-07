"""A-share trading calendar module.

Provides functions to load A-share trading calendar from AKShare and
navigate between trading dates.
"""
import datetime
from typing import Union

import akshare as ak

_CALENDAR_CACHE: list[datetime.date] | None = None


def _to_date(value: Union[str, datetime.date]) -> datetime.date:
    """Normalize input to a ``datetime.date``.

    Accepts ``datetime.date`` or ``'YYYY-MM-DD'`` string.
    """
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    raise TypeError(f"Expected date or str, got {type(value).__name__}")


def load_trading_calendar() -> list[datetime.date]:
    """Load and cache A-share trading calendar from AKShare.

    Returns a sorted list of all A-share trading days as ``datetime.date``.
    Results are cached in the module-level ``_CALENDAR_CACHE`` variable to
    avoid repeated AKShare requests.
    """
    global _CALENDAR_CACHE

    if _CALENDAR_CACHE is not None:
        return _CALENDAR_CACHE

    df = ak.tool_trade_date_hist_sina()
    dates: list[datetime.date] = sorted(df["trade_date"].tolist())
    _CALENDAR_CACHE = dates
    return dates


def get_previous_trading_date(
    base_date: Union[str, datetime.date],
    n: int = 1,
) -> datetime.date:
    """Return the n-th previous trading day before *base_date*.

    Parameters
    ----------
    base_date : str or datetime.date
        The reference date (``'YYYY-MM-DD'`` string or ``datetime.date``).
    n : int
        Number of trading days to go back (default 1).

    Returns
    -------
    datetime.date
        The n-th previous trading day.

    Raises
    ------
    ValueError
        If the requested date is before the earliest known trading day.
    """
    base = _to_date(base_date)
    calendar = load_trading_calendar()

    # Binary search: idx = first trading day >= base_date.
    # For both cases (base is or is not a trading day), the n-th
    # previous trading day is at index (idx - n).
    idx = _bisect_left(calendar, base)
    prev_idx = idx - n

    if prev_idx < 0:
        raise ValueError(
            f"no trading day {n} days before {base_date} "
            f"(earliest: {calendar[0]})"
        )
    return calendar[prev_idx]


def get_next_trading_date(
    base_date: Union[str, datetime.date],
) -> datetime.date:
    """Return the next trading day after *base_date*.

    Parameters
    ----------
    base_date : str or datetime.date
        The reference date (``'YYYY-MM-DD'`` string or ``datetime.date``).

    Returns
    -------
    datetime.date
        The next trading day.

    Raises
    ------
    ValueError
        If *base_date* is the last known trading day (no next day).
    """
    base = _to_date(base_date)
    calendar = load_trading_calendar()

    idx = _bisect_left(calendar, base)
    if idx >= len(calendar) or calendar[idx] <= base:
        # base_date is >= last trading day, or exactly the last known day
        if calendar[-1] <= base:
            raise ValueError(
                f"no next trading day after {base_date} "
                f"(latest: {calendar[-1]})"
            )
        # base_date is a gap after a known day but before the next —
        # insertion point is already the next trading day, so advance by 1
        idx += 1

    next_idx = idx + 1 if calendar[idx] <= base else idx

    if next_idx >= len(calendar):
        raise ValueError(
            f"no next trading day after {base_date} "
            f"(latest: {calendar[-1]})"
        )
    return calendar[next_idx]


def _bisect_left(a: list[datetime.date], x: datetime.date) -> int:
    """Return the insertion point for *x* in *a* to maintain sorted order.

    This is a manual bisect-left implementation for ``datetime.date`` lists
    to avoid a dependency on ``bisect`` for custom key comparisons.
    """
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
