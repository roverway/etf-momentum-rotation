"""Tests for calendar module (A-share trading calendar)."""
import datetime

import pytest

from trading_calendar import (
    get_next_trading_date,
    get_previous_trading_date,
    load_trading_calendar,
)


@pytest.fixture(scope="module")
def trading_calendar():
    """Module-scoped fixture to cache the calendar (one AKShare call per module)."""
    return load_trading_calendar()


class TestLoadTradingCalendar:
    def test_returns_list_of_dates(self, trading_calendar):
        assert isinstance(trading_calendar, list)
        assert len(trading_calendar) > 0
        assert all(isinstance(d, datetime.date) for d in trading_calendar)

    def test_returns_over_4000_trading_days(self, trading_calendar):
        assert len(trading_calendar) > 4000, (
            f"Expected >4000 trading days, got {len(trading_calendar)}"
        )

    def test_sorted_ascending(self, trading_calendar):
        assert trading_calendar == sorted(trading_calendar)

    def test_cache_works(self):
        """Calling load_trading_calendar twice returns the same list object."""
        cal1 = load_trading_calendar()
        cal2 = load_trading_calendar()
        assert cal1 is cal2, "Module-level cache not working"


class TestGetPreviousTradingDate:
    def test_previous_trading_day_string_input(self):
        """2026-07-06 (Mon) -> previous should be 2026-07-03 (Fri)."""
        result = get_previous_trading_date("2026-07-06", n=1)
        assert isinstance(result, datetime.date)
        assert result == datetime.date(2026, 7, 3), f"Got {result}"

    def test_previous_trading_day_date_input(self, trading_calendar):
        base = datetime.date(2026, 7, 6)
        result = get_previous_trading_date(base, n=1)
        assert result == datetime.date(2026, 7, 3), f"Got {result}"

    def test_previous_n_equals_2(self):
        """2026-07-06 -> 2 trading days before should be 2026-07-02 (Thu)."""
        result = get_previous_trading_date("2026-07-06", n=2)
        assert result == datetime.date(2026, 7, 2), f"Got {result}"

    def test_previous_n_equals_5_one_week_back(self):
        """2026-07-06 -> 5 trading days before should be 2026-06-29 (Mon)."""
        result = get_previous_trading_date("2026-07-06", n=5)
        assert result == datetime.date(2026, 6, 29), f"Got {result}"

    def test_base_date_is_trading_day_itself(self):
        """When base_date itself is a trading day, n=1 returns previous trading day."""
        result = get_previous_trading_date("2026-07-03", n=1)  # Fri
        assert result == datetime.date(2026, 7, 2), f"Got {result}"


class TestGetNextTradingDate:
    def test_next_trading_day_string_input(self):
        """2026-07-03 (Fri) -> next should be 2026-07-06 (Mon)."""
        result = get_next_trading_date("2026-07-03")
        assert isinstance(result, datetime.date)
        assert result == datetime.date(2026, 7, 6), f"Got {result}"

    def test_next_trading_day_date_input(self, trading_calendar):
        base = datetime.date(2026, 7, 3)
        result = get_next_trading_date(base)
        assert result == datetime.date(2026, 7, 6), f"Got {result}"

    def test_next_from_midweek(self):
        """2026-07-02 (Thu) -> next should be 2026-07-03 (Fri)."""
        result = get_next_trading_date("2026-07-02")
        assert result == datetime.date(2026, 7, 3), f"Got {result}"

    def test_next_from_monday(self):
        """2026-07-06 (Mon) -> next should be 2026-07-07 (Tue)."""
        result = get_next_trading_date("2026-07-06")
        assert result == datetime.date(2026, 7, 7), f"Got {result}"

    def test_base_date_is_last_trading_day_in_calendar_raises(
        self, trading_calendar
    ):
        """Last known trading day should raise ValueError (no next day)."""
        last_day = trading_calendar[-1]
        with pytest.raises(ValueError, match="no next trading day"):
            get_next_trading_date(last_day)
