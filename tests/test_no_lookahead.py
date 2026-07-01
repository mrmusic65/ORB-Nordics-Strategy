from __future__ import annotations

import pandas as pd

from orb_features_pipeline import (
    ROLLING_DAYS,
    add_no_lookahead_features,
    assert_no_lookahead,
    make_daily_ohlcv,
)


def test_rolling_features_exclude_current_day_data() -> None:
    dates = pd.bdate_range("2024-01-01", periods=ROLLING_DAYS + 2)
    bar1_volumes = [100.0] * ROLLING_DAYS + [10_000.0, 200.0]
    daily_ranges = [2.0] * ROLLING_DAYS + [1_000.0, 4.0]

    bar1 = pd.DataFrame(
        {
            "date": dates,
            "bar1_open": 100.0,
            "bar1_high": 101.0,
            "bar1_low": 99.0,
            "bar1_close": 100.5,
            "bar1_volume": bar1_volumes,
        }
    )
    daily = pd.DataFrame(
        {
            "date": dates,
            "daily_open": 100.0,
            "daily_high": [100.0 + value for value in daily_ranges],
            "daily_low": 100.0,
            "daily_close": 100.0,
            "daily_volume": 1_000.0,
            "last_available_bar_time": "17:20",
            "exit_time": "17:20",
            "has_valid_exit": True,
            "is_full_day": True,
        }
    )

    features = add_no_lookahead_features(bar1, daily)

    first_calculable_row = ROLLING_DAYS
    assert features.loc[first_calculable_row, "bar1_volume"] == 10_000.0
    assert features.loc[first_calculable_row, "avg_volume_14d"] == 100.0
    assert features.loc[first_calculable_row, "atr_14d"] == 2.0
    assert features.loc[first_calculable_row, "relative_volume"] == 100.0

    second_calculable_row = ROLLING_DAYS + 1
    expected_avg_including_previous_spike = (13 * 100.0 + 10_000.0) / ROLLING_DAYS
    expected_atr_including_previous_spike = (13 * 2.0 + 1_000.0) / ROLLING_DAYS
    assert (
        features.loc[second_calculable_row, "avg_volume_14d"]
        == expected_avg_including_previous_spike
    )
    assert features.loc[second_calculable_row, "atr_14d"] == expected_atr_including_previous_spike


def test_assert_no_lookahead_rejects_current_day_rolling_values() -> None:
    dates = pd.bdate_range("2024-01-01", periods=ROLLING_DAYS + 1)
    features = pd.DataFrame(
        {
            "date": dates,
            "bar1_volume": [100.0] * ROLLING_DAYS + [10_000.0],
            "true_range": [2.0] * ROLLING_DAYS + [1_000.0],
            "avg_volume_14d": [float("nan")] * ROLLING_DAYS + [(13 * 100.0 + 10_000.0) / ROLLING_DAYS],
            "atr_14d": [float("nan")] * ROLLING_DAYS + [(13 * 2.0 + 1_000.0) / ROLLING_DAYS],
        }
    )

    try:
        assert_no_lookahead(features)
    except AssertionError as exc:
        assert "2024-01-19" in str(exc)
        assert "leaked" in str(exc)
    else:
        raise AssertionError("assert_no_lookahead accepted current-day leakage")


def test_daily_ohlcv_uses_orb_session_rules() -> None:
    day = pd.Timestamp("2024-01-02")
    raw = pd.DataFrame(
        {
            "datetime": [
                day + pd.Timedelta(hours=9),
                day + pd.Timedelta(hours=9, minutes=5),
                day + pd.Timedelta(hours=16, minutes=55),
                day + pd.Timedelta(hours=17, minutes=35),
            ],
            "open": [10.0, 11.0, 12.0, 99.0],
            "high": [12.0, 13.0, 14.0, 200.0],
            "low": [9.0, 8.0, 7.0, 1.0],
            "close": [11.5, 12.5, 13.5, 199.0],
            "volume": [100.0, 200.0, 300.0, 9_999.0],
        }
    )
    raw["timestamp"] = raw["datetime"]
    raw["date"] = raw["datetime"].dt.date

    daily = make_daily_ohlcv(raw)

    assert daily.loc[0, "daily_open"] == 11.5
    assert daily.loc[0, "daily_high"] == 14.0
    assert daily.loc[0, "daily_low"] == 7.0
    assert daily.loc[0, "daily_close"] == 13.5
    assert daily.loc[0, "daily_volume"] == 600.0
    assert daily.loc[0, "last_available_bar_time"] == "16:55"
    assert daily.loc[0, "exit_time"] == "16:55"
    assert daily.loc[0, "has_valid_exit"] == False
    assert daily.loc[0, "is_full_day"] == False


def test_daily_ohlcv_uses_1720_exit_and_1715_fallback() -> None:
    day_one = pd.Timestamp("2024-01-02")
    day_two = pd.Timestamp("2024-01-03")
    raw = pd.DataFrame(
        {
            "datetime": [
                day_one + pd.Timedelta(hours=9),
                day_one + pd.Timedelta(hours=17, minutes=15),
                day_one + pd.Timedelta(hours=17, minutes=20),
                day_one + pd.Timedelta(hours=17, minutes=25),
                day_two + pd.Timedelta(hours=9),
                day_two + pd.Timedelta(hours=17, minutes=15),
                day_two + pd.Timedelta(hours=17, minutes=25),
            ],
            "open": [10.0, 10.0, 10.0, 99.0, 20.0, 20.0, 99.0],
            "high": [11.0, 12.0, 13.0, 200.0, 21.0, 22.0, 200.0],
            "low": [9.0, 8.0, 7.0, 1.0, 19.0, 18.0, 1.0],
            "close": [10.5, 11.5, 12.5, 199.0, 20.5, 21.5, 199.0],
            "volume": [100.0, 200.0, 300.0, 9_999.0, 400.0, 500.0, 9_999.0],
        }
    )
    raw["timestamp"] = raw["datetime"]
    raw["date"] = raw["datetime"].dt.date

    daily = make_daily_ohlcv(raw)

    assert daily.loc[0, "daily_close"] == 12.5
    assert daily.loc[0, "exit_time"] == "17:20"
    assert daily.loc[0, "daily_high"] == 13.0
    assert daily.loc[0, "daily_low"] == 7.0
    assert daily.loc[0, "has_valid_exit"] == True
    assert daily.loc[0, "is_full_day"] == True

    assert daily.loc[1, "daily_close"] == 21.5
    assert daily.loc[1, "exit_time"] == "17:15"
    assert daily.loc[1, "last_available_bar_time"] == "17:25"
    assert daily.loc[1, "has_valid_exit"] == True
    assert daily.loc[1, "is_full_day"] == True
