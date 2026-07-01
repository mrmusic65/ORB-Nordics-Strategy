"""Generic ORB feature pipeline for 5-minute Swedish equity OHLCV data.

Usage:
    python orb_features_pipeline.py <csv_path> <ticker_name>

Outputs are written to output/:
    <ticker_name>_orb_daily_features.csv
    <ticker_name>_flags.csv
    pipeline_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TIMESTAMP_ALIASES = ["timestamp", "time", "datetime", "date"]
VOLUME_ALIASES = ["volume", "Volume", "vol", "Vol"]
MARKET_OPEN = "09:00"
MARKET_CLOSE = "17:30"
VALID_SESSION_MIN_LAST_BAR = "17:20"
EXIT_TARGET_TIME = "17:20"
EXIT_FALLBACK_MIN_TIME = "17:15"
BAR_FREQ = "5min"
ROLLING_DAYS = 14
MARKET_TIMEZONE = "Europe/Stockholm"
OUTPUT_DIR = Path("output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create no-look-ahead ORB daily features from 5-minute OHLCV data."
    )
    parser.add_argument("csv_path", type=Path, help="Path to input 5-minute CSV.")
    parser.add_argument("ticker_name", help="Ticker used in output filenames and logs.")
    return parser.parse_args()


def sanitize_ticker(ticker_name: str) -> str:
    ticker = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker_name.strip())
    if not ticker:
        raise ValueError("ticker_name cannot be empty after sanitization")
    return ticker


def market_open_time() -> pd.Timestamp:
    return pd.Timestamp(MARKET_OPEN)


def last_regular_bar_start_time() -> pd.Timestamp:
    return pd.Timestamp(MARKET_CLOSE) - pd.Timedelta(BAR_FREQ)


def valid_session_min_last_bar_time() -> pd.Timestamp:
    return pd.Timestamp(VALID_SESSION_MIN_LAST_BAR)


def exit_target_time() -> pd.Timestamp:
    return pd.Timestamp(EXIT_TARGET_TIME)


def exit_fallback_min_time() -> pd.Timestamp:
    return pd.Timestamp(EXIT_FALLBACK_MIN_TIME)


def expected_intraday_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    last_bar_start = pd.Timestamp(f"{day.date()} {MARKET_CLOSE}") - pd.Timedelta(BAR_FREQ)
    return pd.date_range(
        f"{day.date()} {MARKET_OPEN}",
        last_bar_start,
        freq=BAR_FREQ,
    )


def infer_unix_unit(timestamp: pd.Series) -> str:
    max_abs = timestamp.dropna().astype("int64").abs().max()
    return "ms" if max_abs > 10_000_000_000 else "s"


def pick_column(columns: pd.Index, candidates: list[str], canonical_name: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Input CSV is missing required column '{canonical_name}'. "
        f"Tried aliases: {candidates}"
    )


def parse_timestamp_column(timestamp: pd.Series) -> pd.Series:
    numeric_timestamp = pd.to_numeric(timestamp, errors="coerce")
    if numeric_timestamp.notna().all():
        unit = infer_unix_unit(numeric_timestamp)
        return pd.to_datetime(numeric_timestamp, unit=unit)

    parsed = pd.to_datetime(timestamp, utc=True, errors="raise")
    return parsed.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None)


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    timestamp_column = pick_column(raw.columns, TIMESTAMP_ALIASES, "timestamp")
    volume_column = pick_column(raw.columns, VOLUME_ALIASES, "volume")

    missing_price_columns = sorted(set(["open", "high", "low", "close"]) - set(raw.columns))
    if missing_price_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_price_columns}")

    df = raw[[timestamp_column, "open", "high", "low", "close", volume_column]].copy()
    df = df.rename(columns={timestamp_column: "timestamp", volume_column: "volume"})
    df["datetime"] = parse_timestamp_column(df["timestamp"])
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime").reset_index(drop=True)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        bad_counts = df[numeric_columns].isna().sum()
        raise ValueError(f"Found non-numeric OHLCV values:\n{bad_counts}")

    return df


def validate_intraday_bars(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for date_value, day_df in df.groupby("date", sort=True):
        day = pd.Timestamp(date_value)
        expected = expected_intraday_index(day)
        actual = pd.DatetimeIndex(day_df["datetime"])
        duplicate_mask = actual.duplicated(keep=False)
        unique_actual = pd.DatetimeIndex(actual.drop_duplicates())
        missing = expected.difference(unique_actual)
        extra = unique_actual.difference(expected)

        actual_count = len(actual)
        expected_count = len(expected)
        duplicate_count = int(duplicate_mask.sum())
        missing_count = len(missing)
        extra_count = len(extra)

        if (
            actual_count != expected_count
            or duplicate_count
            or missing_count
            or extra_count
        ):
            records.append(
                {
                    "date": date_value,
                    "expected_bars": expected_count,
                    "actual_bars": actual_count,
                    "missing_bars": missing_count,
                    "duplicate_rows": duplicate_count,
                    "extra_bars": extra_count,
                    "too_few_bars": actual_count < expected_count,
                    "too_many_bars": actual_count > expected_count,
                    "first_missing": missing[0] if missing_count else pd.NaT,
                    "first_extra": extra[0] if extra_count else pd.NaT,
                    "issue": classify_validation_issue(
                        actual_count,
                        expected_count,
                        duplicate_count,
                        missing_count,
                        extra_count,
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


def classify_validation_issue(
    actual_count: int,
    expected_count: int,
    duplicate_count: int,
    missing_count: int,
    extra_count: int,
) -> str:
    issue_parts: list[str] = []
    if actual_count < expected_count:
        issue_parts.append("too_few_bars_possible_half_day_or_halt")
    if actual_count > expected_count:
        issue_parts.append("too_many_bars_possible_duplicates_or_dst_error")
    if duplicate_count:
        issue_parts.append("duplicates")
    if missing_count:
        issue_parts.append("gaps")
    if extra_count:
        issue_parts.append("outside_expected_session")
    return ";".join(issue_parts)


def make_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    open_time = market_open_time().time()
    valid_min_last_time = valid_session_min_last_bar_time().time()
    exit_time = exit_target_time().time()
    fallback_min_time = exit_fallback_min_time().time()
    last_regular_time = last_regular_bar_start_time().time()

    for date_value, day_df in df.groupby("date", sort=True):
        day_df = day_df.sort_values("datetime")
        regular_session_df = day_df[
            (day_df["datetime"].dt.time >= open_time)
            & (day_df["datetime"].dt.time <= last_regular_time)
        ]
        session_df = day_df[
            (day_df["datetime"].dt.time >= open_time)
            & (day_df["datetime"].dt.time <= exit_time)
        ]

        if regular_session_df.empty or session_df.empty:
            continue

        last_available_bar_time = regular_session_df.iloc[-1]["datetime"].time()
        is_full_day = last_available_bar_time >= valid_min_last_time

        open_bar = session_df[session_df["datetime"].dt.time == open_time]
        daily_open = np.nan
        if not open_bar.empty:
            daily_open = open_bar.iloc[0]["close"]

        target_exit_bar = session_df[session_df["datetime"].dt.time == exit_time]
        has_valid_exit = True
        if target_exit_bar.empty:
            exit_candidates = session_df[
                (session_df["datetime"].dt.time >= fallback_min_time)
                & (session_df["datetime"].dt.time <= exit_time)
            ]
            if exit_candidates.empty:
                has_valid_exit = False
                close_bar = session_df.iloc[-1]
            else:
                close_bar = exit_candidates.iloc[-1]
        else:
            close_bar = target_exit_bar.iloc[-1]

        selected_exit_time = close_bar["datetime"].time()

        records.append(
            {
                "date": pd.Timestamp(date_value),
                "daily_open": daily_open,
                "daily_high": session_df["high"].max(),
                "daily_low": session_df["low"].min(),
                "daily_close": close_bar["close"],
                "daily_volume": session_df["volume"].sum(),
                "last_available_bar_time": last_available_bar_time.strftime("%H:%M"),
                "exit_time": selected_exit_time.strftime("%H:%M"),
                "has_valid_exit": has_valid_exit,
                "is_full_day": is_full_day,
            }
        )

    return pd.DataFrame.from_records(records)


def extract_bar1(df: pd.DataFrame) -> pd.DataFrame:
    bar1_time = pd.to_datetime(MARKET_OPEN).time()
    bar1 = df[df["datetime"].dt.time == bar1_time].copy()
    bar1 = bar1.sort_values("datetime").drop_duplicates("date", keep="first")
    bar1["date"] = pd.to_datetime(bar1["date"])
    return bar1.rename(
        columns={
            "open": "bar1_open",
            "high": "bar1_high",
            "low": "bar1_low",
            "close": "bar1_close",
            "volume": "bar1_volume",
        }
    )[
        [
            "date",
            "bar1_open",
            "bar1_high",
            "bar1_low",
            "bar1_close",
            "bar1_volume",
        ]
    ]


def add_no_lookahead_features(bar1: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if "is_full_day" in daily.columns and not daily["is_full_day"].all():
        raise AssertionError("Half-days must be excluded before feature calculation")
    if "has_valid_exit" in daily.columns and not daily["has_valid_exit"].all():
        raise AssertionError("Days without a valid exit must be excluded before feature calculation")

    features = bar1.merge(daily, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # NO LOOK-AHEAD: shift(1) moves today's bar1 volume out of today's rolling
    # window. avg_volume_14d for day N can only see days N-14 through N-1.
    features["avg_volume_14d"] = (
        features["bar1_volume"].shift(1).rolling(ROLLING_DAYS).mean()
    )
    features["relative_volume"] = features["bar1_volume"] / features["avg_volume_14d"]

    previous_close = features["daily_close"].shift(1)
    true_range = pd.concat(
        [
            features["daily_high"] - features["daily_low"],
            (features["daily_high"] - previous_close).abs(),
            (features["daily_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # NO LOOK-AHEAD: today's true range uses today's high/low, so it must be
    # shifted before rolling. atr_14d for day N uses only completed prior days.
    features["true_range"] = true_range
    features["atr_14d"] = features["true_range"].shift(1).rolling(ROLLING_DAYS).mean()

    assert_no_lookahead(features)

    return features[
        [
            "date",
            "bar1_open",
            "bar1_high",
            "bar1_low",
            "bar1_close",
            "bar1_volume",
            "last_available_bar_time",
            "exit_time",
            "is_full_day",
            "avg_volume_14d",
            "relative_volume",
            "atr_14d",
        ]
    ]


def assert_no_lookahead(features: pd.DataFrame) -> None:
    for idx in range(len(features)):
        date_value = features.iloc[idx].get("date", f"row_{idx}")
        expected_avg_volume = np.nan
        expected_atr = np.nan

        if idx >= ROLLING_DAYS:
            previous_window = features.iloc[idx - ROLLING_DAYS : idx]
            expected_avg_volume = previous_window["bar1_volume"].mean()
            expected_atr = previous_window["true_range"].mean()

        actual_avg_volume = features.iloc[idx]["avg_volume_14d"]
        actual_atr = features.iloc[idx]["atr_14d"]

        if pd.isna(expected_avg_volume):
            if not pd.isna(actual_avg_volume):
                raise AssertionError(
                    f"date {date_value}: avg_volume_14d should be NaN before 14 prior days"
                )
        elif not np.isclose(actual_avg_volume, expected_avg_volume, equal_nan=True):
            raise AssertionError(
                f"date {date_value}: avg_volume_14d leaked or used wrong window; "
                f"expected {expected_avg_volume}, got {actual_avg_volume}"
            )

        if pd.isna(expected_atr):
            if not pd.isna(actual_atr):
                raise AssertionError(
                    f"date {date_value}: atr_14d should be NaN before 14 prior days"
                )
        elif not np.isclose(actual_atr, expected_atr, equal_nan=True):
            raise AssertionError(
                f"date {date_value}: atr_14d leaked or used wrong window; "
                f"expected {expected_atr}, got {actual_atr}"
            )


def build_flags(
    ticker: str,
    daily_all: pd.DataFrame,
) -> pd.DataFrame:
    flag_frames: list[pd.DataFrame] = []

    if not daily_all.empty:
        excluded = daily_all[(daily_all["is_full_day"] == False) | (daily_all["has_valid_exit"] == False)].copy()
        if not excluded.empty:
            excluded["ticker"] = ticker
            excluded["flag_type"] = np.where(
                excluded["is_full_day"] == False,
                "excluded_halfday",
                "excluded_invalid_exit",
            )
            excluded["reason"] = np.where(
                excluded["is_full_day"] == False,
                "last_available_bar_before_17_20",
                "missing_exit_bar_17_20_or_fallback_17_15",
            )
            flag_frames.append(
                excluded[
                    [
                        "ticker",
                        "date",
                        "flag_type",
                        "reason",
                        "last_available_bar_time",
                        "exit_time",
                        "is_full_day",
                        "has_valid_exit",
                    ]
                ]
            )

    if not flag_frames:
        return pd.DataFrame(columns=["ticker", "date", "flag_type", "reason"])

    return pd.concat(flag_frames, ignore_index=True, sort=False)


def make_summary_row(
    ticker: str,
    df: pd.DataFrame,
    features: pd.DataFrame,
    excluded_halfdays: int,
) -> dict[str, object]:
    dates = pd.to_datetime(sorted(df["date"].unique()))
    exit_counts = features["exit_time"].value_counts() if not features.empty else pd.Series(dtype=int)
    feature_count = len(features)
    return {
        "ticker": ticker,
        "start_date": dates.min().date() if len(dates) else pd.NaT,
        "end_date": dates.max().date() if len(dates) else pd.NaT,
        "total_days": len(dates),
        "excluded_halfdays": excluded_halfdays,
        "pct_exit_1720": exit_counts.get("17:20", 0) / feature_count if feature_count else 0.0,
        "pct_exit_1715": exit_counts.get("17:15", 0) / feature_count if feature_count else 0.0,
    }


def upsert_summary_row(summary_path: Path, row: dict[str, object]) -> None:
    columns = [
        "ticker",
        "start_date",
        "end_date",
        "total_days",
        "excluded_halfdays",
        "pct_exit_1720",
        "pct_exit_1715",
    ]
    new_row = pd.DataFrame([row], columns=columns)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["ticker"] != row["ticker"]]
        summary = pd.concat([summary, new_row], ignore_index=True)
    else:
        summary = new_row

    summary = summary.reindex(columns=columns)
    summary = summary.sort_values("ticker")
    summary.to_csv(summary_path, index=False)


def run_pipeline(csv_path: Path, ticker_name: str) -> pd.DataFrame:
    ticker = sanitize_ticker(ticker_name)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(csv_path)
    validate_intraday_bars(df)
    daily_all = make_daily_ohlcv(df)
    excluded_halfdays = int((daily_all["is_full_day"] == False).sum())
    daily = daily_all[daily_all["is_full_day"] & daily_all["has_valid_exit"]].copy()

    bar1 = extract_bar1(df)
    bar1 = bar1[bar1["date"].isin(daily["date"])].copy()

    try:
        features = add_no_lookahead_features(bar1, daily)
    except AssertionError as exc:
        raise RuntimeError(f"No-look-ahead assertion failed for {ticker}: {exc}") from exc

    features_path = output_dir / f"{ticker}_orb_daily_features.csv"
    flags_path = output_dir / f"{ticker}_flags.csv"
    summary_path = output_dir / "pipeline_summary.csv"

    flags = build_flags(ticker, daily_all)
    summary_row = make_summary_row(
        ticker=ticker,
        df=df,
        features=features,
        excluded_halfdays=excluded_halfdays,
    )

    features.to_csv(features_path, index=False)
    flags.to_csv(flags_path, index=False)
    upsert_summary_row(summary_path, summary_row)

    print_ticker_summary(ticker, summary_row, features_path, flags_path)
    return features


def print_ticker_summary(
    ticker: str,
    summary_row: dict[str, object],
    features_path: Path,
    flags_path: Path,
) -> None:
    print(f"\n[{ticker}] klar")
    print(f"Datumintervall: {summary_row['start_date']} till {summary_row['end_date']}")
    print(f"Antal handelsdagar: {summary_row['total_days']}")
    print(f"Exkluderade halvdagar: {summary_row['excluded_halfdays']}")
    print(f"Andel exit_time 17:20: {summary_row['pct_exit_1720']:.2%}")
    print(f"Andel exit_time 17:15: {summary_row['pct_exit_1715']:.2%}")
    print(f"Feature-CSV: {features_path}")
    print(f"Flags-CSV: {flags_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.csv_path, args.ticker_name)


if __name__ == "__main__":
    main()
