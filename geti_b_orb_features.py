"""Validate and feature-engineer 5-minute GETI_B OHLCV data for ORB backtests.

Preferred input CSV columns:
    timestamp, open, high, low, close, volume

The loader also accepts common export variants such as "time" and "Volume".
Unix timestamps are treated as Europe/Stockholm wall-clock values when they are
numeric, matching the requested data contract. ISO timestamps with timezone
offsets are converted to Europe/Stockholm and then made timezone-naive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
CORPORATE_ACTION_GAP_THRESHOLD = 0.15
MARKET_TIMEZONE = "Europe/Stockholm"


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


@dataclass(frozen=True)
class OutputPaths:
    features: Path
    validation_log: Path
    corporate_actions_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate 5-minute Swedish equity OHLCV data and create strict "
            "no-look-ahead ORB features."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input 5-minute OHLCV CSV.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("geti_b_orb_daily_features.csv"),
        help="Daily feature output CSV.",
    )
    parser.add_argument(
        "--validation-log-csv",
        type=Path,
        default=Path("geti_b_validation_anomalies.csv"),
        help="CSV log for missing/duplicate/abnormal trading days.",
    )
    parser.add_argument(
        "--corporate-actions-log-csv",
        type=Path,
        default=Path("geti_b_possible_corporate_actions.csv"),
        help="CSV log for possible unadjusted corporate actions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of random feature rows to print for manual inspection.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for terminal sample output.",
    )
    return parser.parse_args()


def expected_intraday_index(day: pd.Timestamp) -> pd.DatetimeIndex:
    # Timestamps are bar starts. A full session ending 17:30 therefore has its
    # last 5-minute bar starting at 17:25 and covering 17:25-17:30.
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
        # Requested contract: numeric Unix-like values have already been
        # converted to Europe/Stockholm wall-clock time, so do not shift them.
        return pd.to_datetime(numeric_timestamp, unit=unit)

    parsed = pd.to_datetime(timestamp, utc=True, errors="raise")
    return parsed.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None)


def load_ohlcv(input_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
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
                        actual_count=actual_count,
                        expected_count=expected_count,
                        duplicate_count=duplicate_count,
                        missing_count=missing_count,
                        extra_count=extra_count,
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

    for date_value, day_df in df.groupby("date", sort=True):
        day_df = day_df.sort_values("datetime")
        regular_session_df = day_df[
            (day_df["datetime"].dt.time >= open_time)
            & (day_df["datetime"].dt.time <= last_regular_bar_start_time().time())
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
            # Daily open is defined here as the close of the 09:00 bar.
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


def detect_possible_corporate_actions(daily: pd.DataFrame) -> pd.DataFrame:
    actions = daily[["date", "daily_open", "daily_close"]].copy()
    actions["previous_date"] = actions["date"].shift(1)
    actions["previous_close"] = actions["daily_close"].shift(1)
    actions["open_gap_pct"] = (
        actions["daily_open"] / actions["previous_close"] - 1.0
    )
    actions = actions[
        actions["open_gap_pct"].abs() > CORPORATE_ACTION_GAP_THRESHOLD
    ].copy()
    actions["reason"] = (
        "close_t_minus_1_to_open_t_gap_gt_15pct_possible_unadjusted_corporate_action"
    )
    return actions[
        [
            "date",
            "previous_date",
            "previous_close",
            "daily_open",
            "open_gap_pct",
            "reason",
        ]
    ]


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

    features = bar1.merge(daily, on="date", how="left").sort_values("date")

    # NO LOOK-AHEAD: shift(1) moves today's bar1 volume out of today's rolling
    # window. avg_volume_14d for day N can only see days N-14 through N-1.
    features["avg_volume_14d"] = (
        features["bar1_volume"].shift(1).rolling(ROLLING_DAYS).mean()
    )
    features["relative_volume"] = (
        features["bar1_volume"] / features["avg_volume_14d"]
    )

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
    # shifted before rolling. atr_14d for day N therefore uses only completed
    # trading days before N.
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
    """Programmatically verify rolling features exclude the current row.

    This is intentionally strict and mirrors the production formulas using an
    explicit previous-days slice. A future edit that accidentally removes
    shift(1) will fail here.
    """

    for idx in range(len(features)):
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
                    f"avg_volume_14d at row {idx} should be NaN before 14 prior days"
                )
        elif not np.isclose(actual_avg_volume, expected_avg_volume, equal_nan=True):
            raise AssertionError(
                f"avg_volume_14d at row {idx} includes wrong data; "
                f"expected {expected_avg_volume}, got {actual_avg_volume}"
            )

        if pd.isna(expected_atr):
            if not pd.isna(actual_atr):
                raise AssertionError(
                    f"atr_14d at row {idx} should be NaN before 14 prior days"
                )
        elif not np.isclose(actual_atr, expected_atr, equal_nan=True):
            raise AssertionError(
                f"atr_14d at row {idx} includes wrong data; "
                f"expected {expected_atr}, got {actual_atr}"
            )


def print_summary(
    df: pd.DataFrame,
    validation_log: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    features: pd.DataFrame,
    excluded_half_days: int,
    excluded_invalid_exit_days: int,
    sample_size: int,
    random_state: int,
) -> None:
    dates = pd.to_datetime(sorted(df["date"].unique()))
    flagged_too_few = (
        int(validation_log["too_few_bars"].sum()) if not validation_log.empty else 0
    )
    flagged_too_many = (
        int(validation_log["too_many_bars"].sum()) if not validation_log.empty else 0
    )
    flagged_gaps = (
        int((validation_log["missing_bars"] > 0).sum())
        if not validation_log.empty
        else 0
    )
    flagged_duplicates = (
        int((validation_log["duplicate_rows"] > 0).sum())
        if not validation_log.empty
        else 0
    )

    print("\nSAMMANFATTNING")
    print(f"Antal handelsdagar: {len(dates)}")
    print(f"Datumintervall: {dates.min().date()} till {dates.max().date()}")
    print(f"Dagar med för få bars: {flagged_too_few}")
    print(f"Dagar med för många bars: {flagged_too_many}")
    print(f"Dagar med luckor: {flagged_gaps}")
    print(f"Dagar med dubbletter: {flagged_duplicates}")
    print(f"Möjliga ojusterade corporate actions: {len(corporate_actions)}")

    print(f"Antal exkluderade halvdagar: {excluded_half_days}")
    print(f"Antal exkluderade dagar utan giltig exit: {excluded_invalid_exit_days}")
    print("Fem vanligaste exit_time i feature-output:")
    if features.empty:
        print("Ingen feature-output")
    else:
        print(features["exit_time"].value_counts().head(5).to_string())

    sample_n = min(sample_size, len(features))
    if sample_n:
        print(f"\nSLUMPMÄSSIGT URVAL ({sample_n} rader)")
        sample = features.sample(sample_n, random_state=random_state).sort_values("date")
        print(sample.to_string(index=False))


def write_outputs(
    features: pd.DataFrame,
    validation_log: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    for path in [
        paths.features,
        paths.validation_log,
        paths.corporate_actions_log,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(paths.features, index=False)
    validation_log.to_csv(paths.validation_log, index=False)
    corporate_actions.to_csv(paths.corporate_actions_log, index=False)


def run(
    input_csv: Path,
    output_csv: Path,
    validation_log_csv: Path,
    corporate_actions_log_csv: Path,
    sample_size: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    df = load_ohlcv(input_csv)
    validation_log = validate_intraday_bars(df)
    daily = make_daily_ohlcv(df)
    excluded_half_days = int((daily["is_full_day"] == False).sum())
    excluded_invalid_exit_days = int((daily["has_valid_exit"] == False).sum())
    daily = daily[daily["is_full_day"] & daily["has_valid_exit"]].copy()
    corporate_actions = detect_possible_corporate_actions(daily)
    bar1 = extract_bar1(df)
    bar1 = bar1[bar1["date"].isin(daily["date"])].copy()
    features = add_no_lookahead_features(bar1, daily)

    paths = OutputPaths(
        features=output_csv,
        validation_log=validation_log_csv,
        corporate_actions_log=corporate_actions_log_csv,
    )
    write_outputs(features, validation_log, corporate_actions, paths)
    print_summary(
        df=df,
        validation_log=validation_log,
        corporate_actions=corporate_actions,
        features=features,
        excluded_half_days=excluded_half_days,
        excluded_invalid_exit_days=excluded_invalid_exit_days,
        sample_size=sample_size,
        random_state=random_state,
    )
    print(f"\nSkrev feature-CSV: {paths.features}")
    print(f"Skrev valideringslogg: {paths.validation_log}")
    print(f"Skrev corporate-actions-logg: {paths.corporate_actions_log}")

    return features


def main() -> None:
    args = parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        validation_log_csv=args.validation_log_csv,
        corporate_actions_log_csv=args.corporate_actions_log_csv,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
