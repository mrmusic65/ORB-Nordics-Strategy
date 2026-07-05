"""Analyze whether ORB performance is concentrated around earnings dates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
EARNINGS_PATH = ROOT / "data" / "earnings_dates.csv"
RESULT_PATH = OUTPUT_DIR / "report_day_analysis.csv"
MIN_RELATIVE_VOLUME = 3.0

TICKER_MAP = {
    "ABB": "ABB_5_min",
    "ALFA": "ALFA_LAVAL_5_MIN",
    "ASSA": "ASSA_5_min",
    "ATCO": "ATLAS_5_min",
    "BOL": "BOLIDEN_5_min",
    "ERIC": "ERICSSON_5_MIN",
    "ESSITY": "ESSITY_5_min",
    "GETI": "GETINGE_5_min",
    "SHB": "HANDELSBANKEN_5_min",
    "HEXA": "HEXAGON_5_min",
    "HM": "HM_5_min",
    "HUSQ": "HUSQVARNA_5_min",
    "INVE": "INVESTOR_5_MIN",
    "KINV": "KINNEVIK_5_min",
    "NDA": "NORDEA_5_min",
    "SAND": "SANDVIK_min",
    "SEB": "SEB_5_MIN",
    "SKF": "SKF_5_min",
    "SWED": "SWEDBANK_5_MIN",
    "VOLV": "VOLVO_5_MIN",
}
FEATURE_TO_EARNINGS = {feature: ticker for ticker, feature in TICKER_MAP.items()}

PERIODS = [
    ("2009-2011", pd.Timestamp("2009-01-01"), pd.Timestamp("2011-12-31")),
    ("2012-2014", pd.Timestamp("2012-01-01"), pd.Timestamp("2014-12-31")),
    ("2015-2017", pd.Timestamp("2015-01-01"), pd.Timestamp("2017-12-31")),
    ("2018-2020", pd.Timestamp("2018-01-01"), pd.Timestamp("2020-12-31")),
    ("2021-2023", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")),
]


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def parse_dates(values: pd.Series, source: Path) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.normalize()
    invalid = parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{source} contains invalid dates, for example: {examples}")
    return parsed


def load_trades() -> pd.DataFrame:
    paths = sorted(OUTPUT_DIR.glob("orb_trades_atr1.5_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No orb_trades_atr1.5_*.csv files found in {OUTPUT_DIR}"
        )

    required = {"date", "ticker", "net_pnl", "gross_pnl", "costs", "exit_type"}
    frames: list[pd.DataFrame] = []
    for path in paths:
        trades = pd.read_csv(path)
        missing = sorted(required - set(trades.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if "relative_volume" not in trades.columns:
            trades["relative_volume"] = pd.NA
        trades["_source_file"] = path.name
        frames.append(trades)

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = parse_dates(combined["date"], OUTPUT_DIR)
    for column in ("relative_volume", "net_pnl", "gross_pnl", "costs"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    if combined[["net_pnl", "gross_pnl", "costs"]].isna().any().any():
        raise ValueError("Trade files contain non-numeric PnL or cost values.")
    return combined


def load_feature_data(
    trade_tickers: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DatetimeIndex]]:
    frames: list[pd.DataFrame] = []
    calendars: dict[str, pd.DatetimeIndex] = {}

    for feature_prefix in trade_tickers:
        path = OUTPUT_DIR / f"{feature_prefix}_orb_daily_features.csv"
        if not path.exists():
            warn(
                f"No feature file found for trade ticker '{feature_prefix}': "
                f"{path.name}"
            )
            continue

        features = pd.read_csv(path, usecols=["date", "relative_volume"])
        features["date"] = parse_dates(features["date"], path)
        features["relative_volume"] = pd.to_numeric(
            features["relative_volume"], errors="coerce"
        )
        features["ticker"] = feature_prefix
        features = features.drop_duplicates(["ticker", "date"], keep="last")
        frames.append(features)
        calendars[feature_prefix] = pd.DatetimeIndex(
            features["date"].dropna().sort_values().unique()
        )

    if not frames:
        return (
            pd.DataFrame(columns=["date", "relative_volume", "ticker"]),
            calendars,
        )
    return pd.concat(frames, ignore_index=True), calendars


def add_relative_volume(
    trades: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    feature_volume = feature_data.rename(
        columns={"relative_volume": "_feature_relative_volume"}
    )
    merged = trades.merge(
        feature_volume[["ticker", "date", "_feature_relative_volume"]],
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    merged["relative_volume"] = merged["relative_volume"].fillna(
        merged["_feature_relative_volume"]
    )
    merged = merged.drop(columns=["_feature_relative_volume"])

    missing_count = int(merged["relative_volume"].isna().sum())
    if missing_count:
        warn(
            f"{missing_count} trades lack relative_volume in both trade and "
            "feature files and will be excluded."
        )
    return merged[merged["relative_volume"] >= MIN_RELATIVE_VOLUME].copy()


def report_window_dates(
    calendar: pd.DatetimeIndex,
    report_dates: pd.Series,
) -> set[pd.Timestamp]:
    eligible: set[pd.Timestamp] = set()
    if calendar.empty:
        return eligible

    for report_date in report_dates:
        if report_date < calendar[0] or report_date > calendar[-1]:
            continue
        insertion = int(calendar.searchsorted(report_date))
        if insertion < len(calendar) and calendar[insertion] == report_date:
            positions = (insertion - 1, insertion, insertion + 1)
        else:
            # If a report date is not a market session, use the immediately
            # preceding and following actual sessions.
            positions = (insertion - 1, insertion)
        for position in positions:
            if 0 <= position < len(calendar):
                eligible.add(pd.Timestamp(calendar[position]).normalize())
    return eligible


def add_report_day_flag(
    trades: pd.DataFrame,
    earnings: pd.DataFrame,
    calendars: dict[str, pd.DatetimeIndex],
) -> tuple[pd.DataFrame, int, int, int]:
    trade_tickers = sorted(trades["ticker"].dropna().astype(str).unique())
    missing_map = [ticker for ticker in trade_tickers if ticker not in FEATURE_TO_EARNINGS]
    for ticker in missing_map:
        warn(
            f"Trade ticker '{ticker}' has no exact entry in TICKER_MAP; "
            "its trades will be skipped in report-day logic."
        )

    trades = trades.copy()
    trades["earnings_ticker"] = trades["ticker"].map(FEATURE_TO_EARNINGS)
    earnings_tickers = set(earnings["ticker"].dropna().astype(str))
    mapped_tickers = set(trades["earnings_ticker"].dropna().astype(str))
    missing_earnings = sorted(mapped_tickers - earnings_tickers)
    for ticker in missing_earnings:
        feature_prefix = TICKER_MAP[ticker]
        warn(
            f"Mapped company '{feature_prefix}' ({ticker}) has no dates in "
            "earnings_dates.csv; its trades will be skipped."
        )

    matched_mask = trades["earnings_ticker"].isin(earnings_tickers)
    trades["is_report_day"] = pd.Series(pd.NA, index=trades.index, dtype="boolean")

    for earnings_ticker in sorted(mapped_tickers & earnings_tickers):
        feature_prefix = TICKER_MAP[earnings_ticker]
        calendar = calendars.get(feature_prefix, pd.DatetimeIndex([]))
        if calendar.empty:
            warn(
                f"No trading calendar is available for '{feature_prefix}'; "
                "its trades will be skipped."
            )
            matched_mask &= trades["earnings_ticker"] != earnings_ticker
            continue
        company_reports = earnings.loc[
            earnings["ticker"] == earnings_ticker, "report_date"
        ]
        eligible_dates = report_window_dates(calendar, company_reports)
        company_mask = trades["earnings_ticker"] == earnings_ticker
        trades.loc[company_mask, "is_report_day"] = trades.loc[
            company_mask, "date"
        ].isin(eligible_dates)

    matched_count = int(matched_mask.sum())
    missing_mapping_count = int(trades["earnings_ticker"].isna().sum())
    missing_earnings_count = int(
        (
            trades["earnings_ticker"].notna()
            & ~trades["earnings_ticker"].isin(earnings_tickers)
        ).sum()
    )
    return trades, matched_count, missing_mapping_count, missing_earnings_count


def summarize_group(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    known = trades[trades["is_report_day"].notna()].copy()
    if known.empty:
        return pd.DataFrame()

    summary = (
        known.groupby("is_report_day", observed=True)
        .agg(
            trade_count=("ticker", "size"),
            total_net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            avg_gross_pnl=("gross_pnl", "mean"),
            avg_cost=("costs", "mean"),
            hit_rate=("net_pnl", lambda values: (values > 0).mean()),
            stop_rate=("exit_type", lambda values: (values == "stop").mean()),
        )
        .reset_index()
    )
    summary["group"] = summary["is_report_day"].map(
        {True: "report_day", False: "other"}
    )
    summary["period"] = period
    return summary[
        [
            "period",
            "group",
            "trade_count",
            "total_net_pnl",
            "avg_net_pnl",
            "avg_gross_pnl",
            "avg_cost",
            "hit_rate",
            "stop_rate",
        ]
    ]


def build_summary(trades: pd.DataFrame) -> pd.DataFrame:
    frames = [summarize_group(trades, "all")]
    for label, start, end in PERIODS:
        period_trades = trades[
            (trades["date"] >= start) & (trades["date"] <= end)
        ]
        frames.append(summarize_group(period_trades, label))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "period",
                "group",
                "trade_count",
                "total_net_pnl",
                "avg_net_pnl",
                "avg_gross_pnl",
                "avg_cost",
                "hit_rate",
                "stop_rate",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def print_summary(summary: pd.DataFrame) -> None:
    display = summary.copy()
    for column in (
        "total_net_pnl",
        "avg_net_pnl",
        "avg_gross_pnl",
        "avg_cost",
    ):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    for column in ("hit_rate", "stop_rate"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print("\nReport-day analysis")
    print(display.to_string(index=False))


def main() -> None:
    if not EARNINGS_PATH.exists():
        raise FileNotFoundError(f"Earnings file not found: {EARNINGS_PATH}")

    earnings = pd.read_csv(EARNINGS_PATH)
    required_earnings = {"ticker", "report_date", "source"}
    missing = sorted(required_earnings - set(earnings.columns))
    if missing:
        raise ValueError(f"{EARNINGS_PATH} is missing required columns: {missing}")
    earnings["ticker"] = earnings["ticker"].astype(str).str.strip()
    earnings["report_date"] = parse_dates(
        earnings["report_date"], EARNINGS_PATH
    )

    trades = load_trades()
    feature_data, calendars = load_feature_data(
        sorted(trades["ticker"].dropna().astype(str).unique())
    )
    trades = add_relative_volume(trades, feature_data)
    (
        trades,
        matched_count,
        missing_mapping_count,
        missing_earnings_count,
    ) = add_report_day_flag(trades, earnings, calendars)

    summary = build_summary(trades)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_PATH, index=False)

    print(f"Trades after relative_volume >= {MIN_RELATIVE_VOLUME:.1f}: {len(trades)}")
    print(f"Trades matched to a company in earnings_dates.csv: {matched_count}")
    print(f"Trades dropped due to missing TICKER_MAP entry: {missing_mapping_count}")
    print(
        "Trades dropped due to missing earnings dates: "
        f"{missing_earnings_count}"
    )
    print_summary(summary)
    print(f"\nOutput: {RESULT_PATH}")


if __name__ == "__main__":
    main()
