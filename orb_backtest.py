"""Daily Opening Range Breakout backtest from pipeline feature CSVs.

Usage:
    py orb_backtest.py <feature_dir> <intraday_csv_dir> <atr_multiplier> [train_start train_end]

Outputs:
    output/orb_trades_atr{multiplier}_{start}_{end}.csv
    output/orb_daily_summary_atr{multiplier}_{start}_{end}.csv
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from orb_features_pipeline import load_ohlcv, sanitize_ticker


FEATURE_SUFFIX = "_orb_daily_features.csv"
OUTPUT_DIR = Path("output")
START_CAPITAL = 100_000.0
RISK_FRACTION = 0.01
MAX_LEVERAGE = 4.0
COST_BPS_PER_SIDE = 6.0
ENTRY_START = "09:05"
ENTRY_END = "17:15"
EOD_EXIT_TIME = "17:20"


@dataclass(frozen=True)
class TradeCandidate:
    date: pd.Timestamp
    ticker: str
    direction: str
    entry_price: float
    stop_price: float
    exit_price: float
    exit_type: str
    entry_bar_time: str
    atr_multiplier: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a daily ORB backtest using pipeline feature CSVs."
    )
    parser.add_argument(
        "feature_dir",
        type=Path,
        help="Folder containing *_orb_daily_features.csv files.",
    )
    parser.add_argument(
        "intraday_csv_dir",
        type=Path,
        help="Folder containing the original 5-minute OHLCV CSV files.",
    )
    parser.add_argument(
        "atr_multiplier",
        type=float,
        nargs="?",
        default=1.0,
        help="ATR stop multiplier. Default: 1.0.",
    )
    parser.add_argument(
        "train_start",
        nargs="?",
        default=None,
        help="Optional period start date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "train_end",
        nargs="?",
        default=None,
        help="Optional period end date, YYYY-MM-DD.",
    )
    return parser.parse_args()


def ticker_from_feature_path(path: Path) -> str:
    if not path.name.endswith(FEATURE_SUFFIX):
        raise ValueError(f"Unexpected feature filename: {path.name}")
    return path.name[: -len(FEATURE_SUFFIX)]


def build_intraday_file_map(intraday_csv_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for csv_path in intraday_csv_dir.glob("*.csv"):
        ticker = sanitize_ticker(csv_path.stem)
        mapping[ticker] = csv_path
    return mapping


def load_features(feature_path: Path) -> pd.DataFrame:
    features = pd.read_csv(feature_path)
    required = {
        "date",
        "bar1_open",
        "bar1_high",
        "bar1_low",
        "bar1_close",
        "bar1_volume",
        "avg_volume_14d",
        "relative_volume",
        "atr_14d",
        "exit_time",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"{feature_path} is missing required columns: {missing}")

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    return features


def prepare_intraday(csv_path: Path) -> pd.DataFrame:
    intraday = load_ohlcv(csv_path)
    intraday["date"] = pd.to_datetime(intraday["date"])
    intraday["bar_time"] = intraday["datetime"].dt.strftime("%H:%M")
    return intraday


def parse_period(
    train_start: str | None,
    train_end: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str, str]:
    if (train_start is None) != (train_end is None):
        raise ValueError("Provide both train_start and train_end, or omit both.")

    if train_start is None and train_end is None:
        return None, None, "all", "all"

    start = pd.Timestamp(train_start).normalize()
    end = pd.Timestamp(train_end).normalize()
    if start > end:
        raise ValueError("train_start must be <= train_end")
    return start, end, start.date().isoformat(), end.date().isoformat()


def filter_features_by_period(
    features: pd.DataFrame,
    period_start: pd.Timestamp | None,
    period_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if period_start is None or period_end is None:
        return features
    return features[
        (features["date"] >= period_start) & (features["date"] <= period_end)
    ].copy()


def format_multiplier(value: float) -> str:
    return str(float(value))


def make_trade_candidates(
    ticker: str,
    features: pd.DataFrame,
    intraday: pd.DataFrame,
    atr_multiplier: float,
) -> list[TradeCandidate]:
    candidates: list[TradeCandidate] = []
    intraday_by_date = {
        pd.Timestamp(date_value).normalize(): day_df.sort_values("datetime")
        for date_value, day_df in intraday.groupby("date", sort=False)
    }

    for row in features.sort_values("date").itertuples(index=False):
        if pd.isna(row.atr_14d) or pd.isna(row.relative_volume):
            continue
        if row.relative_volume <= 1.0:
            continue
        if row.bar1_close == row.bar1_open:
            continue

        date_value = pd.Timestamp(row.date).normalize()
        day_df = intraday_by_date.get(date_value)
        if day_df is None:
            continue

        direction = "long" if row.bar1_close > row.bar1_open else "short"
        entry_price = float(row.bar1_high if direction == "long" else row.bar1_low)
        stop_distance_pct = atr_multiplier * (
            float(row.atr_14d) / float(row.bar1_close)
        )
        stop_price = (
            entry_price * (1 - stop_distance_pct)
            if direction == "long"
            else entry_price * (1 + stop_distance_pct)
        )

        candidate = simulate_intraday_trade(
            ticker=ticker,
            date_value=date_value,
            day_df=day_df,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            atr_multiplier=atr_multiplier,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def simulate_intraday_trade(
    ticker: str,
    date_value: pd.Timestamp,
    day_df: pd.DataFrame,
    direction: str,
    entry_price: float,
    stop_price: float,
    atr_multiplier: float,
) -> TradeCandidate | None:
    # Entry simulation is strictly price-based and strictly post-bar1:
    # only bars with start times 09:05-17:15 are allowed to trigger entry.
    # The 09:00 bar defines the ORB only, and the 17:20 bar is reserved for
    # end-of-day exit, so neither can create a new position.
    entry_window = day_df[
        (day_df["bar_time"] >= ENTRY_START) & (day_df["bar_time"] <= ENTRY_END)
    ].copy()
    if entry_window.empty:
        return None

    if direction == "long":
        entry_hits = entry_window[entry_window["high"] >= entry_price]
    else:
        entry_hits = entry_window[entry_window["low"] <= entry_price]

    if entry_hits.empty:
        return None

    entry_bar = entry_hits.iloc[0]
    entry_time = entry_bar["bar_time"]
    if entry_time == "09:00":
        raise AssertionError(f"{ticker} {date_value.date()}: entry_bar_time == 09:00")

    exit_bar = day_df[day_df["bar_time"] == EOD_EXIT_TIME]
    if exit_bar.empty:
        return None

    # Exit checks begin on the entry bar itself. This intentionally catches
    # bars whose high/low range touches entry and the stop in the same
    # 5-minute bar.
    exit_window = day_df[
        (day_df["datetime"] >= entry_bar["datetime"])
        & (day_df["bar_time"] <= EOD_EXIT_TIME)
    ].copy()

    for bar in exit_window.itertuples(index=False):
        if direction == "long":
            stop_hit = bar.low <= stop_price
        else:
            stop_hit = bar.high >= stop_price

        if stop_hit:
            return TradeCandidate(
                date=date_value,
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                exit_price=stop_price,
                exit_type="stop",
                entry_bar_time=entry_time,
                atr_multiplier=atr_multiplier,
            )

    return TradeCandidate(
        date=date_value,
        ticker=ticker,
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        exit_price=float(exit_bar.iloc[0]["close"]),
        exit_type="eod",
        entry_bar_time=entry_time,
        atr_multiplier=atr_multiplier,
    )


def price_trade(candidate: TradeCandidate, capital: float) -> dict[str, object]:
    risk_per_share = abs(candidate.entry_price - candidate.stop_price)
    if risk_per_share <= 0:
        raise ValueError(
            f"{candidate.ticker} {candidate.date.date()}: risk_per_share <= 0"
        )

    risk_sized_shares = (capital * RISK_FRACTION) / risk_per_share
    max_position_shares = (capital * MAX_LEVERAGE) / candidate.entry_price
    shares = min(risk_sized_shares, max_position_shares)

    entry_value = candidate.entry_price * shares
    exit_value = candidate.exit_price * shares
    costs = (entry_value + exit_value) * (COST_BPS_PER_SIDE / 10_000.0)

    if candidate.direction == "long":
        gross_pnl = (candidate.exit_price - candidate.entry_price) * shares
    else:
        gross_pnl = (candidate.entry_price - candidate.exit_price) * shares

    net_pnl = gross_pnl - costs
    capital_after = capital + net_pnl

    return {
        "date": candidate.date.date(),
        "ticker": candidate.ticker,
        "direction": candidate.direction,
        "entry_price": candidate.entry_price,
        "stop_price": candidate.stop_price,
        "exit_price": candidate.exit_price,
        "exit_type": candidate.exit_type,
        "atr_multiplier": candidate.atr_multiplier,
        "shares": shares,
        "gross_pnl": gross_pnl,
        "costs": costs,
        "net_pnl": net_pnl,
        "capital_after": capital_after,
        "_entry_bar_time": candidate.entry_bar_time,
    }


def run_backtest(
    feature_dir: Path,
    intraday_csv_dir: Path,
    atr_multiplier: float,
    train_start: str | None = None,
    train_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], Path, Path]:
    period_start, period_end, period_start_label, period_end_label = parse_period(
        train_start, train_end
    )
    feature_paths = sorted(feature_dir.glob(f"*{FEATURE_SUFFIX}"))
    if not feature_paths:
        raise FileNotFoundError(f"No *{FEATURE_SUFFIX} files found in {feature_dir}")

    intraday_map = build_intraday_file_map(intraday_csv_dir)
    candidates: list[TradeCandidate] = []

    for feature_path in feature_paths:
        ticker = ticker_from_feature_path(feature_path)
        intraday_path = intraday_map.get(ticker)
        if intraday_path is None:
            print(f"Skipping {ticker}: no matching intraday CSV in {intraday_csv_dir}")
            continue

        features = load_features(feature_path)
        features = filter_features_by_period(features, period_start, period_end)
        intraday = prepare_intraday(intraday_path)
        ticker_candidates = make_trade_candidates(
            ticker, features, intraday, atr_multiplier
        )
        candidates.extend(ticker_candidates)
        print(f"{ticker}: {len(ticker_candidates)} trade candidates")

    candidates = sorted(
        candidates,
        key=lambda trade: (trade.date, trade.entry_bar_time, trade.ticker),
    )

    capital = START_CAPITAL
    trade_rows: list[dict[str, object]] = []
    for candidate in candidates:
        row = price_trade(candidate, capital)
        capital = float(row["capital_after"])
        trade_rows.append(row)

    trades = pd.DataFrame.from_records(trade_rows)
    if not trades.empty:
        assert (trades["_entry_bar_time"] != "09:00").all(), "entry_bar_time == 09:00 detected"
        trades = trades.drop(columns=["_entry_bar_time"])

    daily_summary = make_daily_summary(trades, period_start_label, period_end_label)
    stats = calculate_summary_stats(trades, daily_summary)
    trade_path, daily_path = write_outputs(
        trades,
        daily_summary,
        atr_multiplier,
        period_start_label,
        period_end_label,
    )
    return trades, daily_summary, stats, trade_path, daily_path


def make_daily_summary(
    trades: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "period_start",
                "period_end",
                "trades",
                "net_pnl",
                "cumulative_pnl",
                "capital",
            ]
        )

    daily = (
        trades.groupby("date", sort=True)
        .agg(trades=("ticker", "size"), net_pnl=("net_pnl", "sum"), capital=("capital_after", "last"))
        .reset_index()
    )
    daily["period_start"] = period_start
    daily["period_end"] = period_end
    daily["cumulative_pnl"] = daily["net_pnl"].cumsum()
    return daily[
        [
            "date",
            "period_start",
            "period_end",
            "trades",
            "net_pnl",
            "cumulative_pnl",
            "capital",
        ]
    ]


def calculate_summary_stats(
    trades: pd.DataFrame,
    daily_summary: pd.DataFrame,
) -> dict[str, float]:
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "stop_rate": 0.0,
            "eod_rate": 0.0,
            "hit_rate": 0.0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "final_capital": START_CAPITAL,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    daily_net = daily_summary["net_pnl"]
    daily_std = daily_net.std(ddof=1)
    sharpe = 0.0
    if not pd.isna(daily_std) and daily_std > 0:
        sharpe = (daily_net.mean() / daily_std) * math.sqrt(252)

    equity = pd.concat(
        [pd.Series([START_CAPITAL]), daily_summary["capital"].reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = equity - equity.cummax()

    return {
        "total_trades": total_trades,
        "stop_rate": float((trades["exit_type"] == "stop").mean()),
        "eod_rate": float((trades["exit_type"] == "eod").mean()),
        "hit_rate": float((trades["net_pnl"] > 0).mean()),
        "avg_net_pnl": float(trades["net_pnl"].mean()),
        "total_net_pnl": float(trades["net_pnl"].sum()),
        "final_capital": float(daily_summary.iloc[-1]["capital"]),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }


def write_outputs(
    trades: pd.DataFrame,
    daily_summary: pd.DataFrame,
    atr_multiplier: float,
    period_start: str,
    period_end: str,
) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    multiplier_label = format_multiplier(atr_multiplier)
    trade_path = OUTPUT_DIR / f"orb_trades_atr{multiplier_label}_{period_start}_{period_end}.csv"
    daily_path = OUTPUT_DIR / f"orb_daily_summary_atr{multiplier_label}_{period_start}_{period_end}.csv"
    trades.to_csv(trade_path, index=False)
    daily_summary.to_csv(daily_path, index=False)
    return trade_path, daily_path


def main() -> None:
    args = parse_args()
    trades, daily_summary, stats, trade_path, daily_path = run_backtest(
        args.feature_dir,
        args.intraday_csv_dir,
        args.atr_multiplier,
        args.train_start,
        args.train_end,
    )
    print("\nBacktest complete")
    print(f"Totalt antal trades: {stats['total_trades']}")
    print(f"Stop rate: {stats['stop_rate']:.2%}")
    print(f"EOD rate: {stats['eod_rate']:.2%}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Genomsnittlig net_pnl per trade: {stats['avg_net_pnl']:.2f}")
    print(f"Total net_pnl: {stats['total_net_pnl']:.2f}")
    print(f"Slutkapital: {stats['final_capital']:.2f}")
    print(f"Sharpe ratio: {stats['sharpe_ratio']:.4f}")
    print(f"Max drawdown: {stats['max_drawdown']:.2f}")
    print(f"Trade output: {trade_path}")
    print(f"Daily summary: {daily_path}")


if __name__ == "__main__":
    main()
