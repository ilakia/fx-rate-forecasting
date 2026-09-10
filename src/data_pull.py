"""
Pull daily historical FX reference rates from the Frankfurter API
(https://frankfurter.dev), a free, keyless wrapper around the European
Central Bank's daily reference rates.

Reusable across currency pairs: this script pulled EUR/USD for the initial
build, and the same logic (just different --base/--quote flags) will be
used for GBP/USD and USD/JPY in a later phase.

Usage:
    python src/data_pull.py --base EUR --quote USD \
        --start 1999-01-04 --end 2026-09-10

Output:
    data/raw/{base}{quote}_raw.json        - untouched API response
    data/processed/{base}{quote}_daily.csv - cleaned daily series
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://api.frankfurter.dev/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent


def fetch_range(base: str, quote: str, start: str, end: str) -> dict:
    """Fetch the full date range in a single request.

    Frankfurter has no published rate limit and returned ~7,100 daily
    observations for EUR/USD (1999-01-04 to today) in one call with no
    pagination needed, so no chunking/retry logic is implemented here.
    If a future pair or wider range ever fails, chunk by year and retry.
    """
    url = f"{API_BASE}/{start}..{end}"
    resp = requests.get(url, params={"from": base, "to": quote}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if "rates" not in payload or not payload["rates"]:
        raise ValueError(f"No rates returned for {base}/{quote} {start}..{end}")
    return payload


def save_raw(payload: dict, base: str, quote: str) -> Path:
    raw_dir = REPO_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{base.lower()}{quote.lower()}_raw.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def clean_and_save(payload: dict, base: str, quote: str) -> Path:
    """Convert the raw {date: {quote: rate}} map into a tidy, deduplicated,
    date-sorted CSV. No gap-filling: FX markets don't trade on weekends or
    ECB/target2 holidays, so missing calendar dates are expected and are
    left absent rather than imputed. See DECISIONS_AND_ISSUES_LOG.md.
    """
    rates = payload["rates"]
    records = [(date, values[quote]) for date, values in rates.items()]
    df = pd.DataFrame(records, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{base.lower()}{quote.lower()}_daily.csv"
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    return out_path, df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="EUR", help="Base currency, e.g. EUR")
    parser.add_argument("--quote", default="USD", help="Quote currency, e.g. USD")
    parser.add_argument("--start", default="1999-01-04", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Fetching {args.base}/{args.quote} from {args.start} to {end} ...")
    payload = fetch_range(args.base, args.quote, args.start, end)

    raw_path = save_raw(payload, args.base, args.quote)
    print(f"Raw response saved to {raw_path}")

    processed_path, df = clean_and_save(payload, args.base, args.quote)
    print(f"Processed series saved to {processed_path}")
    print(f"Rows: {len(df)}  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")


if __name__ == "__main__":
    sys.exit(main())
