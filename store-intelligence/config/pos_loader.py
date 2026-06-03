"""
Store Intelligence System — POS Transaction Loader.

Loads point-of-sale transaction data from CSV and provides
time-windowed lookup functions for correlating billing-zone
exits with actual purchases.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# ── Module-level cache ──────────────────────────────────────
_transactions_cache: Optional[List[Dict]] = None
_cache_path: Optional[str] = None


def _resolve_csv_path() -> Path:
    """Resolve the absolute path to pos_transactions.csv."""
    from config.settings import get_settings

    settings = get_settings()
    csv_path = Path(settings.POS_DATA_PATH)
    if not csv_path.is_absolute():
        # Resolve relative to the project root (parent of config/)
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / csv_path
    return csv_path


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime.

    Handles both 'Z' suffix and offset formats.  Naive timestamps are
    assumed to be UTC.
    """
    ts_str = ts_str.strip()
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_transactions(csv_path: Optional[Path] = None) -> List[Dict]:
    """Load and cache all POS transactions from the CSV file.

    Each row is returned as a dict with keys:
        store_id, transaction_id, timestamp (datetime), basket_value_inr (float)

    Args:
        csv_path: Override path; if *None*, resolved from settings.

    Returns:
        List of transaction dicts sorted by timestamp ascending.
    """
    global _transactions_cache, _cache_path

    resolved = str(csv_path) if csv_path else str(_resolve_csv_path())

    if _transactions_cache is not None and _cache_path == resolved:
        return _transactions_cache

    if not os.path.exists(resolved):
        logger.warning("pos_csv_not_found", path=resolved)
        _transactions_cache = []
        _cache_path = resolved
        return _transactions_cache

    transactions: List[Dict] = []
    with open(resolved, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                transactions.append(
                    {
                        "store_id": row["store_id"].strip(),
                        "transaction_id": row["transaction_id"].strip(),
                        "timestamp": _parse_timestamp(row["timestamp"]),
                        "basket_value_inr": float(row["basket_value_inr"]),
                    }
                )
            except (KeyError, ValueError) as exc:
                logger.warning("pos_row_parse_error", row=row, error=str(exc))

    transactions.sort(key=lambda t: t["timestamp"])
    _transactions_cache = transactions
    _cache_path = resolved
    logger.info("pos_transactions_loaded", count=len(transactions), path=resolved)
    return _transactions_cache


def reload_transactions(csv_path: Optional[Path] = None) -> List[Dict]:
    """Force-reload the POS transaction cache.

    Useful after the CSV file has been updated on disk.

    Args:
        csv_path: Override path; if *None*, resolved from settings.

    Returns:
        Freshly loaded list of transaction dicts.
    """
    global _transactions_cache, _cache_path
    _transactions_cache = None
    _cache_path = None
    return _load_transactions(csv_path)


def get_transactions(
    store_id: str,
    start_time: datetime,
    end_time: datetime,
) -> List[Dict]:
    """Return all transactions for *store_id* within [start_time, end_time].

    Args:
        store_id:   Store identifier, e.g. ``"STORE_BLR_002"``.
        start_time: Inclusive lower bound (timezone-aware or naive-UTC).
        end_time:   Inclusive upper bound (timezone-aware or naive-UTC).

    Returns:
        List of matching transaction dicts, sorted by timestamp.
    """
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    all_txns = _load_transactions()
    return [
        txn
        for txn in all_txns
        if txn["store_id"] == store_id
        and start_time <= txn["timestamp"] <= end_time
    ]


def get_transactions_in_window(
    store_id: str,
    timestamp: datetime,
    window_seconds: int = 300,
) -> List[Dict]:
    """Find transactions within ±*window_seconds* of *timestamp*.

    This is designed for correlating a billing-zone exit with a POS
    transaction.  A 5-minute (300 s) default window accounts for
    minor clock-skew between camera and POS system.

    Args:
        store_id:       Store identifier.
        timestamp:      Center of the lookup window.
        window_seconds: Half-window size in seconds (default 300 = 5 min).

    Returns:
        List of matching transaction dicts, sorted by timestamp.
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    delta = timedelta(seconds=window_seconds)
    return get_transactions(
        store_id=store_id,
        start_time=timestamp - delta,
        end_time=timestamp + delta,
    )
