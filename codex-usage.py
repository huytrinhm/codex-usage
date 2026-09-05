#!/usr/bin/env python3
"""Print Codex token usage and estimated cost."""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

from dateutil.parser import parse as parse_datetime


TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
STATE_DB_FILENAME = "state_5.sqlite"
DB_TIME_MARGIN = timedelta(seconds=5)
SUMMARY_HEADERS = (
    "Resp", "Input", "Cached", "CW", "Output", "Reason", "Total", "Cost",
)


def rollout_time(value: str) -> datetime:
    value = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("rollout timestamp has no timezone")
    return parsed


def user_time(value: str, local_timezone: Any) -> datetime:
    parsed = parse_datetime(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=local_timezone)


def empty_usage() -> dict[str, int]:
    return dict.fromkeys(TOKEN_KEYS, 0)


def token_usage(value: Any) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    return {key: max(int(value.get(key, 0) or 0), 0) for key in TOKEN_KEYS}


def add_usage(total: dict[str, int], value: dict[str, int]) -> None:
    for key in TOKEN_KEYS:
        total[key] += value[key]


def json_items(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as lines:
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def progress(done: int, total: int) -> None:
    if done == total:
        print("\r\033[K", end="", file=sys.stderr, flush=True)
        return
    width = 28
    filled = width * done // total
    percent = 100 * done // total
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rScanning [{bar}] {percent:3d}% ({done:,}/{total:,})",
        end="",
        file=sys.stderr,
        flush=True,
    )


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def candidate_paths(root: Path, since: datetime) -> list[Path]:
    sqlite_home = Path(os.environ.get("CODEX_SQLITE_HOME", root.parent)).expanduser()
    db_path = sqlite_home / STATE_DB_FILENAME
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        with connection:
            rows = connection.execute(
                """
                SELECT rollout_path
                FROM threads
                WHERE updated_at_ms >= ?
                """,
                (int((since - DB_TIME_MARGIN).timestamp() * 1000),),
            )
            return sorted(Path(path) for path, in rows if Path(path).is_file())
    except sqlite3.Error:
        return sorted(
            path
            for path in root.rglob("*.jsonl")
            if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) >= since
        )
    finally:
        if "connection" in locals():
            connection.close()


@dataclass(frozen=True)
class Record:
    time: datetime
    model: str
    tier: str
    workspace: str
    usage: dict[str, int]


def scan(since: datetime, until: datetime) -> list[Record]:
    root = codex_home() / "sessions"
    if not root.is_dir():
        raise FileNotFoundError(f"sessions directory not found: {root}")
    responses: dict[str, Record] = {}
    paths = candidate_paths(root, since)
    show_progress = sys.stderr.isatty()
    last_percent = -1

    for index, path in enumerate(paths, 1):
        percent = 100 * (index - 1) // len(paths) if paths else 100
        if show_progress and percent != last_percent:
            progress(index - 1, len(paths))
            last_percent = percent
        contexts: dict[str, tuple[str, str]] = {}
        initial_tier = "unknown"
        initial_workspace = "unknown"
        for item in json_items(path):
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if item.get("type") == "session_meta":
                initial_tier = str(payload.get("service_tier") or initial_tier)
                initial_workspace = str(payload.get("cwd") or initial_workspace)
            elif item.get("type") == "turn_context" and payload.get("turn_id"):
                contexts[str(payload["turn_id"])] = (
                    str(payload.get("model") or "unknown"),
                    str(payload.get("cwd") or initial_workspace),
                )

        tier = initial_tier
        for item in json_items(path):
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if item.get("type") == "event_msg":
                settings = payload.get("thread_settings")
                if payload.get("type") == "thread_settings_applied" and isinstance(
                    settings, dict
                ):
                    tier = str(settings.get("service_tier") or "default")
                continue
            if item.get("type") != "token_usage_record" or not payload.get("response_id"):
                continue
            try:
                timestamp = rollout_time(str(item["timestamp"]))
            except (KeyError, ValueError):
                continue
            if not since <= timestamp <= until:
                continue

            model, workspace = contexts.get(
                str(payload.get("turn_id") or ""), ("unknown", initial_workspace)
            )
            responses.setdefault(
                str(payload["response_id"]),
                Record(timestamp, model, tier, workspace, token_usage(payload.get("usage"))),
            )
    if show_progress:
        progress(len(paths), len(paths))
    return list(responses.values())


@dataclass(frozen=True)
class Prices:
    models: dict[str, dict[str, Decimal]]
    tiers: dict[str, Decimal]


def load_prices() -> Prices:
    raw = json.loads(Path(__file__).with_name("prices.json").read_text())
    decimal_map = lambda values: {
        key: Decimal(str(value)) for key, value in values.items()
    }
    return Prices(
        {model: decimal_map(rates) for model, rates in raw["models"].items()},
        decimal_map(raw["tier_multipliers"]),
    )


def effective_tier(tier: str, unknown_fast: bool) -> str:
    if tier == "unknown":
        return "fast" if unknown_fast else "standard"
    if tier in {"default", "standard"}:
        return "standard"
    if tier in {"fast", "priority"}:
        return "fast"
    return tier


def estimate(record: Record, tier: str, prices: Prices) -> Decimal | None:
    rates, multiplier = prices.models.get(record.model), prices.tiers.get(tier)
    if rates is None or multiplier is None:
        return None
    usage = record.usage
    cached = usage["cached_input_tokens"]
    cache_write = usage["cache_write_input_tokens"]
    uncached = max(usage["input_tokens"] - cached - cache_write, 0)
    return multiplier * (
        Decimal(uncached) * rates["input"]
        + Decimal(cached) * rates["cached_input"]
        + Decimal(cache_write) * rates["cache_write_input"]
        + Decimal(usage["output_tokens"]) * rates["output"]
    ) / Decimal(1_000_000)


@dataclass
class Summary:
    responses: int = 0
    usage: dict[str, int] = field(default_factory=empty_usage)
    cost: Decimal | None = Decimal(0)


Billed = tuple[Record, str, Decimal | None]


def summarize(billed: list[Billed], key) -> dict[Any, Summary]:
    groups: dict[Any, Summary] = defaultdict(Summary)
    for record, tier, price in billed:
        group = groups[key(record, tier)]
        group.responses += 1
        add_usage(group.usage, record.usage)
        group.cost = group.cost + price if group.cost is not None and price is not None else None
    return groups


def money(value: Decimal | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def cells(group: Summary) -> tuple[str, ...]:
    usage = group.usage
    return tuple(
        f"{value:,}"
        for value in (
            group.responses,
            usage["input_tokens"],
            usage["cached_input_tokens"],
            usage["cache_write_input_tokens"],
            usage["output_tokens"],
            usage["reasoning_output_tokens"],
            usage["total_tokens"],
        )
    ) + (money(group.cost),)


def table(headers: tuple[str, ...], rows: list[tuple[str, ...]], left=2) -> None:
    widths = [max([len(header), *(len(row[i]) for row in rows)]) for i, header in enumerate(headers)]
    render = lambda row: "  ".join(
        value.ljust(width) if i < left else value.rjust(width)
        for i, (value, width) in enumerate(zip(row, widths))
    )
    print(render(headers))
    print(render(tuple("-" * width for width in widths)))
    for row in rows:
        print(render(row))


def workspace_name(value: str) -> str:
    home = str(Path.home())
    return "~" if value == home else "~" + value[len(home) :] if value.startswith(home + "/") else value


def report(
    records: list[Record], prices: Prices, since: datetime, until: datetime, unknown_fast: bool
) -> None:
    billed = [
        (record, tier, estimate(record, tier, prices))
        for record in records
        for tier in (effective_tier(record.tier, unknown_fast),)
    ]
    total = next(iter(summarize(billed, lambda *_: None).values()), Summary())
    usage = total.usage
    uncached = max(
        usage["input_tokens"] - usage["cached_input_tokens"] - usage["cache_write_input_tokens"],
        0,
    )
    print(
        f"{since.isoformat(timespec='seconds')} → "
        f"{until.astimezone(since.tzinfo).isoformat(timespec='seconds')}\n"
        f"{total.responses:,} responses | {usage['total_tokens']:,} tokens | "
        f"~{money(total.cost)}\n"
        f"input {usage['input_tokens']:,} | uncached {uncached:,} | "
        f"cached {usage['cached_input_tokens']:,} | cache-write {usage['cache_write_input_tokens']:,} | "
        f"output {usage['output_tokens']:,} | reasoning {usage['reasoning_output_tokens']:,}"
    )
    if not records:
        return

    groups = summarize(billed, lambda record, tier: (record.model, tier))
    print()
    table(
        ("Model", "Tier", *SUMMARY_HEADERS),
        [
            (model, tier, *cells(group))
            for (model, tier), group in sorted(groups.items())
        ],
    )

    groups = summarize(
        billed, lambda record, _tier: record.time.astimezone(since.tzinfo).date().isoformat()
    )
    print()
    table(
        ("Date", *SUMMARY_HEADERS),
        [(day, *cells(group)) for day, group in sorted(groups.items())],
        left=1,
    )

    groups = summarize(billed, lambda record, _tier: record.workspace)
    print()
    table(
        ("Workspace", *SUMMARY_HEADERS),
        [(workspace_name(name), *cells(group)) for name, group in sorted(groups.items())],
        left=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--unknown-fast", action="store_true")
    args = parser.parse_args()
    now = datetime.now().astimezone()
    try:
        until = user_time(args.until, now.tzinfo) if args.until else now
        since = user_time(args.since, now.tzinfo) if args.since else until - timedelta(days=7)
        if since > until:
            raise ValueError("--since must be no later than --until")
        report(scan(since, until), load_prices(), since, until, args.unknown_fast)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
