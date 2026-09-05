# Codex usage breakdown

Print the last seven days of token usage and estimated API-equivalent cost:

```bash
./codex-usage.py
```

Override the window with automatically parsed dates or timestamps:

```bash
./codex-usage.py \
  --since '2026-09-03 10:35' \
  --until '2026-09-05 12:00'
```

It reports input, cached input, cache-write input, output, reasoning, totals,
and a model/service-tier breakdown. Copied fork and checkpoint records are
deduplicated by response ID. Prices come from the adjacent `prices.json`.

Unknown service tiers are assumed and merged into Standard. Treat and merge
them as Fast when desired:

```bash
./codex-usage.py --since '2026-09-03T10:35:00+07:00' --unknown-fast
```

