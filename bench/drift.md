# Drift

Threshold, fixed before the first real run: |ratio − 1| ≤ 1e-4 for at least 99% of (sample, SDE step) entries. Constants `RATIO_TOL` and `MAX_OVER_FRACTION` in `capstan/drift.py`.

| Step | Entries | Mean abs drift | Max abs drift | Ratio p50 | Ratio p99 | Over threshold |
| --- | --- | --- | --- | --- | --- | --- |

## Batch invariance

| Batch size a | Batch size b | Max per-element abs drift |
| --- | --- | --- |
