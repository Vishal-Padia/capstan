# Drift

Threshold, fixed before the first real run: |ratio − 1| ≤ 1e-4 for at least 99% of (sample, SDE step) entries. Constants `RATIO_TOL` and `MAX_OVER_FRACTION` in `capstan/drift.py`.

| Step | SDE step index | Per-element mean abs drift | Per-element max abs drift | Importance ratio | Over threshold |
| --- | --- | --- | --- | --- | --- |

## Batch invariance

| Batch size a | Batch size b | Max per-element abs drift |
| --- | --- | --- |
