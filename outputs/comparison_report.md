# Token Averaging — Cross-Method Comparison Report

Generated: 2026-03-04 18:39:42

## Configuration

- Model: `EleutherAI/pythia-410m`
- Dataset: `wikitext` (`wikitext-103-v1`)

## Methods run

- **uniform**: 8 rows  (8 configurations)
- **dynamic**: 6 rows  (6 configurations)
- **overlapping**: 8 rows  (8 configurations)
- **weighted**: 20 rows  (20 configurations)
- **learnable**: 3 rows  (3 configurations)

## Summary table (mean across all layers and configurations)

| method | variance_shrinkage_factor | norm_shrinkage_factor | info_retention_ratio | spectral_total_energy_loss_pct | rank_reduction |
| --- | --- | --- | --- | --- | --- |
| dynamic | 0.5635 | 0.8615 | 1.2119 | 94.4316 | 30.0 |
| learnable | 0.3235 | 0.6477 | 1.0443 | 94.5917 | 132.6667 |
| overlapping | 0.2992 | 0.6275 | 1.0377 | 82.8914 | 67.75 |
| uniform | 0.284 | 0.577 | 1.1305 | 85.4504 | 249.5 |
| weighted | 0.3349 | 0.6425 | 1.0513 | 95.2201 | 85.8 |

## Per-method output directories

- `outputs/uniform/`
- `outputs/dynamic/`
- `outputs/overlapping/`
- `outputs/weighted/`
- `outputs/learnable/`

## Metric descriptions

| Metric | Ideal value | Interpretation |
|--------|-------------|----------------|
| `variance_shrinkage_factor` | close to 1 | Little variance lost |
| `norm_shrinkage_factor`     | close to 1 | Norm preserved |
| `info_retention_ratio`      | close to 1 | Information retained |
| `spectral_total_energy_loss_pct` | close to 0 | Little spectral energy lost |
| `rank_reduction`            | close to 0 | Intrinsic dimensionality preserved |

