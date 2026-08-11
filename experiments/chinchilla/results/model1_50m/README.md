# FAIRC/token-averaging-model1_50m

Checkpoint dump from the **token averaging** research project.

- **run name:** `model1_50m`
- **results tree:** `results`

## Contents

### Loss logs

- `loss_log.csv`
- `loss_log_1x_ctx.csv`
- `loss_log_50m_1024ctx.csv`
- `loss_log_rope_bug.csv`
- `loss_log_same_ctx.csv`
- `loss_log_tied.csv`
- `loss_log_untied.csv`

### Checkpoints

- `checkpoints/final.pt`

## Loading a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download('FAIRC/token-averaging-model1_50m', 'checkpoints/final.pt')
state = torch.load(path, map_location='cpu', weights_only=False)
model.load_state_dict(state['model'])  # your OLMAveraged / OLMTransformerBody
print(state['step'], state['tokens_seen'], state['cumulative_flops'])
```

These are **not** Hugging Face `transformers` weights. Rebuild the
architecture from `config.json` → `model_config` (or from
`experiments/chinchilla/model_configs.py` in the source repo) and load
the raw `state_dict`.

## Architecture

```json
{
  "d_model": 512,
  "n_heads": 8,
  "n_layers": 8,
  "context_len": 1024,
  "averaging_k": 1,
  "tie_embeddings": true,
  "lr": 0.0002,
  "warmup_steps": 2000,
  "target_tokens": 1000000000,
  "n_params_approx": 50897408
}
```
