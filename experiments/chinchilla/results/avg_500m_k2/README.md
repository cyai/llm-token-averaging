# cyai/token-averaging-avg_500m_k2

Checkpoint dump from the **token averaging** research project.

- **run name:** `avg_500m_k2`
- **results tree:** `results`

## Contents

### Loss logs

- `loss_log.csv`
- `loss_log_0.1.csv`

### Checkpoints

- _(none — loss logs only)_

## Loading a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download('cyai/token-averaging-avg_500m_k2', 'checkpoints/final.pt')
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
  "d_model": 1280,
  "n_heads": 20,
  "n_layers": 22,
  "context_len": 1024,
  "averaging_k": 2,
  "tie_embeddings": true,
  "lr": 0.00012,
  "warmup_steps": 2000,
  "target_tokens": 20000000000,
  "n_params_approx": 496866560
}
```
