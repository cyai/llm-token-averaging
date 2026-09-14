# FAIRC/token-averaging-seed_replicates-avg_50m_k2_seed1

Checkpoint dump from the **token averaging** research project.

- **run name:** `avg_50m_k2_seed1`
- **results tree:** `seed_replicates`

## Contents

### Loss logs

- `loss_log.csv`

### Checkpoints

- `checkpoints/final.pt`
- `checkpoints/step_00050000.pt`
- `checkpoints/step_00100000.pt`

## Loading a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download('FAIRC/token-averaging-seed_replicates-avg_50m_k2_seed1', 'checkpoints/final.pt')
state = torch.load(path, map_location='cpu', weights_only=False)
model.load_state_dict(state['model'])  # your OLMAveraged / OLMTransformerBody
print(state['step'], state['tokens_seen'], state['cumulative_flops'])
```

These are **not** Hugging Face `transformers` weights. Rebuild the
architecture from `config.json` → `model_config` (or from
`experiments/chinchilla/model_configs.py` in the source repo) and load
the raw `state_dict`.
