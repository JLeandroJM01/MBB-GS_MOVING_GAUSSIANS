# scripts/compression

Reducing model size after training: pruning Gaussians and quantizing
coefficients to UINT16.

| Script | What it does |
| --- | --- |
| `run_binary_pruning_adaptativo.py` | searches by percentage for how many Gaussians can be pruned without losing PSNR |
| `viz_prune_checkpoint_by_stats.py` | builds a pruned checkpoint from IDs or statistics, without retraining |
| `pack_checkpoint_uint16.py` / `unpack_checkpoint_uint16.py` | UINT16 SAFE: quantizes the bulky high-order coefficients, keeps the rest in float32 |
| `pack_checkpoint_uint16_all.py` / `unpack_checkpoint_uint16_all.py` | UINT16 ALL: quantizes everything |

See the wiki page **Pruning and Quantization**.
