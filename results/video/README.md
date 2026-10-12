# Video experiment records

Lightweight records for 14 video experiments: the aggregated and
per-frame metrics, the exact configuration each run used, and the clip
metadata. They are here so the numbers reported in the paper can be
checked without retraining anything.

| File | Contents |
| --- | --- |
| `metricas.json` | aggregates and per-frame values, split pre/post pruning |
| `metricas_por_frame.csv` | one row per frame |
| `config_usada.json` | the exact configuration used |
| `info_clip.json` | clip metadata: frames, resolution, seed |
| `log_entrenamiento.csv` | per-epoch losses and timings |

Rendered frames, videos and checkpoints are published separately; see
the link in the main README.
