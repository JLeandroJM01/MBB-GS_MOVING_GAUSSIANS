# scripts/reconstruction

Evaluating a trained model from its checkpoint. These train nothing: they load
coefficients and rasterise.

| Script | What it does |
| --- | --- |
| `regenerar_clip_desde_checkpoint_streaming.py` | regenerates every frame without stacking the clip in GPU memory; the route to use at 720p or with many Gaussians |
| `regenerar_fps_interpolado.py` | evaluates the model at intermediate instants to raise the frame rate (temporal interpolation and slow motion) |

See the wiki page **Reconstruction and Interpolation**.
