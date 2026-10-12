# scripts/visualization

Figures for inspecting what the Gaussians do over time. These produce the
material used in the thesis and the paper.

| Script | What it does |
| --- | --- |
| `viz_gaussian_stats.py` | per-Gaussian statistics: which ones move, change colour or change opacity |
| `viz_trayectorias_marcadores.py` | draws the mu_i(t) trajectories over the frame |
| `viz_trayectorias_marcadores_rango.py` | the same, restricted to a range of frames |
| `viz_elipses_velocidades.py` | draws Gaussian ellipses and their velocities |
| `viz_atributos_gaussiana_tiempo.py` | curves of one Gaussian attributes against time |
| `viz_rank_gaussianas_rango.py` | ranks Gaussians within a temporal range |
| `viz_render_subset.py` | renders using only, or excluding, selected Gaussians |
| `viz_tira_evolucion.py` | filmstrip of how the reconstruction evolves during training |

`viz_gaussian_stats.py` also feeds the adaptive pruning ranking.
