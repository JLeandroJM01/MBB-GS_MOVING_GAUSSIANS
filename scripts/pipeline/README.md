# scripts/pipeline

Orchestrators that chain several stages. They call the scripts in the other
folders through subprocess.

| Script | What it does |
| --- | --- |
| `run_pipeline_video_audio.py` | full audiovisual pipeline: extraction, video and audio training, pruning, UINT16 and metrics |
| `run_tests_secuencial.py` | runs the loss ablation phases in order |

See the wiki page **Audiovisual Pipeline**.
