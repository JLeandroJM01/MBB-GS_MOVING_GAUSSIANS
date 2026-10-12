# scripts/data

Input preparation: turning an MP4 into the PNG sequence the trainer consumes,
and turning a folder of frames back into an MP4.

| Script | What it does |
| --- | --- |
| `extraer_clips_720p.py` | extracts 720p frames from a video into `data/clips/<clip>/` |
| `extraer_frames_originales.py` | extracts ground-truth frames aligned with an experiment, from a given start frame |
| `extraer_video_gt.py` | extracts the original segment as an MP4 matching an experiment in duration and resolution |
| `extraer.py` | generic frame extraction |
| `frames_a_video.py` | turns a `frame_NNNN.png` folder into an MP4 |

See the wiki page **Data Preparation**.
