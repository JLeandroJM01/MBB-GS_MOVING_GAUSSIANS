# Data Preparation

Source: `scripts/data/`, `src/gs2d_video/io/video.py`

## What the trainer expects

A clip is a folder of numbered PNGs:

```
data/
└── clips/
    └── mi_clip/
        ├── frame_0000.png
        ├── frame_0001.png
        └── ...
```

The name `mi_clip` is what goes in a configuration's `clip` field. Files are
read in sorted order, so the zero padding matters.

## Extracting a clip

```bash
python scripts/data/extraer_clips_720p.py \
    --video data/videos/mi_video.mp4 \
    --nombre_clip mi_clip \
    --inicio_seg 10 --duracion_seg 20 \
    --H 720 --W 1280
```

Other options: `--fps` forces an output frame rate (default is the video's own),
`--max_frames` caps the count, and `--forzar` re-extracts even if the target
folder already has PNGs.

Extraction can also happen automatically at the start of training, by giving the
configuration a `video_mp4` together with `fps_extraccion`, `n_frames_extraer`
and `resolucion_extraccion`. `src/gs2d_video/io/video.py` handles it through
FFmpeg, with a cache check so an existing valid extraction is not redone.

Note that this module's automatic path applies a centred square crop to the
shorter side before scaling. For 720p widescreen work, extract with
`scripts/data/extraer_clips_720p.py` instead.

## Ground truth aligned with an experiment

To compare a reconstruction against the original, the frames have to line up
exactly — same start frame, same count, same resolution.

```bash
python scripts/data/extraer_frames_originales.py --inicio_frame 150 --n_frames 600
```

For a side-by-side video instead of frames:

```bash
python scripts/data/extraer_video_gt.py --exp outputs/mi_experimento
```

This one reads `info_clip.json` from the experiment folder to get the frame
count and resolution, so alignment is automatic rather than something you have
to get right by hand.

## Frames back to video

```bash
python scripts/data/frames_a_video.py \
    --frames outputs/mi_experimento/frames_renderizados \
    --salida outputs/mi_experimento/video_reconstruido.mp4 \
    --fps 30
```

## Audio

The Gabor configurations point directly at WAV files. The pipeline extracts
audio from the same MP4 segment as the frames; see
[Audiovisual Pipeline](Audiovisual-Pipeline).

## A note on the source material

The videos and audio used in the reported experiments are commercial recordings
and are not distributed with this repository. Each configuration records the
source file, the start point and the duration, so the clips can be regenerated
from your own copy of the material. `data/` is excluded from version control.
