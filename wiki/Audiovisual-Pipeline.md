# Audiovisual Pipeline

Source: `scripts/pipeline/run_pipeline_video_audio.py`

The integration: one MP4 segment in, one fully reconstructed audiovisual clip
out, with both modalities represented by explicit primitives — Chebyshev
Gaussians for the image, Gabor atoms for the sound.

```bash
python scripts/pipeline/run_pipeline_video_audio.py \
    --config configs/audiovisual/thriller_10s_1ep/pipeline.json
```

## Stages

1. Extract frames and audio from **the same segment** of the MP4.
2. Train the video model and the audio model.
3. Run adaptive pruning at increasing percentages, accepting on global PSNR.
4. Keep the last approved pruning, or the original model if none passed.
5. Produce UINT16 SAFE and UINT16 ALL packages, render from each, measure PSNR.
6. Compress the packages with 7-Zip when available.
7. Mux the reconstructed video with the reconstructed stereo audio.
8. Write a JSON and text summary.

The point of stages 3 to 5 is that nothing is accepted on faith: every reduction
is rendered and measured against the baseline reconstruction before the pipeline
moves on.

## The master configuration

An experiment needs three files. `pipeline.json` holds the shared settings and
names the other two:

```json
{
  "nombre_pipeline": "thriller_10s_1ep",
  "mp4_original": "data/videos/thriller.mp4",
  "inicio_segundos": 0.0,
  "duracion_segundos": 10.0,
  "fps": 30,
  "resolucion": [720, 1280],
  "device": "cuda",
  "nombre_clip": "thriller_10s_1ep_clip",
  "config_video": "video.json",
  "config_audio": "audio.json",
  "forzar_extraccion": true,
  "pruning": {
    "inicio_pct": 10, "paso_pct": 5,
    "min_pct": 10, "max_pct": 50,
    "psnr_global_min": 65.0
  },
  "video_cuantizacion": {
    "generar_uint16_safe": true,
    "generar_uint16_all": true,
    "usar_para_video_final": "uint16_safe"
  },
  "video_final": { "bitrate_audio_kbps": 192 }
}
```

`inicio_segundos` and `duracion_segundos` are used for **both** the frames and
the audio, which is what keeps the two modalities aligned. Getting this wrong is
the classic way to end up with a reconstruction whose sound drifts against the
picture.

`config_video` and `config_audio` are resolved relative to the master file, so
each experiment folder is self-contained.

## What it calls

The pipeline is an orchestrator: it invokes the same scripts documented
elsewhere, through subprocess.

| Stage | Script |
| --- | --- |
| frame extraction | `scripts/data/extraer_clips_720p.py` |
| video training | `scripts/train.py` |
| audio training | `scripts/audio/train_gabor_stereo.py` |
| pruning | `scripts/compression/run_binary_pruning_adaptativo.py` |
| quantization | `scripts/compression/pack_checkpoint_uint16*.py` and their unpackers |
| reconstruction | `scripts/reconstruction/regenerar_clip_desde_checkpoint_streaming.py` |
| comparison | `scripts/metrics/comparar_frames_psnr.py` |

Because it renders frames from checkpoints in later stages, the video
configuration it drives must keep the rasterised frames rather than discarding
them.

## External requirements

FFmpeg and FFprobe must be on `PATH`: they do the extraction and the final mux.
7-Zip is optional and only used to losslessly compress the final packages; on
Windows the script looks for it at its default install location.

## Reported result

Chapter 8 of the thesis reports reconstruction metrics for both modalities on
the same segment, with a visual comparison of frame 23 and waveform and
spectrogram comparisons for the audio. See [Results](Results).
