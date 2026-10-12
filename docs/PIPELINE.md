# MBB-GS - Pipeline audiovisual

Este documento describe el flujo audiovisual completo de MBB-GS.

El pipeline permite partir de un unico archivo MP4 y producir una reconstruccion final de video y audio.

## Flujo general

```text
MP4 original
    |
    +--> extraccion de frames
    |        |
    |        +--> entrenamiento MBB-GS
    |        |        |
    |        |        +--> checkpoint FP32
    |        |
    |        +--> pruning adaptativo
    |                 |
    |                 +--> checkpoint podado
    |                          |
    |                          +--> cuantizacion UINT16
    |                                   |
    |                                   +--> reconstruccion de video
    |
    +--> extraccion de audio
             |
             +--> entrenamiento Gabor
                      |
                      +--> audio reconstruido

video reconstruido + audio reconstruido
                 |
                 +--> MP4 final
```

---

# 1. Entrada

El pipeline recibe una configuracion maestra `pipeline.json`.

Esta configuracion define principalmente:

```text
archivo MP4
inicio del segmento
duracion
FPS
resolucion
config de video
config de audio
pruning
cuantizacion
```

Ejemplo de ejecucion:

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```

---

# 2. Extraccion del segmento

A partir del MP4 original se generan dos representaciones sincronizadas:

```text
frames de video
audio WAV estereo
```

El mismo `inicio_segundos` y `duracion_segundos` se utilizan para ambos.

La cantidad de frames depende de:

```text
duracion_segundos * fps
```

Ejemplo:

```text
2 segundos * 10 FPS = 20 frames
```

Los frames se almacenan normalmente en:

```text
data/clips/<nombre_clip>/
```

El audio temporal utilizado por el pipeline se guarda en:

```text
data/audio_pipeline/
```

---

# 3. Entrenamiento de video

Los frames son representados mediante una poblacion fija de Gaussianas 2D.

Los atributos temporales principales son:

```text
posicion
opacidad
color
escala
rotacion
profundidad
```

Su evolucion temporal se representa principalmente mediante coeficientes de Chebyshev.

El rasterizado de entrenamiento utiliza el rasterizador CUDA tiled cuando:

```text
usar_cuda_tiled = true
```

El checkpoint final se guarda normalmente en:

```text
outputs/<experimento_video>/checkpoints/checkpoint_final.pt
```

---

# 4. Entrenamiento de audio

El audio se representa mediante atomos Gabor.

En audio estereo puede utilizarse representacion Mid-Side:

```text
M = (L + R) / 2
S = (L - R) / 2
```

Mid y Side pueden utilizar cantidades distintas de atomos.

Con `forzar_pytorch=false`, el sistema intenta utilizar la extension CUDA Gabor.

Los resultados se almacenan normalmente en:

```text
outputs/gabor/<experimento_audio>/
```

La reconstruccion estereo utilizada por el pipeline es:

```text
recon_stereo.wav
```

---

# 5. Pruning adaptativo

Despues del entrenamiento de video se analiza la poblacion de gaussianas.

El proceso genera estadisticas relacionadas con:

```text
movimiento
variacion de color
variacion de opacidad
variacion de escala
```

Posteriormente se prueban porcentajes de eliminacion.

Cada candidato se compara contra el render baseline del modelo completo.

El objetivo es conservar el mayor nivel de pruning que cumpla el criterio de calidad configurado.

Para un smoke test puede utilizarse un unico porcentaje.

Ejemplo:

```text
inicio_pct = 5
min_pct = 5
max_pct = 5
```

---

# 6. Cuantizacion

El checkpoint podado puede comprimirse mediante cuantizacion UINT16.

El pipeline soporta principalmente:

```text
UINT16 SAFE
UINT16 ALL
```

UINT16 SAFE mantiene una estrategia mas conservadora y cuantiza tensores temporales seleccionados.

UINT16 ALL aplica cuantizacion a un conjunto mas amplio de tensores flotantes.

La representacion seleccionada se desempaqueta temporalmente para verificar que sigue siendo renderizable.

---

# 7. Reconstruccion

El checkpoint seleccionado se evalua frame por frame.

Los frames reconstruidos se guardan y posteriormente se convierten a MP4.

El audio Gabor reconstruido se conserva como WAV.

---

# 8. Metricas

El pipeline puede registrar metricas de video y audio.

Entre las metricas utilizadas por diferentes etapas se encuentran:

```text
PSNR
SSIM
LPIPS
SNR
SI-SDR
LSD
MR-STFT
```

No todas las metricas deben activarse en un smoke test.

Las metricas deben interpretarse segun la etapa que se esta comparando.

Por ejemplo, el PSNR utilizado durante pruning compara el modelo podado contra el render baseline, no necesariamente contra los frames originales.

---

# 9. MP4 final

Finalmente FFmpeg combina:

```text
video reconstruido
+
audio reconstruido
```

El resultado se guarda normalmente en:

```text
outputs/AV_PIPELINE/<nombre_pipeline>/final/
```

Ejemplo:

```text
smoke_20frames_reconstruido_con_audio.mp4
```

---

# 10. Salidas principales

Una ejecucion completa genera informacion en varias carpetas.

```text
outputs/
|
+-- <experimento_video>/
|   +-- checkpoints/
|   +-- frames_renderizados/
|   +-- binary_pruning/
|   +-- viz_stats/
|
+-- gabor/
|   +-- <experimento_audio>/
|
+-- AV_PIPELINE/
    +-- <nombre_pipeline>/
        +-- runtime_configs/
        +-- video_quant/
        +-- final/
        +-- resumen_pipeline.txt
```

---

# 11. Runtime configs

El pipeline genera configuraciones efectivas dentro de:

```text
outputs/AV_PIPELINE/<nombre_pipeline>/runtime_configs/
```

Estas configuraciones contienen los valores finalmente utilizados por los entrenamientos de video y audio.

Son importantes para reproducibilidad y para revisar exactamente que parametros utilizo una ejecucion.

---

# 12. Smoke test oficial

El smoke test oficial se encuentra en:

```text
configs/examples/smoke_20frames/
```

Actualmente verifica una ejecucion pequena con:

```text
20 frames
10 FPS
144x256
4000 gaussianas
1 epoch de video
512 atomos Mid
256 atomos Side
1 epoch de audio
5% de pruning
UINT16 SAFE
```

Su objetivo no es obtener calidad visual o auditiva.

Su objetivo es verificar:

```text
extraccion
CUDA video
CUDA audio
entrenamiento
pruning
cuantizacion
reconstruccion
metricas
FFmpeg
```

---

# 13. Antes de un experimento grande

Se recomienda comprobar:

```text
python -m pytest -q
```

y ejecutar primero el smoke audiovisual.

Despues se puede aumentar progresivamente:

```text
duracion
FPS
resolucion
gaussianas
grados temporales
epochs
atomos Gabor
```

---

# 14. Documentacion relacionada

```text
docs/COMMANDS.md
docs/CONFIGURATION.md
docs/COMPRESSION.md
docs/VISUALIZATIONS.md
docs/INSTALLATION.md
docs/TROUBLESHOOTING.md
```
