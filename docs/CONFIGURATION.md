# MBB-GS - Guia de configuracion

Este documento describe las configuraciones principales utilizadas por MBB-GS.

El proyecto separa la configuracion en tres archivos:

```text
pipeline.json   -> orquesta video, audio, pruning, cuantizacion y salida final
video.json      -> entrenamiento temporal MBB-GS de video
audio.json      -> entrenamiento Gabor de audio
```

Las plantillas base se encuentran en:

```text
configs/templates/pipeline.template.json
configs/templates/video.template.json
configs/templates/audio.template.json
```

El ejemplo minimo validado se encuentra en:

```text
configs/examples/smoke_20frames/
```

> El smoke test esta disenado para verificar funcionamiento. Sus valores no deben interpretarse como una configuracion recomendada para calidad final.

---

# 1. Pipeline audiovisual

El archivo `pipeline.json` controla el flujo completo a partir de un MP4.

Flujo general:

```text
MP4
 -> extraccion de frames
 -> extraccion de audio
 -> entrenamiento MBB-GS
 -> entrenamiento Gabor
 -> pruning
 -> cuantizacion
 -> reconstruccion
 -> metricas
 -> MP4 final
```

## nombre_pipeline

Nombre general de la ejecucion audiovisual.

Tambien se utiliza para organizar las salidas en:

```text
outputs/AV_PIPELINE/<nombre_pipeline>/
```

Ejemplo:

```text
smoke_20frames
```

## mp4_original

Ruta del archivo audiovisual de entrada.

Ejemplo:

```text
data/videos/smoke_input.mp4
```

## inicio_segundos

Segundo del video original donde comienza el segmento.

Ejemplo:

```text
0.0
```

## duracion_segundos

Duracion del segmento que se procesa.

Ejemplo:

```text
2.0
```

## fps

FPS utilizados para extraer y reconstruir el video.

La cantidad aproximada de frames es:

```text
frames = duracion_segundos * fps
```

Ejemplo:

```text
2 segundos * 10 FPS = 20 frames
```

## resolucion

Resolucion de procesamiento en formato:

```text
[alto, ancho]
```

Ejemplo smoke:

```text
[144, 256]
```

Ejemplo 720p:

```text
[720, 1280]
```

## device

Dispositivo utilizado.

Valor recomendado para entrenamiento:

```text
cuda
```

## nombre_clip

Nombre de la carpeta donde se almacenan los frames extraidos.

Los frames quedan normalmente en:

```text
data/clips/<nombre_clip>/
```

## config_video

Archivo de configuracion del modelo de video.

Normalmente:

```text
video.json
```

## config_audio

Archivo de configuracion del modelo de audio.

Normalmente:

```text
audio.json
```

## forzar_extraccion

Si es `true`, vuelve a generar frames y audio aunque existan archivos previos.

Para pruebas reproducibles es recomendable utilizar:

```text
true
```

---

## Bloque pruning

Ejemplo:

```text
inicio_pct
paso_pct
min_pct
max_pct
psnr_global_min
```

### inicio_pct

Primer porcentaje de gaussianas que intenta eliminar.

### paso_pct

Incremento entre intentos sucesivos.

Ejemplo:

```text
10, 15, 20, 25, ...
```

cuando `inicio_pct=10` y `paso_pct=5`.

### min_pct y max_pct

Limites de la busqueda.

Para ejecutar una sola prueba de pruning se puede utilizar:

```text
inicio_pct = 5
paso_pct = 5
min_pct = 5
max_pct = 5
```

### psnr_global_min

Umbral minimo de PSNR respecto al render baseline para aceptar un porcentaje de pruning.

Un valor bajo como `0.0` sirve solamente para smoke tests.

En experimentos reales debe utilizarse un umbral coherente con el criterio de calidad del experimento.

---

## Bloque video_cuantizacion

### generar_uint16_safe

Genera la cuantizacion UINT16 SAFE.

Cuantiza principalmente coeficientes temporales seleccionados y conserva otros parametros con mayor precision.

### generar_uint16_all

Genera UINT16 ALL.

Este modo intenta cuantizar todos los tensores flotantes compatibles.

### usar_para_video_final

Selecciona que representacion se utilizara para reconstruir el video final.

Ejemplo:

```text
uint16_safe
```

---

## Bloque video_final

### bitrate_audio_kbps

Bitrate AAC utilizado cuando FFmpeg combina el video reconstruido y el audio reconstruido.

Ejemplos:

```text
96
192
```

---

# 2. Configuracion de video MBB-GS

El archivo `video.json` controla el modelo temporal de Gaussian Splatting 2D.

## nombre_experimento

Nombre de la ejecucion de video.

Los resultados se guardan normalmente en:

```text
outputs/<nombre_experimento>/
```

## clip

Nombre del conjunto de frames utilizado para entrenamiento.

Cuando se utiliza el pipeline este valor es preparado automaticamente.

## video_mp4

Permite asociar directamente un MP4 en ciertos flujos de entrenamiento.

Cuando el pipeline ya realizo la extraccion normalmente permanece en `null`.

## max_frames

Cantidad maxima de frames utilizados.

En el smoke:

```text
20
```

En experimentos reales depende de la duracion y FPS del segmento.

## device

Para utilizar los kernels CUDA:

```text
cuda
```

## seed

Semilla utilizada para reproducibilidad.

Valor comun:

```text
42
```

---

## base_temporal

Base matematica utilizada para representar la evolucion temporal de los parametros.

La base principal del proyecto es:

```text
chebyshev
```

La base monomial se conserva para compatibilidad y experimentos de ablacion, pero no es la configuracion principal.

---

## n_gaussianas_inicial

Numero de gaussianas utilizadas por el modelo.

Smoke:

```text
4000
```

Los experimentos reales pueden utilizar decenas o cientos de miles de gaussianas.

Incrementar este valor aumenta capacidad, memoria y costo computacional.

## inicializar_color_desde_frame0

Inicializa el color utilizando informacion del primer frame.

Normalmente:

```text
true
```

## escala_inicial_px

Escala espacial inicial de las gaussianas expresada aproximadamente en pixeles.

---

# 3. Grados temporales

El bloque `grados` controla cuantos coeficientes temporales utiliza cada atributo.

Atributos:

```text
mu       -> posicion
opacity  -> opacidad
color    -> RGB
scale    -> escala
theta    -> rotacion
depth    -> profundidad
```

El smoke utiliza grados pequenos para reducir el costo.

Una configuracion historicamente utilizada en experimentos de mayor calidad es:

```text
mu       = 100
opacity  = 80
color    = 30
scale    = 12
theta    = 6
depth    = 4
```

Estos valores no son obligatorios. Deben elegirse segun duracion, movimiento, complejidad y presupuesto computacional.

Un grado mayor permite mas variacion temporal pero tambien incrementa parametros y costo.

---

# 4. Learning rates

El bloque `lrs` define tasas de aprendizaje independientes.

Cada atributo puede separar:

```text
*_a0    -> termino base
*_high  -> coeficientes temporales de orden superior
```

Ejemplos:

```text
mu_a0
mu_high
opacity_a0
opacity_high
color_a0
color_high
scale_a0
scale_high
theta_a0
theta_high
depth_a0
depth_high
```

Cambiar estos valores modifica directamente la dinamica de optimizacion.

No deben modificarse solamente para acelerar un smoke test.

---

# 5. Entrenamiento de video

## n_epochs

Numero de epochs.

Smoke:

```text
1
```

Una ejecucion de 1 epoch solamente verifica que el pipeline funciona.

## checkpoint_cada_n_epochs

Frecuencia de checkpoints intermedios.

## sub_batch_frames

Cantidad de frames procesados en grupos durante entrenamiento.

Puede ayudar a controlar memoria GPU.

---

# 6. Loss de video

## tipo_loss

Selecciona la estrategia principal de loss.

Una configuracion utilizada en el flujo actual es:

```text
motion
```

## lambda_dssim

Peso de DSSIM.

## lambda_mse

Peso del error cuadratico.

## lambda_motion

Peso de la componente asociada al movimiento.

## lambda_hard

Peso de la variante hard cuando corresponde.

## lambda_edge

Peso de componentes de borde si se utilizan.

## lambda_temporal

Peso de regularizacion temporal adicional cuando corresponde.

## motion_umbral

Umbral utilizado para determinar regiones con movimiento.

## motion_blur

Suavizado utilizado en la mascara de movimiento.

## motion_clip y hard_clip

Limites aplicados a los pesos de las respectivas estrategias.

## exponente_pixel

Controla el reponderado a nivel de pixel.

## exponente_frame

Controla el reponderado entre frames.

En los experimentos realizados, `exponente_frame=1.0` ha sido una configuracion importante y debe conservarse como referencia.

## usar_loss_cuda

Controla si determinadas operaciones de loss utilizan implementacion CUDA cuando esta disponible.

Esto es independiente del rasterizador tiled.

Por ejemplo, es posible tener:

```text
usar_loss_cuda = false
usar_cuda_tiled = true
```

y seguir utilizando CUDA para el render diferenciable.

---

# 7. Smoothness

## beta_smoothness

Peso global de regularizacion de suavidad.

## pesos_smoothness

Permite asignar diferente regularizacion a:

```text
mu
opacity
color
scale
theta
depth
```

---

# 8. CUDA de video

## usar_cuda_tiled

Activa el rasterizador tiled CUDA diferenciable.

Configuracion recomendada con GPU compatible:

```text
true
```

## cuda_tile_size

Tamano del tile utilizado por el rasterizador.

Valor utilizado habitualmente:

```text
16
```

## cuda_k_sigma

Radio efectivo utilizado para limitar la influencia espacial de una gaussiana.

## usar_cuda_conic

Controla rutas CUDA adicionales relacionadas con la representacion conica.

El raster tiled ya utiliza internamente componentes CUDA especificos.

---

# 9. Scheduler

## usar_scheduler_lento

Activa reduccion adaptativa de learning rates cuando el entrenamiento deja de mejorar.

Los campos asociados incluyen:

```text
scheduler_lento_window
scheduler_lento_min_delta
scheduler_lento_factor
scheduler_lento_min_lr
scheduler_lento_cooldown
```

Para smoke tests puede desactivarse.

---

# 10. Pruning post-training antiguo

Campos relacionados:

```text
umbral_pruning_post
pruning_n_samples
ejecutar_pruning_post
```

Cuando se utiliza el pipeline audiovisual actual:

```text
ejecutar_pruning_post = false
```

El pipeline entrena primero el modelo completo y posteriormente ejecuta el pruning adaptativo independiente.

---

# 11. Memoria de frames

## frames_en_cpu

Mantiene los frames en RAM y los transfiere segun necesidad.

Es util para reducir consumo de VRAM.

## frames_en_gpu_uint8

Permite otra estrategia de almacenamiento cuando se desea conservar frames en GPU.

## evitar_render_completo_en_train

Evita mantener todo el clip renderizado simultaneamente durante entrenamiento.

Es especialmente importante para videos largos o alta resolucion.

---

# 12. Metricas y salidas de video

## calcular_metricas

Calcula metricas de calidad despues del entrenamiento.

## usar_ssim

Activa SSIM.

## usar_lpips

Activa LPIPS.

LPIPS tiene mayor costo que metricas simples.

## guardar_visualizaciones

Genera visualizaciones adicionales.

## guardar_gif

Genera GIF cuando corresponde.

## guardar_frames_rasterizados

Conserva los frames reconstruidos.

El pipeline los necesita para varias comparaciones posteriores.

## guardar_checkpoints_intermedios

Controla si se conservan checkpoints de epochs intermedios.

## guardar_verificacion_visual

Genera salidas adicionales para revision visual.

## usar_metricas_streaming

Calcula metricas frame por frame sin mantener todos los renders simultaneamente en memoria.

---

# 13. Configuracion de audio Gabor

El archivo `audio.json` controla la representacion del audio mediante atomos Gabor.

## audio

Ruta del WAV utilizado para entrenamiento.

Cuando se ejecuta el pipeline, este WAV es extraido automaticamente desde el MP4 original.

## max_segundos

Duracion maxima de audio utilizada.

Debe ser coherente con el segmento seleccionado por el pipeline.

## sr

Sample rate.

Smoke:

```text
16000
```

Experimentos de mayor fidelidad pueden utilizar:

```text
44100
```

## device

Para CUDA:

```text
cuda
```

## seed

Semilla para reproducibilidad.

---

# 14. Atomos Gabor

## n_atomos

Cantidad general de atomos.

En entrenamiento estereo pueden existir valores especificos por componente.

## dominio

Para representacion estereo Mid-Side:

```text
MS
```

## n_atomos_mid

Numero de atomos para la componente Mid.

## n_atomos_side

Numero de atomos para la componente Side.

Si estos campos estan presentes, el entrenamiento estereo utiliza los valores especificos para cada componente.

## k_sigma

Controla hasta cuantos sigmas se evalua aproximadamente la influencia temporal de cada atomo.

## f_min_hz

Frecuencia minima permitida.

## f_max_hz

Frecuencia maxima.

Puede ser `null` para permitir que el sistema determine el limite apropiado segun el sample rate.

## sigma_inicial_samples

Permite fijar manualmente una escala temporal inicial.

Puede mantenerse en `null` cuando se utiliza inicializacion automatica.

---

# 15. Entrenamiento Gabor

## epochs

Numero de epochs de audio.

Smoke:

```text
1
```

## log_cada

Frecuencia de impresion de metricas durante entrenamiento.

## lambda_wave

Peso de la loss directa sobre waveform.

## lambda_mrstft

Peso de la loss multi-resolution STFT.

## lambda_sisdr

Peso de SI-SDR cuando esta componente se utiliza.

## mrstft_ffts

Tamanos FFT evaluados por la loss multi-resolution STFT.

Ejemplo smoke:

```text
[256, 512]
```

Ejemplo de mayor resolucion:

```text
[512, 1024, 2048]
```

---

# 16. Learning rates Gabor

El bloque `lrs` puede contener:

```text
mu_t
log_sigma
amp
freq_raw
phi
gain
```

Cada parametro controla una propiedad distinta de los atomos Gabor.

No se recomienda cambiar varios learning rates simultaneamente sin registrar el experimento.

---

# 17. Scheduler Gabor

## usar_scheduler

Activa scheduler de learning rate.

Campos asociados cuando esta activo:

```text
scheduler_factor
scheduler_paciencia
scheduler_min_lr
```

Para smoke tests puede desactivarse.

---

# 18. Inicializacion Gabor

## init_modo

Metodo utilizado para distribuir inicialmente los atomos.

Una opcion utilizada es:

```text
energia
```

## init_alpha

Parametro asociado a la inicializacion basada en energia.

## init_n_fft

FFT utilizada para analizar energia durante inicializacion.

## init_hop

Hop utilizado durante la inicializacion.

---

# 19. CUDA Gabor

## forzar_pytorch

Si es `false`, el sistema intenta utilizar la extension CUDA Gabor.

```text
false -> intentar CUDA
true  -> usar implementacion PyTorch
```

La implementacion PyTorch sirve como fallback y referencia.

---

# 20. Smoke test vs experimento real

El smoke oficial utiliza deliberadamente:

```text
20 frames
1 epoch de video
4000 gaussianas
grados temporales pequenos
1 epoch de audio
512 atomos Mid
256 atomos Side
1 intento de pruning
UINT16 SAFE
```

Su objetivo es responder solamente:

```text
¿El proyecto puede ejecutarse de inicio a fin?
```

No debe utilizarse para comparar calidad, compresion final ni rendimiento experimental.

---

# 21. Recomendaciones para nuevos experimentos

1. Copiar los tres templates a una nueva carpeta.
2. Asignar nombres unicos al pipeline, video, audio y clip.
3. Seleccionar el MP4 de entrada.
4. Definir inicio, duracion, FPS y resolucion.
5. Definir gaussianas y grados temporales.
6. Definir atomos Gabor y sample rate.
7. Ejecutar primero un smoke pequeno.
8. Ejecutar despues la configuracion completa.
9. No reutilizar nombres de experimentos si se cambiaron parametros importantes.
10. Conservar las configs exactas utilizadas para cada resultado de tesis.

---

# 22. Archivos relacionados

```text
docs/COMMANDS.md
docs/PIPELINE.md
docs/COMPRESSION.md
docs/VISUALIZATIONS.md
docs/INSTALLATION.md
docs/TROUBLESHOOTING.md
```

## limpiar_salida_pipeline

Si es `true`, elimina la salida anterior de `outputs/AV_PIPELINE/<nombre_pipeline>/` antes de iniciar una nueva ejecucion.

Esto evita conservar paquetes UINT16, renders, logs o videos finales pertenecientes a una configuracion anterior con el mismo nombre.

No elimina los resultados independientes de entrenamiento de video o audio.

Para ejecuciones reproducibles se recomienda:

```text
true
```

## limpiar_salida

Controla la limpieza de la carpeta propia de un entrenamiento de video o audio.

```json
"sobreescribir_salida": true,
"limpiar_salida": true
```

Con ambos valores en `true`, la carpeta anterior del experimento se elimina antes de comenzar.

Esto evita mezclar checkpoints, renders, logs o reconstrucciones de dos ejecuciones distintas que usen el mismo `nombre_experimento`.

El pipeline audiovisual activa esta opcion automaticamente en las configuraciones runtime de video y audio.
