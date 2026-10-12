# MBB-GS - Visualizaciones

Este documento describe las herramientas disponibles para analizar visualmente las gaussianas temporales.

Los scripts se encuentran en:

```text
scripts/visualization/
```

Scripts disponibles:

```text
viz_gaussian_stats.py
viz_trayectorias_marcadores.py
viz_trayectorias_marcadores_rango.py
viz_elipses_velocidades.py
viz_atributos_gaussiana_tiempo.py
viz_rank_gaussianas_rango.py
viz_render_subset.py
viz_tira_evolucion.py
```

> En los comandos siguientes, EXP representa el nombre real de un experimento.

---

# 1. Estadisticas por gaussiana

Script:

```text
viz_gaussian_stats.py
```

Analiza todas las gaussianas y calcula estadisticas temporales relacionadas con:

```text
movimiento
velocidad
cambio de color
variacion de opacidad
variacion de escala
rotacion
visibilidad
```

Comando:

```powershell
python scripts\visualization\viz_gaussian_stats.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --device cuda
```

Para reducir memoria y tiempo:

```powershell
python scripts\visualization\viz_gaussian_stats.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --chunk 2048 --sample_every 2 --device cuda
```

Salidas principales:

```text
viz_stats/gaussian_stats.csv
viz_stats/top_movimiento.csv
viz_stats/top_color.csv
viz_stats/top_opacidad.csv
viz_stats/top_estaticas_visibles.csv
viz_stats/resumen.txt
```

Esta informacion tambien es utilizada por el pruning adaptativo.

---

# 2. Trayectorias de gaussianas

Script:

```text
viz_trayectorias_marcadores.py
```

Dibuja la trayectoria temporal de gaussianas sobre un frame de referencia.

Puede seleccionar gaussianas mediante:

```text
top
aleatorio
region
IDs manuales
CSV de IDs
```

Ejemplo con gaussianas de mayor movimiento:

```powershell
python scripts\visualization\viz_trayectorias_marcadores.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n 20 --frame_fondo 0 --device cuda
```

Generar tambien GIF:

```powershell
python scripts\visualization\viz_trayectorias_marcadores.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n 20 --gif --fps 12 --paso 2 --estela 15 --device cuda
```

El argumento `estela` controla cuantos pasos anteriores de la trayectoria permanecen visibles en la animacion.

---

# 3. Trayectorias dentro de un rango temporal

Script:

```text
viz_trayectorias_marcadores_rango.py
```

Es similar al script anterior, pero restringe el analisis a un intervalo concreto del video.

Ejemplo:

```powershell
python scripts\visualization\viz_trayectorias_marcadores_rango.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n 20 --inicio 0 --fin 100 --salida outputs\EXP\viz_trayectorias_rango --device cuda
```

Con GIF:

```powershell
python scripts\visualization\viz_trayectorias_marcadores_rango.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n 20 --inicio 0 --fin 100 --salida outputs\EXP\viz_trayectorias_rango --gif --fps 16 --paso 2 --estela 12 --device cuda
```

---

# 4. Elipses y velocidades

Script:

```text
viz_elipses_velocidades.py
```

Genera dos visualizaciones para un frame:

```text
forma y orientacion de las gaussianas
vectores de velocidad entre frames consecutivos
```

Ejemplo usando las gaussianas con mayor movimiento:

```powershell
python scripts\visualization\viz_elipses_velocidades.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --frame 50 --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n_max 100 --device cuda
```

Opcionalmente se puede utilizar como fondo el render del modelo:

```powershell
python scripts\visualization\viz_elipses_velocidades.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --frame 50 --fondo_dir outputs\EXP\frames_renderizados --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --n_max 100 --device cuda
```

Parametros utiles:

```text
k_sigma          -> tamano visual de la elipse
escala_flecha    -> escala grafica de los vectores de velocidad
n_max            -> numero maximo de gaussianas dibujadas
```

---

# 5. Atributos temporales de una gaussiana

Script:

```text
viz_atributos_gaussiana_tiempo.py
```

Grafica la evolucion temporal de una unica gaussiana.

Incluye:

```text
posicion X/Y
opacidad
escala X/Y
color R/G/B
```

Seleccionar directamente un ID:

```powershell
python scripts\visualization\viz_atributos_gaussiana_tiempo.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --id 100 --inicio 0 --fin 100 --frame_ref 50 --salida outputs\EXP\viz_atributos_gaussiana --device cuda
```

Tambien puede leer la primera gaussiana de un CSV:

```powershell
python scripts\visualization\viz_atributos_gaussiana_tiempo.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --inicio 0 --fin 100 --frame_ref 50 --salida outputs\EXP\viz_atributos_gaussiana --device cuda
```

Salida principal:

```text
atributos_gaussiana_seleccionada.png
gaussiana_id.txt
```

---

# 6. Ranking dentro de un rango temporal

Script:

```text
viz_rank_gaussianas_rango.py
```

Calcula rankings de gaussianas solamente dentro de un intervalo del video.

Ejemplo:

```powershell
python scripts\visualization\viz_rank_gaussianas_rango.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --inicio 0 --fin 100 --salida outputs\EXP\ranking_rango --device cuda
```

Para reducir costo:

```powershell
python scripts\visualization\viz_rank_gaussianas_rango.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --inicio 0 --fin 100 --salida outputs\EXP\ranking_rango --chunk 2048 --sample_every 2 --device cuda
```

Genera, entre otros:

```text
stats_rango_<inicio>_<fin>.csv
top30_mov_vis.csv
top30_importancia_visual.csv
top200_mov_vis.csv
top200_importancia_visual.csv
resumen.txt
```

---

# 7. Render de subconjuntos de gaussianas

Script:

```text
viz_render_subset.py
```

Permite comprobar visualmente que aporta un subconjunto de gaussianas.

Modos disponibles:

```text
only
exclude
only_static
exclude_static
```

## Renderizar solo gaussianas seleccionadas

```powershell
python scripts\visualization\viz_render_subset.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --modo only --salida outputs\EXP\subset_movimiento --device cuda --crear_video --fps 30
```

## Excluir gaussianas seleccionadas

```powershell
python scripts\visualization\viz_render_subset.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --ids_csv outputs\EXP\viz_stats\top_movimiento.csv --modo exclude --salida outputs\EXP\sin_top_movimiento --device cuda --crear_video --fps 30
```

## Detectar y excluir gaussianas casi estaticas

```powershell
python scripts\visualization\viz_render_subset.py --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --stats_csv outputs\EXP\viz_stats\gaussian_stats.csv --modo exclude_static --salida outputs\EXP\sin_static --static_path_max 1.0 --static_color_max 0.01 --static_opstd_max 0.005 --static_scalestd_max 0.01 --device cuda --crear_video --fps 30
```

El script tambien genera metricas simples y un resumen del subconjunto.

---

# 8. Evolucion durante entrenamiento

Script:

```text
viz_tira_evolucion.py
```

Visualiza artefactos guardados durante el entrenamiento.

Busca principalmente:

```text
outputs/EXP/verificacion/epochXXXX_frameYYYY.png
outputs/EXP/evol_mu/epochXXXX.npz
```

Comando:

```powershell
python scripts\visualization\viz_tira_evolucion.py --exp outputs\EXP
```

Con GIF:

```powershell
python scripts\visualization\viz_tira_evolucion.py --exp outputs\EXP --gif --fps 2 --max_gauss 12
```

> Este script no crea los archivos evolutivos. Solo visualiza los que fueron guardados durante entrenamiento.

---

# 9. Flujo recomendado para analizar un experimento

Una secuencia util es:

```text
1. Entrenar modelo
2. Generar gaussian_stats
3. Revisar top_movimiento / top_color / top_opacidad
4. Dibujar trayectorias
5. Dibujar elipses y velocidades
6. Revisar atributos de gaussianas concretas
7. Renderizar subsets
8. Comparar el efecto visual de eliminar grupos
```

---

# 10. Ayuda de cada comando

Todos los scripts utilizan argparse.

Para consultar parametros disponibles:

```powershell
python scripts\visualization\viz_gaussian_stats.py --help
python scripts\visualization\viz_trayectorias_marcadores.py --help
python scripts\visualization\viz_trayectorias_marcadores_rango.py --help
python scripts\visualization\viz_elipses_velocidades.py --help
python scripts\visualization\viz_atributos_gaussiana_tiempo.py --help
python scripts\visualization\viz_rank_gaussianas_rango.py --help
python scripts\visualization\viz_render_subset.py --help
python scripts\visualization\viz_tira_evolucion.py --help
```

---

# 11. Documentacion relacionada

```text
docs/COMMANDS.md
docs/CONFIGURATION.md
docs/PIPELINE.md
docs/COMPRESSION.md
```
