# MBB-GS - Pruning y cuantizacion

Este documento describe las estrategias de reduccion de tamano utilizadas despues del entrenamiento del modelo de video.

El flujo principal es:

```text
checkpoint FP32
    |
    +--> pruning de gaussianas
            |
            +--> checkpoint FP32 podado
                    |
                    +--> cuantizacion UINT16
                            |
                            +--> reconstruccion
                            +--> comparacion contra baseline
```

Las herramientas se encuentran en:

```text
scripts/compression/
```

---

# 1. Objetivo

El objetivo de la etapa de compresion es reducir el tamano de la representacion entrenada sin introducir una degradacion visual significativa.

Se utilizan dos estrategias complementarias:

```text
1. reducir el numero de gaussianas
2. reducir la precision utilizada para almacenar coeficientes
```

El pruning elimina gaussianas completas.

La cuantizacion conserva las gaussianas restantes pero representa determinados parametros con enteros de 16 bits.

---

# 2. Baseline

Antes de evaluar pruning o cuantizacion se necesita un render baseline.

Normalmente corresponde a:

```text
outputs/<experimento>/frames_renderizados/
```

Ese baseline es la reconstruccion producida por el modelo antes de aplicar la transformacion que se esta evaluando.

Esto significa que el PSNR utilizado durante pruning y cuantizacion mide:

```text
modelo comprimido vs modelo baseline
```

y no necesariamente:

```text
modelo comprimido vs video original
```

Esta diferencia es importante al interpretar los resultados.

---

# 3. Pruning adaptativo

Script principal:

```text
scripts/compression/run_binary_pruning_adaptativo.py
```

El proceso calcula estadisticas temporales por gaussiana y construye un ranking para decidir que gaussianas son candidatas a eliminar.

Las metricas principales utilizadas por el ranking son:

```text
path_length_px
color_path
op_std
scale_std
```

Estas representan respectivamente informacion relacionada con:

```text
movimiento espacial
variacion temporal de color
variacion temporal de opacidad
variacion temporal de escala
```

---

# 4. Busqueda por porcentaje

El pruning adaptativo no necesita fijar un unico porcentaje desde el inicio.

Puede probar una secuencia como:

```text
10%
15%
20%
25%
...
```

controlada por:

```text
inicio_pct
paso_pct
min_pct
max_pct
```

Para una sola prueba:

```text
inicio_pct = 5
paso_pct = 5
min_pct = 5
max_pct = 5
```

---

# 5. Criterio PSNR

Cada modelo podado se renderiza y se compara contra el baseline.

El parametro:

```text
psnr_min
```

define el PSNR minimo necesario para aceptar una reduccion.

En una busqueda normal, el objetivo es conservar el mayor porcentaje de pruning que siga cumpliendo el umbral.

Un valor como:

```text
psnr_min = 0
```

solo tiene sentido para smoke tests donde se quiere obligar al pipeline a recorrer la etapa de pruning.

---

# 6. Comando de pruning adaptativo

Ejemplo:

```powershell
python scripts\compression\run_binary_pruning_adaptativo.py --exp outputs\EXP --checkpoint outputs\EXP\checkpoints\checkpoint_final.pt --baseline_frames outputs\EXP\frames_renderizados --fps 30 --device cuda --inicio_pct 10 --paso_pct 5 --min_pct 10 --max_pct 50 --psnr_min 65 --crear_video_ganador
```

La opcion:

```text
--crear_video_ganador
```

genera un MP4 correspondiente al mejor candidato aceptado.

---

# 7. Salidas del pruning

Las salidas se organizan normalmente en:

```text
outputs/<experimento>/binary_pruning/
```

Entre los archivos generados se encuentran:

```text
ids/
tests/
metrics/
checkpoints/
ganador/
resumen_busqueda_pruning.csv
resumen_busqueda_pruning.txt
```

El checkpoint ganador conserva solamente las gaussianas seleccionadas.

---

# 8. UINT16 SAFE

Scripts:

```text
pack_checkpoint_uint16.py
unpack_checkpoint_uint16.py
```

UINT16 SAFE es la estrategia conservadora de cuantizacion.

En el pipeline actual se cuantizan principalmente:

```text
mu_high
color_high
opacity_high
scale_high
```

mientras los otros valores pueden mantenerse en FP32.

Esta estrategia busca reducir los tensores temporalmente grandes sin reducir agresivamente la precision de toda la representacion.

---

# 9. Cuantizacion afin UINT16

Cada tensor se transforma de manera independiente.

Para un tensor con minimo `min` y maximo `max`, la escala utilizada es:

```text
scale = (max - min) / 65535
```

Los valores son convertidos al rango:

```text
0 ... 65535
```

y almacenados como:

```text
torch.uint16
```

Para reconstruir:

```text
x ~= min + q * scale
```

Cada tensor utiliza su propio minimo y escala.

---

# 10. Tensores constantes

Si:

```text
max == min
```

la escala se almacena como cero y el tensor cuantizado puede quedar completamente en cero.

Durante la reconstruccion se recupera correctamente el valor constante original.

---

# 11. depth_high igual a cero

La opcion:

```text
--omit_zero_depth_high
```

permite no almacenar `depth_high` cuando todos sus valores estan efectivamente en cero.

La forma del tensor queda registrada y se reconstruye posteriormente como un tensor de ceros.

Esto evita almacenar informacion redundante.

---

# 12. Empaquetar UINT16 SAFE

Ejemplo:

```powershell
python scripts\compression\pack_checkpoint_uint16.py --in_ckpt modelo_pruneado.pt --out_pkg modelo_uint16_safe.pkg.pt --other_float fp32 --quant_tensors mu_high color_high opacity_high scale_high --omit_zero_depth_high
```

Para la ruta SAFE utilizada por el pipeline se recomienda mantener:

```text
--other_float fp32
```

---

# 13. Desempaquetar UINT16 SAFE

Ejemplo:

```powershell
python scripts\compression\unpack_checkpoint_uint16.py --in_pkg modelo_uint16_safe.pkg.pt --out_ckpt modelo_uint16_safe_render.pt --out_float fp32
```

El checkpoint reconstruido puede utilizarse con los mismos scripts de render del modelo FP32.

---

# 14. UINT16 ALL

Scripts:

```text
pack_checkpoint_uint16_all.py
unpack_checkpoint_uint16_all.py
```

UINT16 ALL es una estrategia mas agresiva.

En lugar de seleccionar solamente determinados coeficientes, cuantiza los tensores flotantes del state_dict que son compatibles.

Los elementos no flotantes y la metadata necesaria se conservan.

---

# 15. Empaquetar UINT16 ALL

```powershell
python scripts\compression\pack_checkpoint_uint16_all.py --in_ckpt modelo_pruneado.pt --out_pkg modelo_uint16_all.pkg.pt --omit_zero_depth_high
```

Desempaquetar:

```powershell
python scripts\compression\unpack_checkpoint_uint16_all.py --in_pkg modelo_uint16_all.pkg.pt --out_ckpt modelo_uint16_all_render.pt
```

---

# 16. SAFE vs ALL

Resumen conceptual:

```text
UINT16 SAFE
  menor agresividad
  seleccion de tensores grandes
  otros parametros pueden permanecer FP32

UINT16 ALL
  mayor agresividad
  cuantizacion de todos los tensores flotantes compatibles
  potencial de mayor reduccion
  requiere mayor cuidado al validar calidad
```

No debe asumirse automaticamente que UINT16 ALL es mejor solamente porque ocupa menos espacio.

La seleccion debe considerar tamano y degradacion.

---

# 17. Por que UINT16 puede superar a FP16

FP16 utiliza un formato flotante generico con rango y precision distribuidos para muchos tipos de valores.

La estrategia UINT16 utilizada por el proyecto calcula una escala especifica para cada tensor.

Esto permite utilizar los 65536 niveles disponibles solamente dentro del rango real observado en ese tensor.

Por ejemplo, un tensor cuyos coeficientes se encuentren solamente entre:

```text
-0.005 y 0.005
```

puede distribuir sus niveles UINT16 especificamente dentro de ese intervalo.

Por esa razon, para determinados coeficientes entrenados, UINT16 afin puede proporcionar una relacion tamano-error mejor que convertir directamente todo a FP16.

El resultado debe comprobarse experimentalmente mediante render y metricas.

---

# 18. Comparacion despues de cuantizar

Despues de desempaquetar, el pipeline vuelve a renderizar el checkpoint.

Los frames cuantizados se comparan contra:

```text
frames_renderizados del modelo baseline
```

Se genera normalmente un CSV como:

```text
psnr_baseline_vs_uint16_safe.csv
psnr_baseline_vs_uint16_all.csv
```

---

# 19. PSNR infinito

Si dos renders son exactamente iguales a nivel de pixel, el MSE es cero.

En ese caso:

```text
PSNR = infinito
```

Esto puede ocurrir especialmente en smoke tests pequenos cuando las diferencias numericas no llegan a cambiar los valores finales guardados en PNG.

Un PSNR infinito no significa que el modelo tenga calidad perfecta respecto al video original.

Significa solamente que las dos reconstrucciones comparadas son identicas segun la representacion utilizada para la comparacion.

---

# 20. Compresion adicional con 7-Zip

El pipeline intenta comprimir los paquetes UINT16 mediante 7-Zip cuando esta disponible.

Esto es una compresion sin perdida aplicada despues de la cuantizacion.

Por tanto existen dos niveles distintos:

```text
cuantizacion -> modifica la representacion numerica
7-Zip        -> comprime bytes sin modificar la representacion
```

---

# 21. Configuracion desde pipeline.json

El bloque:

```text
video_cuantizacion
```

permite controlar:

```text
generar_uint16_safe
generar_uint16_all
usar_para_video_final
```

Ejemplo conservador:

```text
generar_uint16_safe = true
generar_uint16_all = false
usar_para_video_final = uint16_safe
```

Este fue el esquema utilizado por el smoke oficial.

---

# 22. Smoke oficial

En el smoke de 20 frames se utilizo:

```text
4000 gaussianas iniciales
5% pruning
3800 gaussianas restantes
UINT16 SAFE
UINT16 ALL desactivado
```

El checkpoint FP32 podado tenia aproximadamente:

```text
0.78 MiB
```

El paquete UINT16 SAFE quedo aproximadamente en:

```text
0.47 MiB
```

y 7-Zip lo redujo aun mas.

Estos resultados solamente validan que la cadena de compresion funciona.

No deben utilizarse como resultado final de calidad o compresion de la tesis.

---

# 23. Orden recomendado

Para un experimento real:

```text
1. entrenar el modelo completo
2. guardar baseline renderizado
3. calcular estadisticas por gaussiana
4. ejecutar pruning adaptativo
5. seleccionar checkpoint ganador
6. generar UINT16 SAFE
7. renderizar y medir
8. opcionalmente generar UINT16 ALL
9. comparar tamano y calidad
10. seleccionar representacion final
```

---

# 24. Documentacion relacionada

```text
docs/PIPELINE.md
docs/CONFIGURATION.md
docs/COMMANDS.md
docs/VISUALIZATIONS.md
```
