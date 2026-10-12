# Model and Temporal Representation

Source: `src/gs2d_video/core/modelo.py`, `src/gs2d_video/core/bases.py`

## What a Gaussian is

Each of the $N$ Gaussians carries six attributes:

| Attribute | Symbol | Dimension | Meaning |
| --- | --- | ---: | --- |
| `mu` | $\mu_i$ | 2 | position in the image, as (row, column) in pixels |
| `opacity` | $\alpha_i$ | 1 | how visible it is; sigmoid-activated |
| `color` | $c_i$ | 3 | RGB contribution; sigmoid-activated |
| `scale` | $s_i$ | 2 | extent along each axis in pixels; exponential-activated |
| `theta` | $\theta_i$ | 1 | orientation |
| `depth` | $d_i$ | 1 | compositing order within a tile |

The same $N$ Gaussians persist for the whole clip. Nothing is created or
destroyed between frames.

## Attributes as functions of time

Every attribute is a polynomial in normalised time. For frame $j$ of a clip of
$T$ frames, time is mapped to $[-1, 1]$:

$$\tau(j) = \frac{2j}{T-1} - 1$$

and each attribute $p$ of Gaussian $i$ is

$$p_i(t) = a_{i,0}^{(p)} + \sum_{k=1}^{K_p} a_{i,k}^{(p)} \, T_k(\tau(t))$$

where $T_k$ is the Chebyshev polynomial of the first kind of degree $k$. The
coefficients $a^{(p)}_{i,k}$ are what training learns.

$a_{i,0}$ is the attribute's base value; the remaining coefficients describe how
it moves. Each attribute gets its own degree $K_p$, because they do not need the
same temporal capacity — position and opacity change a lot, depth and rotation
usually do not. A typical configuration uses degrees 100 / 80 / 30 / 12 / 6 / 4
for `mu` / `opacity` / `color` / `scale` / `theta` / `depth`.

## Why Chebyshev

The recurrence is standard:

$$T_0(t) = 1, \quad T_1(t) = t, \quad T_k(t) = 2t\,T_{k-1}(t) - T_{k-2}(t)$$

Two properties matter here. Chebyshev polynomials are **bounded**, $|T_k(t)| \le 1$
on $[-1,1]$, so a degree-100 basis does not produce values that dwarf the
low-order ones. And they are **orthogonal** with respect to the weight
$1/\sqrt{1-t^2}$, so the coefficients are close to independent and the
optimisation stays well conditioned at high degree.

A monomial basis $t^k$ has neither property. Its Vandermonde matrix on equally
spaced nodes is notoriously ill-conditioned, and the reconstruction error blows
up near the interval endpoints. The comparison is in [Results](Results): under
identical settings the monomial basis loses about 7 dB of PSNR.

The monomial basis is kept in `bases.py` for exactly that ablation, and for
loading old checkpoints. It is not the main path.

Both matrices are built in `float64` on CPU for numerical stability and then
cast to the requested dtype and device. The result is a matrix $B$ of shape
$(T, K+1)$, with $B[j,k] = T_k(\tau(j))$; one matrix is built per distinct
degree in the configuration.

## How the coefficients are stored

Each attribute is split across **two** tensors rather than one:

```
p_a0    nn.Parameter of shape (N, dim_p, 1)      the a_0 coefficient
p_high  nn.Parameter of shape (N, dim_p, K_p)    coefficients a_1 .. a_K
```

That split is not cosmetic. It buys three things at once:

1. **Separate learning rates.** The optimiser gives `a_0` and `a_high` different
   rates, since the base value and the temporal variation need different step
   sizes. See [Training](Training).
2. **Different initialisation.** `a_0` starts at a sensible constant — a sampled
   colour, a chosen scale — while `a_high` starts at zero, so the model begins
   as a static image and grows motion from there.
3. **Free smoothness regularisation.** The penalty weights coefficient $k$ by
   $k^2$, so $k=0$ contributes nothing. Keeping `a_0` in its own tensor excludes
   it automatically instead of by masking.

To evaluate, the two are concatenated back along the degree axis and multiplied
by the basis matrix.

## Initialisation

Positions are sampled uniformly over the image. Colours can be sampled from
frame 0 by bilinear interpolation at each Gaussian's initial position
(`inicializar_color_desde_frame0`), which starts training much closer to the
target than random colours. Opacity starts at logit 0, so sigmoid gives 0.5.
Scale starts at $\log(\texttt{escala\_inicial\_px})$. All temporal coefficients
start at zero.

Everything is driven by an explicit `torch.Generator` seeded from the config, so
the initial state is reproducible.

## Activations

Raw polynomial outputs are mapped to valid ranges:

- `opacity` and `color` through a sigmoid,
- `scale` through an exponential, clamped to
  $[\log 0.5, \log \max(H,W)]$ beforehand so that extreme high-order
  coefficients cannot produce an enormous exponential,
- `mu`, `theta` and `depth` used directly.

## Evaluating a frame

```
t  →  τ(t)  →  G(t)  →  Î(t)
```

`evaluar_en_frame(j, matrices_base)` returns the activated attributes of all
Gaussians at frame `j`; `evaluar_batch_completo` does every frame at once with a
single matrix product, which is what the visualisations use. Both go through the
same activation function, and the model has a test asserting they agree.

The rendered frame is then produced by the [CUDA rasterizer](CUDA-Rasterizer).

## Model size

A checkpoint stores coefficients only. For $N$ Gaussians the count is

$$N \cdot \sum_p \dim(p) \cdot (K_p + 1)$$

float32 values. With the typical degrees above that is 414 values, or about
1.6 KB, per Gaussian — dominated by `mu`, which alone accounts for 202 of them.
A 150k-Gaussian model is therefore around 250 MB, which is why
[pruning and quantisation](Pruning-and-Quantization) matter for the final
artefact.
