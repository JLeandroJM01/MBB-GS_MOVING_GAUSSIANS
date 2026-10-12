"""
GaussianasPolinomial2D: modelo temporal de gaussianas 2D cuyos parametros
se representan mediante coeficientes polinomiales.

La base temporal se proporciona externamente. Chebyshev es la base principal;
la base monomial se conserva para comparaciones experimentales.

Separacion de a_0 y a_high:
----------------------------------
Cada parametro p tiene grado q_p. Almacenamos sus coeficientes en DOS
tensores separados:
    p_a0   : nn.Parameter de shape (N, dim_p, 1)         -- coef a_0
    p_high : nn.Parameter de shape (N, dim_p, q_p)       -- coefs a_1..a_{q_p}

Razones:
  1. El optimizer recibe (via param_groups) lr distinto para a_0 vs a_high:
        a_0 con lr base; a_high con lr base * 0.1 (mas conservador).
  2. La inicializacion es naturalmente diferente: a_0 lleva el valor
     constante "razonable" del parametro; a_high arranca en 0.
  3. La penalizacion de smoothness opera solo sobre a_high (k=0 -> peso k^2=0).
     Tenerlo separado lo deja fuera automaticamente.

Al evaluar concatenamos:
    coefs = cat([p_a0, p_high], dim=-1)    # (N, dim_p, q_p+1)
    p(t_j) = coefs @ B[j, :q_p+1]
"""
import numpy as np
import torch
from torch import nn



class GaussianasPolinomial2D(nn.Module):

    def __init__(self, n_gaussianas, n_frames, grados, H, W, device,
                 escala_inicial_px=5.0, frame_0_imagen=None, semilla=42):
        """
        Args:
            n_gaussianas       : N
            n_frames           : T del clip
            grados             : dict {'mu':40, 'opacity':40, 'color':20, 'scale':20, 'theta':20, 'depth':20}
            H, W               : alto, ancho de la imagen
            device             : torch.device
            escala_inicial_px  : log(esto) va al a_0 de scale
            frame_0_imagen     : (H, W, 3) en [0, 1] opcional; si se da, se muestrea
                                 color de cada gaussiana en su posicion inicial.
            semilla            : reproducibilidad
        """
        super().__init__()

        self.N = n_gaussianas
        self.n_frames = n_frames
        self.H = H
        self.W = W
        self.grados = dict(grados)
        self.device = device
        self.log_scale_min = float(np.log(0.5))
        self.log_scale_max = float(np.log(max(H, W)))

        g = torch.Generator(device='cpu').manual_seed(semilla)

        # === mu ============================================================
        mu_a0_init = torch.empty(n_gaussianas, 2, 1)
        mu_a0_init[:, 0, 0] = torch.rand(n_gaussianas, generator=g) * H        # filas
        mu_a0_init[:, 1, 0] = torch.rand(n_gaussianas, generator=g) * W        # columnas
        mu_high_init = torch.zeros(n_gaussianas, 2, grados['mu'])
        self.mu_a0   = nn.Parameter(mu_a0_init.to(device))
        self.mu_high = nn.Parameter(mu_high_init.to(device))

        # === opacity =======================================================
        opacity_a0_init = torch.zeros(n_gaussianas, 1)                          # sigmoid(0)=0.5
        opacity_high_init = torch.zeros(n_gaussianas, grados['opacity'])
        self.opacity_a0   = nn.Parameter(opacity_a0_init.to(device))
        self.opacity_high = nn.Parameter(opacity_high_init.to(device))

        # === color =========================================================
        if frame_0_imagen is not None:
            assert frame_0_imagen.shape == (H, W, 3)
            color_a0_real = _muestrear_color_bilineal(
                frame_0_imagen.detach().cpu(), mu_a0_init[:, :, 0]
            )                                                                   # (N, 3) en [0, 1]
            color_a0_init = _logit(color_a0_real.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1)
        else:
            color_a0_init = ((torch.rand(n_gaussianas, 3, 1, generator=g) - 0.5) * 2.0)
        color_high_init = torch.zeros(n_gaussianas, 3, grados['color'])
        self.color_a0   = nn.Parameter(color_a0_init.to(device))
        self.color_high = nn.Parameter(color_high_init.to(device))

        # === scale =========================================================
        scale_a0_init = torch.full((n_gaussianas, 2, 1), float(np.log(escala_inicial_px)))
        scale_high_init = torch.zeros(n_gaussianas, 2, grados['scale'])
        self.scale_a0   = nn.Parameter(scale_a0_init.to(device))
        self.scale_high = nn.Parameter(scale_high_init.to(device))

        # === theta =========================================================
        theta_a0_init = torch.zeros(n_gaussianas, 1)
        theta_high_init = torch.zeros(n_gaussianas, grados['theta'])
        self.theta_a0   = nn.Parameter(theta_a0_init.to(device))
        self.theta_high = nn.Parameter(theta_high_init.to(device))

        # === depth =========================================================
        depth_a0_init = ((torch.rand(n_gaussianas, 1, generator=g) - 0.5) * 2.0)
        depth_high_init = torch.zeros(n_gaussianas, grados['depth'])
        self.depth_a0   = nn.Parameter(depth_a0_init.to(device))
        self.depth_high = nn.Parameter(depth_high_init.to(device))
        # --------
        self.smooth_factors = {}
        for nombre, grado in self.grados.items():
            if grado > 0:
                k_idx = torch.arange(1, grado + 1, device=device, dtype=torch.float32)
                self.smooth_factors[nombre] = k_idx ** 2


    # ---- helpers --------------------------------------------------------
    def _coefs_completos(self, a0, hi):
        """cat a_0 y a_high en el eje del grado: (..., q+1)."""
        return torch.cat([a0, hi], dim=-1)


    def parametros_temporales(self):
        """
        Devuelve dict {nombre: (a_0, a_high, grado, dim_p)} para el optimizador
        y la regularizacion de smoothness.

        dim_p: dimensionalidad "por gaussiana" (mu/scale=2, color=3, resto=1).
        """
        return {
            'mu':      (self.mu_a0,      self.mu_high,      self.grados['mu'],      2),
            'opacity': (self.opacity_a0, self.opacity_high, self.grados['opacity'], 1),
            'color':   (self.color_a0,   self.color_high,   self.grados['color'],   3),
            'scale':   (self.scale_a0,   self.scale_high,   self.grados['scale'],   2),
            'theta':   (self.theta_a0,   self.theta_high,   self.grados['theta'],   1),
            'depth':   (self.depth_a0,   self.depth_high,   self.grados['depth'],   1),
        }


    def parametros_temporales_items(self):
        return (
            ('mu',      self.mu_a0,      self.mu_high,      self.grados['mu'],      2),
            ('opacity', self.opacity_a0, self.opacity_high, self.grados['opacity'], 1),
            ('color',   self.color_a0,   self.color_high,   self.grados['color'],   3),
            ('scale',   self.scale_a0,   self.scale_high,   self.grados['scale'],   2),
            ('theta',   self.theta_a0,   self.theta_high,   self.grados['theta'],   1),
            ('depth',   self.depth_a0,   self.depth_high,   self.grados['depth'],   1),
        )

    def evaluar_en_frame(self, frame_idx, matrices_base):
        """
        Evalua todos los polinomios en el frame `frame_idx`.

        matrices_base: dict {grado: tensor (n_frames, grado+1)}.

        Devuelve dict con valores ya activados:
            mu      : (N, 2)
            scale   : (N, 2)   ya con exp
            theta   : (N,)
            opacity : (N,)     ya con sigmoid
            color   : (N, 3)   ya con sigmoid
            depth   : (N,)
        """
        out = {}
        for nombre, a0, hi, grado, dim_p in self.parametros_temporales_items():
            B = matrices_base[grado]

            # B[frame_idx, 0] siempre es 1.0 (T_0 de Chebyshev).
            fila_high = B[frame_idx, 1:grado + 1]

            if grado > 0:
                val = a0.squeeze(-1) + (hi * fila_high).sum(dim=-1)
            else:
                val = a0.squeeze(-1)

            if dim_p == 1:
                val = val.squeeze(-1)

            out[nombre + '_raw'] = val

        return _aplicar_activaciones(out, self.log_scale_min, self.log_scale_max)

    def evaluar_batch_completo(self, matrices_base):
        """
        Evalua TODOS los frames a la vez (para batch full).

        Devuelve dict con valores ya activados con shape extra n_frames:
            mu      : (n_frames, N, 2)
            scale   : (n_frames, N, 2)
            theta   : (n_frames, N)
            opacity : (n_frames, N)
            color   : (n_frames, N, 3)
            depth   : (n_frames, N)
        """
        out = {}
        for nombre, (a0, hi, grado, dim_p) in self.parametros_temporales().items():
            B = matrices_base[grado]                               # (n_frames, grado+1)
            coefs = self._coefs_completos(a0, hi)                  # (N, dim_p, grado+1)
            # producto: (N, dim_p, grado+1) @ (grado+1, n_frames) = (N, dim_p, n_frames)
            # despues permutamos a (n_frames, N, dim_p)
            val = coefs @ B.T                                       # (N, dim_p, n_frames)
            if dim_p == 1:
                val = val.squeeze(1)                                # (N, n_frames)
                val = val.transpose(0, 1)                           # (n_frames, N)
            else:
                val = val.permute(2, 0, 1).contiguous()             # (n_frames, N, dim_p)
            out[nombre + '_raw'] = val

        return _aplicar_activaciones(out, self.log_scale_min, self.log_scale_max)


    def numero_gaussianas(self):
        return self.mu_a0.shape[0]

    def numero_gausianas(self):
        return self.numero_gaussianas()


    def state_dict_coefs(self):
        d = {}
        for nombre, (a0, hi, _, _) in self.parametros_temporales().items():
            d[f"{nombre}_a0"]   = a0.detach().cpu()
            d[f"{nombre}_high"] = hi.detach().cpu()
        d['grados'] = self.grados
        d['N']      = self.N
        d['H']      = self.H
        d['W']      = self.W
        d['n_frames'] = self.n_frames
        return d



# ===========================================================================
# helpers
# ===========================================================================

def _aplicar_activaciones(out, log_min, log_max):
    """Aplica activaciones para single-frame. Modifica `out` in place y lo devuelve."""

    out['mu']      = out['mu_raw']
    out['theta']   = out['theta_raw']
    out['depth']   = out['depth_raw']
    out['opacity'] = torch.sigmoid(out['opacity_raw'])
    out['color']   = torch.sigmoid(out['color_raw'])
    # clamp de scale_raw para evitar exp gigante con coefs altos extremos
    out['scale']   = torch.exp(out['scale_raw'].clamp(min=log_min, max=log_max))
    return out



def _logit(p):
    return torch.log(p / (1.0 - p))


def _muestrear_color_bilineal(img_hw3, pos_fila_col):
    """Color en (fila, col) flotantes con interpolacion bilineal, clamp al borde."""
    H, W, _ = img_hw3.shape
    f = pos_fila_col[:, 0].clamp(0, H - 1)
    c = pos_fila_col[:, 1].clamp(0, W - 1)
    f0 = f.floor().long().clamp(0, H - 1)
    f1 = (f0 + 1).clamp(0, H - 1)
    c0 = c.floor().long().clamp(0, W - 1)
    c1 = (c0 + 1).clamp(0, W - 1)
    df = (f - f0.float()).unsqueeze(-1)
    dc = (c - c0.float()).unsqueeze(-1)
    Ia = img_hw3[f0, c0]
    Ib = img_hw3[f0, c1]
    Ic = img_hw3[f1, c0]
    Id = img_hw3[f1, c1]
    return (Ia * (1 - df) * (1 - dc) + Ib * (1 - df) * dc +
            Ic * df * (1 - dc) + Id * df * dc)



# ===========================================================================
# tests
# ===========================================================================

def _tests():
    from gs2d_video.core.bases import construir_matriz_chebyshev

    device = torch.device('cpu')
    n_g, n_f, H, W = 7, 20, 32, 32
    grados = {'mu': 8, 'opacity': 8, 'color': 4, 'scale': 4, 'theta': 4, 'depth': 4}

    modelo = GaussianasPolinomial2D(n_g, n_f, grados, H, W, device, semilla=0)

    matrices = {
        8: construir_matriz_chebyshev(n_f, 8, device),
        4: construir_matriz_chebyshev(n_f, 4, device),
    }

    # eval en frame 0 con a_high=0 -> p(t) = a_0 * T_0(t) = a_0 * 1
    out_frame = modelo.evaluar_en_frame(0, matrices)
    assert out_frame['mu'].shape == (n_g, 2)
    assert out_frame['opacity'].shape == (n_g,)
    assert out_frame['color'].shape == (n_g, 3)
    assert out_frame['scale'].shape == (n_g, 2)

    out_batch = modelo.evaluar_batch_completo(matrices)
    assert out_batch['mu'].shape == (n_f, n_g, 2)
    assert out_batch['opacity'].shape == (n_f, n_g)
    assert out_batch['color'].shape == (n_f, n_g, 3)
    assert out_batch['scale'].shape == (n_f, n_g, 2)

    out_single_5 = modelo.evaluar_en_frame(5, matrices)
    assert torch.allclose(out_batch['mu'][5], out_single_5['mu'], atol=1e-5), \
        "batch vs single no coincide"
    assert torch.allclose(out_batch['opacity'][5], out_single_5['opacity'], atol=1e-5)

    print("[modelo] OK")


if __name__ == "__main__":
    _tests()
