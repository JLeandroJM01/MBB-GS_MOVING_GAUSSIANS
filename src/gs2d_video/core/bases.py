"""
Base polinomica de Chebyshev de primer tipo, indexada en el frame j de un
clip de N_frames:

    t_cheb(j) = 2 * j / (N_frames - 1) - 1     en [-1, 1]
    recurrencia:
        T_0(t) = 1
        T_1(t) = t
        T_k(t) = 2 t T_{k-1}(t) - T_{k-2}(t)
    fila B[j] = [T_0(t_cheb), T_1(t_cheb), ..., T_q(t_cheb)]

Propiedades clave:
  - |T_k(t)| <= 1 en [-1, 1]: base acotada.
  - Es una base ORTOGONAL con respecto al peso 1/sqrt(1-t^2). En la practica
    los coefs son aproximadamente independientes y el problema de
    optimizacion esta mucho mejor condicionado para grados altos.
  - Error de interpolacion distribuido uniformemente -- no explota en los
    bordes (a diferencia de una Vandermonde monomial sobre nodos
    equiespaciados).

La matriz B tiene shape (N_frames, q+1). El modelo hace simplemente:
    params(t_j) = coefs @ B[j, :q+1]
"""
import torch


# La matriz se construye en CPU con float64 para mejorar la estabilidad
# numerica y luego se convierte al dtype y device solicitados.
def construir_matriz_chebyshev(n_frames, grado_max, device='cpu', dtype=torch.float32):
    """
    Matriz B (n_frames, grado_max+1) en base de Chebyshev de primer tipo.
    B[j, k] = T_k(t_cheb_j), con t_cheb_j = 2 * j / (N - 1) - 1.
    """
    if n_frames < 2:
        t = torch.zeros(n_frames, dtype=torch.float64)
    else:
        idx = torch.arange(n_frames, dtype=torch.float64)
        t = 2.0 * (idx / (n_frames - 1)) - 1.0                       # [-1, 1]

    B = torch.empty(n_frames, grado_max + 1, dtype=torch.float64)
    B[:, 0] = 1.0
    if grado_max >= 1:
        B[:, 1] = t
    for k in range(2, grado_max + 1):
        B[:, k] = 2.0 * t * B[:, k - 1] - B[:, k - 2]

    return B.to(dtype=dtype, device=device)



# ===========================================================================
# tests
# ===========================================================================

def _tests():
    """Verifica identidades basicas de la base de Chebyshev."""
    B = construir_matriz_chebyshev(5, 4)
    t = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert torch.allclose(B[:, 0], torch.ones(5))                       # T_0 = 1
    assert torch.allclose(B[:, 1], t)                                    # T_1 = t
    assert torch.allclose(B[:, 2], 2 * t ** 2 - 1, atol=1e-6)            # T_2
    assert torch.allclose(B[:, 3], 4 * t ** 3 - 3 * t, atol=1e-6)        # T_3
    assert torch.allclose(B[:, 4], 8 * t ** 4 - 8 * t ** 2 + 1, atol=1e-6)
    assert (B.abs() <= 1.0 + 1e-6).all(), "Chebyshev fuera de [-1, 1]"

    # estabilidad numerica: con grado 40 los valores deben quedarse <= 1
    Bbig = construir_matriz_chebyshev(90, 40)
    assert (Bbig.abs() <= 1.0 + 1e-5).all(), "Cheb grado 40: overflow inesperado"

    print("[bases] tests OK")


if __name__ == "__main__":
    _tests()



def construir_matriz_monomial(n_frames, grado_max, device='cpu', dtype=torch.float32):
    """
    Matriz B (n_frames, grado_max+1) en base monomial.
    B[j, k] = t_j^k, con t_j normalizado en [-1, 1].

    Se usa para comparar contra Chebyshev en las mismas condiciones.
    """
    if n_frames < 2:
        t = torch.zeros(n_frames, dtype=torch.float64)
    else:
        idx = torch.arange(n_frames, dtype=torch.float64)
        t = 2.0 * (idx / (n_frames - 1)) - 1.0

    B = torch.empty(n_frames, grado_max + 1, dtype=torch.float64)
    B[:, 0] = 1.0
    for k in range(1, grado_max + 1):
        B[:, k] = B[:, k - 1] * t

    return B.to(dtype=dtype, device=device)
