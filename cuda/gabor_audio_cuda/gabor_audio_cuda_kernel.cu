// Gabor splatting de audio en el dominio del tiempo (waveform directa).
//
// Cada atomo i representa una "rafaga" tiempo-frecuencia:
//
//   g_i(t) = A_i * exp(-(t - mu_i)^2 / (2 sigma_i^2)) * cos(2 pi f_i (t - mu_i) + phi_i)
//
// La senal reconstruida es la superposicion:
//
//   x_hat(t) = sum_i g_i(t)
//
// Parametros por atomo (5):
//   mu_i    : posicion temporal en samples
//   sigma_i : ancho de la envolvente gaussiana en samples (> 0)
//   amp_i   : amplitud (con signo)
//   fnorm_i : frecuencia normalizada en ciclos/sample, en (0, 0.5) por Nyquist
//   phi_i   : fase
//
// omega_i = 2 pi fnorm_i  (radianes/sample).
//
// Soporte local: cada atomo solo toca samples en [mu - k_sigma*sigma, mu + k_sigma*sigma].
//
// Estrategia:
//   forward  -> thread por atomo, atomicAdd a out[t] dentro de la ventana.
//   backward -> thread por atomo, acumula los 5 gradientes localmente (sin atomics).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TWO_PI 6.283185307179586f


static inline int div_up_int(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void gabor_forward_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ fnorm,
    const float* __restrict__ phi,
    float* __restrict__ out,
    const int N,
    const int T,
    const float k_sigma
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float mu_i = mu[i];
    const float sigma_i = fmaxf(sigma[i], 1e-6f);
    const float amp_i = amp[i];
    const float omega_i = TWO_PI * fnorm[i];
    const float phi_i = phi[i];

    const float half_width = k_sigma * sigma_i;
    int t0 = (int)floorf(mu_i - half_width);
    int t1 = (int)ceilf(mu_i + half_width);
    if (t0 < 0) t0 = 0;
    if (t1 > T - 1) t1 = T - 1;

    const float inv_2sig2 = 1.0f / (2.0f * sigma_i * sigma_i);

    for (int t = t0; t <= t1; ++t) {
        const float d = (float)t - mu_i;
        const float env = __expf(-d * d * inv_2sig2);
        const float arg = omega_i * d + phi_i;
        const float val = amp_i * env * __cosf(arg);
        atomicAdd(&out[t], val);
    }
}


__global__ void gabor_backward_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ sigma,
    const float* __restrict__ amp,
    const float* __restrict__ fnorm,
    const float* __restrict__ phi,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_sigma,
    float* __restrict__ grad_amp,
    float* __restrict__ grad_fnorm,
    float* __restrict__ grad_phi,
    const int N,
    const int T,
    const float k_sigma
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float mu_i = mu[i];
    const float sigma_i = fmaxf(sigma[i], 1e-6f);
    const float amp_i = amp[i];
    const float omega_i = TWO_PI * fnorm[i];
    const float phi_i = phi[i];

    const float half_width = k_sigma * sigma_i;
    int t0 = (int)floorf(mu_i - half_width);
    int t1 = (int)ceilf(mu_i + half_width);
    if (t0 < 0) t0 = 0;
    if (t1 > T - 1) t1 = T - 1;

    const float inv_sig2 = 1.0f / (sigma_i * sigma_i);
    const float inv_2sig2 = 0.5f * inv_sig2;
    const float inv_sig3 = inv_sig2 / sigma_i;   // 1 / sigma^3

    float g_mu = 0.0f;
    float g_sigma = 0.0f;
    float g_amp = 0.0f;
    float g_f = 0.0f;
    float g_phi = 0.0f;

    for (int t = t0; t <= t1; ++t) {
        const float go = grad_out[t];
        const float d = (float)t - mu_i;
        const float env = __expf(-d * d * inv_2sig2);
        const float arg = omega_i * d + phi_i;

        float s, c;
        __sincosf(arg, &s, &c);

        const float Aenv = amp_i * env;

        // dg/dA = env * cos
        g_amp += go * env * c;

        // dg/dmu = A env [ (d / sigma^2) cos + omega sin ]
        g_mu += go * Aenv * (d * inv_sig2 * c + omega_i * s);

        // dg/dsigma = A env (d^2 / sigma^3) cos
        g_sigma += go * Aenv * (d * d * inv_sig3) * c;

        // dg/dfnorm = A env (-sin) (2 pi d)
        g_f += go * Aenv * (-s) * (TWO_PI * d);

        // dg/dphi = A env (-sin)
        g_phi += go * Aenv * (-s);
    }

    grad_mu[i] = g_mu;
    grad_sigma[i] = g_sigma;
    grad_amp[i] = g_amp;
    grad_fnorm[i] = g_f;
    grad_phi[i] = g_phi;
}


torch::Tensor gabor_forward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    int T,
    float k_sigma
) {
    const int N = mu.size(0);
    auto out = torch::zeros({T}, mu.options());

    if (N == 0) {
        return out;
    }

    const int threads = 256;
    const int blocks = div_up_int(N, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gabor_forward_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        fnorm.data_ptr<float>(),
        phi.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        T,
        k_sigma
    );

    return out;
}


std::vector<torch::Tensor> gabor_backward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    torch::Tensor grad_out,
    float k_sigma
) {
    const int N = mu.size(0);
    const int T = grad_out.size(0);

    auto grad_mu = torch::zeros({N}, mu.options());
    auto grad_sigma = torch::zeros({N}, mu.options());
    auto grad_amp = torch::zeros({N}, mu.options());
    auto grad_fnorm = torch::zeros({N}, mu.options());
    auto grad_phi = torch::zeros({N}, mu.options());

    if (N == 0) {
        return {grad_mu, grad_sigma, grad_amp, grad_fnorm, grad_phi};
    }

    const int threads = 256;
    const int blocks = div_up_int(N, threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gabor_backward_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        sigma.data_ptr<float>(),
        amp.data_ptr<float>(),
        fnorm.data_ptr<float>(),
        phi.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_sigma.data_ptr<float>(),
        grad_amp.data_ptr<float>(),
        grad_fnorm.data_ptr<float>(),
        grad_phi.data_ptr<float>(),
        N,
        T,
        k_sigma
    );

    return {grad_mu, grad_sigma, grad_amp, grad_fnorm, grad_phi};
}
