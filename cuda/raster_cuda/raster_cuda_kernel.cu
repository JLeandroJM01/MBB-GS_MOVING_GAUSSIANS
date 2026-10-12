#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <climits>  

#ifndef RASTER_BATCH_SIZE
#define RASTER_BATCH_SIZE 256
#endif

// Shared-memory eval used by tiled kernels.
// This avoids reloading the same gaussian data for every pixel in a tile.
__device__ __forceinline__ float evaluar_alpha_values_device(
    float mu_f,
    float mu_c,
    float m00,
    float m01,
    float m11,
    float op,
    float pr,
    float pc,
    float* G_out,
    float* dx_out,
    float* dy_out,
    bool* unclamped_out
) {
    float dx = pr - mu_f;
    float dy = pc - mu_c;

    float quad = dx * (m00 * dx + m01 * dy) +
                 dy * (m01 * dx + m11 * dy);

    float G = expf(-0.5f * quad);
    float alpha_pre = op * G;
    bool unclamped = alpha_pre < 0.99f;

    float alpha = alpha_pre;
    if (alpha > 0.99f) {
        alpha = 0.99f;
    }

    if (G_out) {
        *G_out = G;
    }
    if (dx_out) {
        *dx_out = dx;
    }
    if (dy_out) {
        *dy_out = dy;
    }
    if (unclamped_out) {
        *unclamped_out = unclamped;
    }

    return alpha;
}



__global__ void build_conic_kernel(
    const float* __restrict__ scale,
    const float* __restrict__ theta,
    float* __restrict__ conic,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sx = scale[i * 2 + 0];
    float sy = scale[i * 2 + 1];
    float th = theta[i];

    float s, c;
    __sincosf(th, &s, &c);

    float inv_sx2 = 1.0f / (sx * sx + 1e-8f);
    float inv_sy2 = 1.0f / (sy * sy + 1e-8f);

    conic[i * 3 + 0] = c * c * inv_sx2 + s * s * inv_sy2;
    conic[i * 3 + 1] = c * s * (inv_sx2 - inv_sy2);
    conic[i * 3 + 2] = s * s * inv_sx2 + c * c * inv_sy2;
}


__global__ void grad_conic_to_scale_theta_kernel(
    const float* __restrict__ scale,
    const float* __restrict__ theta,
    const float* __restrict__ grad_conic,
    float* __restrict__ grad_scale,
    float* __restrict__ grad_theta,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sx = scale[i * 2 + 0];
    float sy = scale[i * 2 + 1];
    float th = theta[i];

    float s, c;
    __sincosf(th, &s, &c);

    float den_x = sx * sx + 1e-8f;
    float den_y = sy * sy + 1e-8f;

    float A = 1.0f / den_x;
    float B = 1.0f / den_y;

    float gm00 = grad_conic[i * 3 + 0];
    float gm01 = grad_conic[i * 3 + 1];
    float gm11 = grad_conic[i * 3 + 2];

    float grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s;
    float grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c;

    float dA_dsx = -2.0f * sx / (den_x * den_x);
    float dB_dsy = -2.0f * sy / (den_y * den_y);

    grad_scale[i * 2 + 0] = grad_A * dA_dsx;
    grad_scale[i * 2 + 1] = grad_B * dB_dsy;

    float dm00_dtheta = 2.0f * c * s * (B - A);
    float dm01_dtheta = (c * c - s * s) * (A - B);
    float dm11_dtheta = 2.0f * c * s * (A - B);

    grad_theta[i] = gm00 * dm00_dtheta +
                    gm01 * dm01_dtheta +
                    gm11 * dm11_dtheta;
}


__global__ void raster_forward_kernel(
    const float* mu,
    const float* scale,
    const float* theta,
    const float* opacity,
    const float* color,
    float* out,
    int N,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (idx >= total) {
        return;
    }

    int fila = idx / W;
    int col = idx % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;

    float T = 1.0f;

    for (int i = 0; i < N; i++) {
        float mu_f = mu[i * 2 + 0];
        float mu_c = mu[i * 2 + 1];

        float sx = scale[i * 2 + 0];
        float sy = scale[i * 2 + 1];

        float th = theta[i];

        float df = pr - mu_f;
        float dc = pc - mu_c;

        float c = cosf(th);
        float s = sinf(th);

        float inv_sx2 = 1.0f / (sx * sx + 1e-8f);
        float inv_sy2 = 1.0f / (sy * sy + 1e-8f);

        float m00 = c * c * inv_sx2 + s * s * inv_sy2;
        float m01 = c * s * (inv_sx2 - inv_sy2);
        float m11 = s * s * inv_sx2 + c * c * inv_sy2;

        float quad = df * (m00 * df + m01 * dc) +
                     dc * (m01 * df + m11 * dc);

        float G = expf(-0.5f * quad);

        float alpha = opacity[i] * G;

        if (alpha > 0.99f) {
            alpha = 0.99f;
        }

        float peso = alpha * T;

        r_final += peso * color[i * 3 + 0];
        g_final += peso * color[i * 3 + 1];
        b_final += peso * color[i * 3 + 2];

        T *= (1.0f - alpha);

        if (T < 1e-4f) {
            break;
        }
    }

    out[idx * 3 + 0] = r_final;
    out[idx * 3 + 1] = g_final;
    out[idx * 3 + 2] = b_final;
}

__global__ void raster_forward_conic_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    float* __restrict__ out,
    int N,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (idx >= total) {
        return;
    }

    int fila = idx / W;
    int col = idx % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;

    float T = 1.0f;

    for (int i = 0; i < N; i++) {
        float mu_f = mu[i * 2 + 0];
        float mu_c = mu[i * 2 + 1];

        float df = pr - mu_f;
        float dc = pc - mu_c;

        float m00 = conic[i * 3 + 0];
        float m01 = conic[i * 3 + 1];
        float m11 = conic[i * 3 + 2];

        float quad = df * (m00 * df + m01 * dc) +
                     dc * (m01 * df + m11 * dc);

        float G = expf(-0.5f * quad);

        float alpha = opacity[i] * G;

        if (alpha > 0.99f) {
            alpha = 0.99f;
        }

        float peso = alpha * T;

        r_final += peso * color[i * 3 + 0];
        g_final += peso * color[i * 3 + 1];
        b_final += peso * color[i * 3 + 2];

        T *= (1.0f - alpha);

        if (T < 1e-4f) {
            break;
        }
    }

    out[idx * 3 + 0] = r_final;
    out[idx * 3 + 1] = g_final;
    out[idx * 3 + 2] = b_final;
}


__global__ __launch_bounds__(256) void raster_forward_tiled_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int32_t* __restrict__ gaussian_ids,
    const int32_t* __restrict__ ranges,
    float* __restrict__ out,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int tile_id = blockIdx.x;
    int tid = threadIdx.x;

    int local_y = tid / tile_size;
    int local_x = tid % tile_size;

    int tile_y = tile_id / tiles_x;
    int tile_x = tile_id % tiles_x;

    int fila = tile_y * tile_size + local_y;
    int col = tile_x * tile_size + local_x;
    bool active = (fila < H && col < W);

    long long start = ranges[tile_id * 2 + 0];
    long long end = ranges[tile_id * 2 + 1];

    if (start < 0 || end <= start) {
        return;
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;
    float T = 1.0f;

    bool done = !active;

    __shared__ int s_gid[RASTER_BATCH_SIZE];
    __shared__ float s_mu_f[RASTER_BATCH_SIZE];
    __shared__ float s_mu_c[RASTER_BATCH_SIZE];
    __shared__ float s_m00[RASTER_BATCH_SIZE];
    __shared__ float s_m01[RASTER_BATCH_SIZE];
    __shared__ float s_m11[RASTER_BATCH_SIZE];
    __shared__ float s_opacity[RASTER_BATCH_SIZE];
    __shared__ float s_color_r[RASTER_BATCH_SIZE];
    __shared__ float s_color_g[RASTER_BATCH_SIZE];
    __shared__ float s_color_b[RASTER_BATCH_SIZE];

    const float T_MIN = 1.0e-4f;

    for (int batch_start = start; batch_start < end; batch_start += RASTER_BATCH_SIZE) {
        int remaining = (int)(end - batch_start);
        int batch_count = remaining < RASTER_BATCH_SIZE ? remaining : RASTER_BATCH_SIZE;

        // Threads cooperate to load gaussian data once per tile.
        for (int load_id = tid; load_id < batch_count; load_id += blockDim.x) {
            int gid = (int)gaussian_ids[batch_start + load_id];
            s_gid[load_id] = gid;
            s_mu_f[load_id] = mu[gid * 2 + 0];
            s_mu_c[load_id] = mu[gid * 2 + 1];
            s_m00[load_id] = conic[gid * 3 + 0];
            s_m01[load_id] = conic[gid * 3 + 1];
            s_m11[load_id] = conic[gid * 3 + 2];
            s_opacity[load_id] = opacity[gid];
            s_color_r[load_id] = color[gid * 3 + 0];
            s_color_g[load_id] = color[gid * 3 + 1];
            s_color_b[load_id] = color[gid * 3 + 2];
        }

        __syncthreads();

        if (!done) {
            #pragma unroll 1
            for (int k = 0; k < batch_count; k++) {
                float alpha = evaluar_alpha_values_device(
                    s_mu_f[k],
                    s_mu_c[k],
                    s_m00[k],
                    s_m01[k],
                    s_m11[k],
                    s_opacity[k],
                    pr,
                    pc,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr
                );

                float peso = alpha * T;

                r_final += peso * s_color_r[k];
                g_final += peso * s_color_g[k];
                b_final += peso * s_color_b[k];

                T *= (1.0f - alpha);

                if (T < T_MIN) {
                    done = true;
                    break;
                }
            }
        }

        // Same idea used in tile rasterizers: stop when every pixel is opaque enough.
        int done_count = __syncthreads_count(done);
        if (done_count == blockDim.x) {
            break;
        }
    }

    if (active) {
        int out_idx = (fila * W + col) * 3;
        out[out_idx + 0] = r_final;
        out[out_idx + 1] = g_final;
        out[out_idx + 2] = b_final;
    }
}


__device__ __forceinline__ float evaluar_alpha_conic_device(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    int i,
    float pr,
    float pc,
    float* G_out,
    float* dx_out,
    float* dy_out,
    float* power_out,
    bool* unclamped_out
) {
    float mu_f = mu[i * 2 + 0];
    float mu_c = mu[i * 2 + 1];

    float dx = pr - mu_f;
    float dy = pc - mu_c;

    float m00 = conic[i * 3 + 0];
    float m01 = conic[i * 3 + 1];
    float m11 = conic[i * 3 + 2];

    float quad = dx * (m00 * dx + m01 * dy) +
                 dy * (m01 * dx + m11 * dy);

    float power = -0.5f * quad;
    float G = expf(power);

    float alpha_pre = opacity[i] * G;
    bool unclamped = alpha_pre < 0.99f;

    float alpha = alpha_pre;
    if (alpha > 0.99f) {
        alpha = 0.99f;
    }

    if (G_out) {
        *G_out = G;
    }
    if (dx_out) {
        *dx_out = dx;
    }
    if (dy_out) {
        *dy_out = dy;
    }
    if (power_out) {
        *power_out = power;
    }
    if (unclamped_out) {
        *unclamped_out = unclamped;
    }

    return alpha;
}


__global__ void raster_backward_conic_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_color,
    int N,
    int H,
    int W
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (pix >= total) {
        return;
    }

    int fila = pix / W;
    int col = pix % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float go_r = grad_out[pix * 3 + 0];
    float go_g = grad_out[pix * 3 + 1];
    float go_b = grad_out[pix * 3 + 2];

    // Backward brute-force v1:
    // Para cada gaussiana i:
    //   1. Recomputa T_i con un loop 0..i-1
    //   2. Recomputa el color de lo que viene despues de i
    //   3. Calcula dL/dalpha_i
    //   4. Propaga a color, opacity, mu y conic
    //
    // Esto es O(P*N^2). Es para validar gradientes, no para performance final.

    for (int i = 0; i < N; i++) {
        // -------- prefix T_i --------
        float T_i = 1.0f;

        for (int j = 0; j < i; j++) {
            float alpha_j = evaluar_alpha_conic_device(
                mu, conic, opacity, j, pr, pc,
                nullptr, nullptr, nullptr, nullptr, nullptr
            );

            T_i *= (1.0f - alpha_j);

            if (T_i < 1e-4f) {
                break;
            }
        }

        if (T_i < 1e-4f) {
            continue;
        }

        // -------- alpha_i --------
        float G_i = 0.0f;
        float dx_i = 0.0f;
        float dy_i = 0.0f;
        float power_i = 0.0f;
        bool unclamped_i = true;

        float alpha_i = evaluar_alpha_conic_device(
            mu, conic, opacity, i, pr, pc,
            &G_i, &dx_i, &dy_i, &power_i, &unclamped_i
        );

        // -------- tail color despues de i --------
        float tail_r = 0.0f;
        float tail_g = 0.0f;
        float tail_b = 0.0f;
        float T_tail = 1.0f;

        for (int k = i + 1; k < N; k++) {
            float alpha_k = evaluar_alpha_conic_device(
                mu, conic, opacity, k, pr, pc,
                nullptr, nullptr, nullptr, nullptr, nullptr
            );

            float peso_k = T_tail * alpha_k;

            tail_r += peso_k * color[k * 3 + 0];
            tail_g += peso_k * color[k * 3 + 1];
            tail_b += peso_k * color[k * 3 + 2];

            T_tail *= (1.0f - alpha_k);

            if (T_tail < 1e-4f) {
                break;
            }
        }

        // dC/dalpha_i = T_i * (color_i - tail)
        float dC_da_r = T_i * (color[i * 3 + 0] - tail_r);
        float dC_da_g = T_i * (color[i * 3 + 1] - tail_g);
        float dC_da_b = T_i * (color[i * 3 + 2] - tail_b);

        float dL_dalpha = go_r * dC_da_r + go_g * dC_da_g + go_b * dC_da_b;

        // grad color_i = grad_out * T_i * alpha_i
        float peso_i = T_i * alpha_i;

        atomicAdd(&grad_color[i * 3 + 0], go_r * peso_i);
        atomicAdd(&grad_color[i * 3 + 1], go_g * peso_i);
        atomicAdd(&grad_color[i * 3 + 2], go_b * peso_i);

        // Si alpha fue clamp a 0.99, la derivada del clamp es 0.
        if (!unclamped_i) {
            continue;
        }

        // alpha = opacity * G
        float dL_dopacity = dL_dalpha * G_i;
        atomicAdd(&grad_opacity[i], dL_dopacity);

        float dL_dG = dL_dalpha * opacity[i];

        // G = exp(power)
        float dL_dpower = dL_dG * G_i;

        float m00 = conic[i * 3 + 0];
        float m01 = conic[i * 3 + 1];
        float m11 = conic[i * 3 + 2];

        // power = -0.5 * (m00*dx^2 + 2*m01*dx*dy + m11*dy^2)
        // dx = pixel_fila - mu_fila
        // dy = pixel_col  - mu_col

        float dpower_dmu_f = m00 * dx_i + m01 * dy_i;
        float dpower_dmu_c = m01 * dx_i + m11 * dy_i;

        atomicAdd(&grad_mu[i * 2 + 0], dL_dpower * dpower_dmu_f);
        atomicAdd(&grad_mu[i * 2 + 1], dL_dpower * dpower_dmu_c);

        float dpower_dm00 = -0.5f * dx_i * dx_i;
        float dpower_dm01 = -1.0f * dx_i * dy_i;
        float dpower_dm11 = -0.5f * dy_i * dy_i;

        atomicAdd(&grad_conic[i * 3 + 0], dL_dpower * dpower_dm00);
        atomicAdd(&grad_conic[i * 3 + 1], dL_dpower * dpower_dm01);
        atomicAdd(&grad_conic[i * 3 + 2], dL_dpower * dpower_dm11);
    }
}

__global__ void raster_backward_tiled_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int32_t* __restrict__ gaussian_ids,
    const int32_t* __restrict__ ranges,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_color,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (pix >= total) {
        return;
    }

    int fila = pix / W;
    int col = pix % W;

    int tile_x = col / tile_size;
    int tile_y = fila / tile_size;
    int tile_id = tile_y * tiles_x + tile_x;

    int start = ranges[tile_id * 2 + 0];
    int end = ranges[tile_id * 2 + 1];

    if (start < 0 || end <= start) {
        return;
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float go_r = grad_out[pix * 3 + 0];
    float go_g = grad_out[pix * 3 + 1];
    float go_b = grad_out[pix * 3 + 2];

    // Version v1:
    // Para cada gaussiana del tile:
    //   - recomputa T_i con las gaussianas anteriores del mismo tile
    //   - recomputa tail con las gaussianas posteriores del mismo tile
    //   - acumula gradientes con atomicAdd
    //
    // Esto valida el backward tiled. No es la version final optimizada.

    for (int ii = start; ii < end; ii++) {
        int gid_i = (int)gaussian_ids[ii];

        float T_i = 1.0f;

        for (int jj = start; jj < ii; jj++) {
            int gid_j = (int)gaussian_ids[jj];

            float alpha_j = evaluar_alpha_conic_device(
                mu,
                conic,
                opacity,
                gid_j,
                pr,
                pc,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );

            T_i *= (1.0f - alpha_j);

            if (T_i < 1e-4f) {
                break;
            }
        }

        if (T_i < 1e-4f) {
            continue;
        }

        float G_i = 0.0f;
        float dx_i = 0.0f;
        float dy_i = 0.0f;
        float power_i = 0.0f;
        bool unclamped_i = true;

        float alpha_i = evaluar_alpha_conic_device(
            mu,
            conic,
            opacity,
            gid_i,
            pr,
            pc,
            &G_i,
            &dx_i,
            &dy_i,
            &power_i,
            &unclamped_i
        );

        float tail_r = 0.0f;
        float tail_g = 0.0f;
        float tail_b = 0.0f;
        float T_tail = 1.0f;

        for (int kk = ii + 1; kk < end; kk++) {
            int gid_k = (int)gaussian_ids[kk];

            float alpha_k = evaluar_alpha_conic_device(
                mu,
                conic,
                opacity,
                gid_k,
                pr,
                pc,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );

            float peso_k = T_tail * alpha_k;

            tail_r += peso_k * color[gid_k * 3 + 0];
            tail_g += peso_k * color[gid_k * 3 + 1];
            tail_b += peso_k * color[gid_k * 3 + 2];

            T_tail *= (1.0f - alpha_k);

            if (T_tail < 1e-4f) {
                break;
            }
        }

        float ci_r = color[gid_i * 3 + 0];
        float ci_g = color[gid_i * 3 + 1];
        float ci_b = color[gid_i * 3 + 2];

        float dC_da_r = T_i * (ci_r - tail_r);
        float dC_da_g = T_i * (ci_g - tail_g);
        float dC_da_b = T_i * (ci_b - tail_b);

        float dL_dalpha = go_r * dC_da_r + go_g * dC_da_g + go_b * dC_da_b;

        float peso_i = T_i * alpha_i;

        atomicAdd(&grad_color[gid_i * 3 + 0], go_r * peso_i);
        atomicAdd(&grad_color[gid_i * 3 + 1], go_g * peso_i);
        atomicAdd(&grad_color[gid_i * 3 + 2], go_b * peso_i);

        if (!unclamped_i) {
            continue;
        }

        float dL_dopacity = dL_dalpha * G_i;
        atomicAdd(&grad_opacity[gid_i], dL_dopacity);

        float dL_dG = dL_dalpha * opacity[gid_i];
        float dL_dpower = dL_dG * G_i;

        float m00 = conic[gid_i * 3 + 0];
        float m01 = conic[gid_i * 3 + 1];
        float m11 = conic[gid_i * 3 + 2];

        float dpower_dmu_f = m00 * dx_i + m01 * dy_i;
        float dpower_dmu_c = m01 * dx_i + m11 * dy_i;

        atomicAdd(&grad_mu[gid_i * 2 + 0], dL_dpower * dpower_dmu_f);
        atomicAdd(&grad_mu[gid_i * 2 + 1], dL_dpower * dpower_dmu_c);

        float dpower_dm00 = -0.5f * dx_i * dx_i;
        float dpower_dm01 = -1.0f * dx_i * dy_i;
        float dpower_dm11 = -0.5f * dy_i * dy_i;

        atomicAdd(&grad_conic[gid_i * 3 + 0], dL_dpower * dpower_dm00);
        atomicAdd(&grad_conic[gid_i * 3 + 1], dL_dpower * dpower_dm01);
        atomicAdd(&grad_conic[gid_i * 3 + 2], dL_dpower * dpower_dm11);
    }
}




__global__ __launch_bounds__(256) void raster_forward_tiled_train_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int32_t* __restrict__ gaussian_ids,
    const int32_t* __restrict__ ranges,
    float* __restrict__ out,
    float* __restrict__ final_Ts,
    int32_t* __restrict__ n_contrib,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int tile_id = blockIdx.x;
    int tid = threadIdx.x;

    int local_y = tid / tile_size;
    int local_x = tid % tile_size;

    int tile_x = tile_id % tiles_x;
    int tile_y = tile_id / tiles_x;

    int fila = tile_y * tile_size + local_y;
    int col = tile_x * tile_size + local_x;
    bool active = (fila < H && col < W);

    int pix = fila * W + col;

    int start = ranges[tile_id * 2 + 0];
    int end = ranges[tile_id * 2 + 1];

    if (start < 0 || end <= start) {
        if (active) {
            out[pix * 3 + 0] = 0.0f;
            out[pix * 3 + 1] = 0.0f;
            out[pix * 3 + 2] = 0.0f;
            final_Ts[pix] = 1.0f;
            n_contrib[pix] = 0;
        }
        return;
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float T = 1.0f;
    float acc_r = 0.0f;
    float acc_g = 0.0f;
    float acc_b = 0.0f;

    int32_t processed = 0;
    bool done = !active;

    __shared__ int s_gid[RASTER_BATCH_SIZE];
    __shared__ float s_mu_f[RASTER_BATCH_SIZE];
    __shared__ float s_mu_c[RASTER_BATCH_SIZE];
    __shared__ float s_m00[RASTER_BATCH_SIZE];
    __shared__ float s_m01[RASTER_BATCH_SIZE];
    __shared__ float s_m11[RASTER_BATCH_SIZE];
    __shared__ float s_opacity[RASTER_BATCH_SIZE];
    __shared__ float s_color_r[RASTER_BATCH_SIZE];
    __shared__ float s_color_g[RASTER_BATCH_SIZE];
    __shared__ float s_color_b[RASTER_BATCH_SIZE];

    const float EPS_ALPHA = 1.0f / 255.0f;
    const float T_MIN = 1.0e-4f;

    for (int batch_start = start; batch_start < end; batch_start += RASTER_BATCH_SIZE) {
        int remaining = (int)(end - batch_start);
        int batch_count = remaining < RASTER_BATCH_SIZE ? remaining : RASTER_BATCH_SIZE;

        // Threads cooperate to load gaussian data once per tile.
        for (int load_id = tid; load_id < batch_count; load_id += blockDim.x) {
            int gid = (int)gaussian_ids[batch_start + load_id];
            s_gid[load_id] = gid;
            s_mu_f[load_id] = mu[gid * 2 + 0];
            s_mu_c[load_id] = mu[gid * 2 + 1];
            s_m00[load_id] = conic[gid * 3 + 0];
            s_m01[load_id] = conic[gid * 3 + 1];
            s_m11[load_id] = conic[gid * 3 + 2];
            s_opacity[load_id] = opacity[gid];
            s_color_r[load_id] = color[gid * 3 + 0];
            s_color_g[load_id] = color[gid * 3 + 1];
            s_color_b[load_id] = color[gid * 3 + 2];
        }

        __syncthreads();

        if (!done) {
            #pragma unroll 1
            for (int k = 0; k < batch_count; k++) {
                int global_pos = batch_start + k;
                processed = global_pos - start + 1;

                float alpha = evaluar_alpha_values_device(
                    s_mu_f[k],
                    s_mu_c[k],
                    s_m00[k],
                    s_m01[k],
                    s_m11[k],
                    s_opacity[k],
                    pr,
                    pc,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr
                );

                if (alpha < EPS_ALPHA) {
                    continue;
                }

                float peso = T * alpha;

                acc_r += peso * s_color_r[k];
                acc_g += peso * s_color_g[k];
                acc_b += peso * s_color_b[k];

                T *= (1.0f - alpha);

                if (T < T_MIN) {
                    done = true;
                    break;
                }
            }
        }

        // Same idea used in tile rasterizers: stop when every pixel is opaque enough.
        int done_count = __syncthreads_count(done);
        if (done_count == blockDim.x) {
            break;
        }
    }

    if (active) {
        out[pix * 3 + 0] = acc_r;
        out[pix * 3 + 1] = acc_g;
        out[pix * 3 + 2] = acc_b;

        final_Ts[pix] = T;
        n_contrib[pix] = processed;
    }
}


// Fused training forward + loss.
// Same math as raster_forward_tiled_train_kernel + loss_grad_kernel, but avoids
// writing/reading a temporary render tensor during training loss.
__global__ __launch_bounds__(256) void raster_forward_tiled_train_loss_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int32_t* __restrict__ gaussian_ids,
    const int32_t* __restrict__ ranges,
    const float* __restrict__ target,
    float* __restrict__ loss,
    float* __restrict__ grad_render,
    float* __restrict__ final_Ts,
    int32_t* __restrict__ n_contrib,
    int H,
    int W,
    int tile_size,
    int tiles_x,
    int loss_type
) {
    int tile_id = blockIdx.x;
    int tid = threadIdx.x;

    int local_y = tid / tile_size;
    int local_x = tid % tile_size;

    int tile_x = tile_id % tiles_x;
    int tile_y = tile_id / tiles_x;

    int fila = tile_y * tile_size + local_y;
    int col = tile_x * tile_size + local_x;
    bool active = (fila < H && col < W);
    int pix = fila * W + col;

    int start = ranges[tile_id * 2 + 0];
    int end = ranges[tile_id * 2 + 1];

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float T = 1.0f;
    float acc_r = 0.0f;
    float acc_g = 0.0f;
    float acc_b = 0.0f;
    int32_t processed = 0;
    bool done = !active;

    __shared__ int s_gid[RASTER_BATCH_SIZE];
    __shared__ float s_mu_f[RASTER_BATCH_SIZE];
    __shared__ float s_mu_c[RASTER_BATCH_SIZE];
    __shared__ float s_m00[RASTER_BATCH_SIZE];
    __shared__ float s_m01[RASTER_BATCH_SIZE];
    __shared__ float s_m11[RASTER_BATCH_SIZE];
    __shared__ float s_opacity[RASTER_BATCH_SIZE];
    __shared__ float s_color_r[RASTER_BATCH_SIZE];
    __shared__ float s_color_g[RASTER_BATCH_SIZE];
    __shared__ float s_color_b[RASTER_BATCH_SIZE];

    const float EPS_ALPHA = 1.0f / 255.0f;
    const float T_MIN = 1.0e-4f;

    if (start >= 0 && end > start) {
        for (int batch_start = start; batch_start < end; batch_start += RASTER_BATCH_SIZE) {
            int remaining = (int)(end - batch_start);
            int batch_count = remaining < RASTER_BATCH_SIZE ? remaining : RASTER_BATCH_SIZE;

            for (int load_id = tid; load_id < batch_count; load_id += blockDim.x) {
                int gid = (int)gaussian_ids[batch_start + load_id];
                s_gid[load_id] = gid;
                s_mu_f[load_id] = mu[gid * 2 + 0];
                s_mu_c[load_id] = mu[gid * 2 + 1];
                s_m00[load_id] = conic[gid * 3 + 0];
                s_m01[load_id] = conic[gid * 3 + 1];
                s_m11[load_id] = conic[gid * 3 + 2];
                s_opacity[load_id] = opacity[gid];
                s_color_r[load_id] = color[gid * 3 + 0];
                s_color_g[load_id] = color[gid * 3 + 1];
                s_color_b[load_id] = color[gid * 3 + 2];
            }

            __syncthreads();

            if (!done) {
                #pragma unroll 1
                for (int k = 0; k < batch_count; k++) {
                    int global_pos = batch_start + k;
                    processed = global_pos - start + 1;

                    float alpha = evaluar_alpha_values_device(
                        s_mu_f[k], s_mu_c[k], s_m00[k], s_m01[k], s_m11[k], s_opacity[k],
                        pr, pc, nullptr, nullptr, nullptr, nullptr
                    );

                    if (alpha < EPS_ALPHA) continue;

                    float peso = T * alpha;
                    acc_r += peso * s_color_r[k];
                    acc_g += peso * s_color_g[k];
                    acc_b += peso * s_color_b[k];

                    T *= (1.0f - alpha);
                    if (T < T_MIN) {
                        done = true;
                        break;
                    }
                }
            }

            int done_count = __syncthreads_count(done);
            if (done_count == blockDim.x) break;
        }
    }

    if (active) {
        final_Ts[pix] = T;
        n_contrib[pix] = processed;

        const int pix3 = pix * 3;
        const float inv_total = 1.0f / (float)(H * W * 3);
        float dr = acc_r - target[pix3 + 0];
        float dg = acc_g - target[pix3 + 1];
        float db = acc_b - target[pix3 + 2];

        if (loss_type == 0) {
            atomicAdd(loss, fabsf(dr) * inv_total);
            atomicAdd(loss, fabsf(dg) * inv_total);
            atomicAdd(loss, fabsf(db) * inv_total);
            grad_render[pix3 + 0] = (dr > 0.0f) ? inv_total : ((dr < 0.0f) ? -inv_total : 0.0f);
            grad_render[pix3 + 1] = (dg > 0.0f) ? inv_total : ((dg < 0.0f) ? -inv_total : 0.0f);
            grad_render[pix3 + 2] = (db > 0.0f) ? inv_total : ((db < 0.0f) ? -inv_total : 0.0f);
        } else {
            atomicAdd(loss, (dr * dr + dg * dg + db * db) * inv_total);
            grad_render[pix3 + 0] = 2.0f * dr * inv_total;
            grad_render[pix3 + 1] = 2.0f * dg * inv_total;
            grad_render[pix3 + 2] = 2.0f * db * inv_total;
        }
    }
}


__global__ __launch_bounds__(256) void raster_backward_tiled_fast_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int32_t* __restrict__ gaussian_ids,
    const int32_t* __restrict__ ranges,
    const float* __restrict__ final_Ts,
    const int32_t* __restrict__ n_contrib,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_color,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int tile_id = blockIdx.x;
    int tid = threadIdx.x;

    int local_y = tid / tile_size;
    int local_x = tid % tile_size;

    int tile_x = tile_id % tiles_x;
    int tile_y = tile_id / tiles_x;

    int fila = tile_y * tile_size + local_y;
    int col = tile_x * tile_size + local_x;
    bool active = (fila < H && col < W);

    int pix = fila * W + col;

    int start = ranges[tile_id * 2 + 0];
    int end = ranges[tile_id * 2 + 1];

    if (start < 0 || end <= start) {
        return;
    }

    int32_t processed = 0;
    int last = start - 1;

    if (active) {
        processed = n_contrib[pix];
        if (processed > 0) {
            last = start + processed - 1;
            if (last >= end) {
                last = end - 1;
            }
        }
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float go_r = active ? grad_out[pix * 3 + 0] : 0.0f;
    float go_g = active ? grad_out[pix * 3 + 1] : 0.0f;
    float go_b = active ? grad_out[pix * 3 + 2] : 0.0f;

    float T_after = active ? final_Ts[pix] : 1.0f;

    float tail_r = 0.0f;
    float tail_g = 0.0f;
    float tail_b = 0.0f;

    __shared__ int s_gid[RASTER_BATCH_SIZE];
    __shared__ float s_mu_f[RASTER_BATCH_SIZE];
    __shared__ float s_mu_c[RASTER_BATCH_SIZE];
    __shared__ float s_m00[RASTER_BATCH_SIZE];
    __shared__ float s_m01[RASTER_BATCH_SIZE];
    __shared__ float s_m11[RASTER_BATCH_SIZE];
    __shared__ float s_opacity[RASTER_BATCH_SIZE];
    __shared__ float s_color_r[RASTER_BATCH_SIZE];
    __shared__ float s_color_g[RASTER_BATCH_SIZE];
    __shared__ float s_color_b[RASTER_BATCH_SIZE];

    const float EPS_ALPHA = 1.0f / 255.0f;

    // Iterate over the tile list back-to-front in shared-memory batches.
    for (int batch_end = end; batch_end > start; ) {
        int remaining = batch_end - start;
        int batch_count = remaining < RASTER_BATCH_SIZE ? remaining : RASTER_BATCH_SIZE;
        int batch_start = batch_end - batch_count;

        // If no pixel in the tile needs this high-depth batch, skip it.
        bool needs_batch = active && (processed > 0) && (last >= batch_start);
        int needs_count = __syncthreads_count(needs_batch);

        if (needs_count > 0) {
            for (int load_id = tid; load_id < batch_count; load_id += blockDim.x) {
                int gid = (int)gaussian_ids[batch_start + load_id];
                s_gid[load_id] = gid;
                s_mu_f[load_id] = mu[gid * 2 + 0];
                s_mu_c[load_id] = mu[gid * 2 + 1];
                s_m00[load_id] = conic[gid * 3 + 0];
                s_m01[load_id] = conic[gid * 3 + 1];
                s_m11[load_id] = conic[gid * 3 + 2];
                s_opacity[load_id] = opacity[gid];
                s_color_r[load_id] = color[gid * 3 + 0];
                s_color_g[load_id] = color[gid * 3 + 1];
                s_color_b[load_id] = color[gid * 3 + 2];
            }

            __syncthreads();

            if (active && processed > 0) {
                #pragma unroll 1
                for (int k = batch_count - 1; k >= 0; k--) {
                    int global_pos = batch_start + k;
                    if (global_pos > last) {
                        continue;
                    }

                    int gid = s_gid[k];

                    float G = 0.0f;
                    float dx = 0.0f;
                    float dy = 0.0f;
                    bool unclamped = true;

                    float alpha = evaluar_alpha_values_device(
                        s_mu_f[k],
                        s_mu_c[k],
                        s_m00[k],
                        s_m01[k],
                        s_m11[k],
                        s_opacity[k],
                        pr,
                        pc,
                        &G,
                        &dx,
                        &dy,
                        &unclamped
                    );

                    if (alpha < EPS_ALPHA) {
                        continue;
                    }

                    float one_minus_alpha = 1.0f - alpha;
                    one_minus_alpha = fmaxf(one_minus_alpha, 1.0e-6f);

                    // Rebuild front-to-back transmitance while walking backward.
                    float T_i = T_after / one_minus_alpha;

                    float ci_r = s_color_r[k];
                    float ci_g = s_color_g[k];
                    float ci_b = s_color_b[k];

                    float dC_da_r = T_i * (ci_r - tail_r);
                    float dC_da_g = T_i * (ci_g - tail_g);
                    float dC_da_b = T_i * (ci_b - tail_b);

                    float dL_dalpha =
                        go_r * dC_da_r +
                        go_g * dC_da_g +
                        go_b * dC_da_b;

                    float peso = T_i * alpha;

                    atomicAdd(&grad_color[gid * 3 + 0], go_r * peso);
                    atomicAdd(&grad_color[gid * 3 + 1], go_g * peso);
                    atomicAdd(&grad_color[gid * 3 + 2], go_b * peso);

                    if (unclamped) {
                        float dL_dopacity = dL_dalpha * G;
                        atomicAdd(&grad_opacity[gid], dL_dopacity);

                        float dL_dG = dL_dalpha * s_opacity[k];
                        float dL_dpower = dL_dG * G;

                        float m00 = s_m00[k];
                        float m01 = s_m01[k];
                        float m11 = s_m11[k];

                        float dpower_dmu_f = m00 * dx + m01 * dy;
                        float dpower_dmu_c = m01 * dx + m11 * dy;

                        atomicAdd(&grad_mu[gid * 2 + 0], dL_dpower * dpower_dmu_f);
                        atomicAdd(&grad_mu[gid * 2 + 1], dL_dpower * dpower_dmu_c);

                        float dpower_dm00 = -0.5f * dx * dx;
                        float dpower_dm01 = -1.0f * dx * dy;
                        float dpower_dm11 = -0.5f * dy * dy;

                        atomicAdd(&grad_conic[gid * 3 + 0], dL_dpower * dpower_dm00);
                        atomicAdd(&grad_conic[gid * 3 + 1], dL_dpower * dpower_dm01);
                        atomicAdd(&grad_conic[gid * 3 + 2], dL_dpower * dpower_dm11);
                    }

                    // Update the already-composited tail.
                    tail_r = alpha * ci_r + one_minus_alpha * tail_r;
                    tail_g = alpha * ci_g + one_minus_alpha * tail_g;
                    tail_b = alpha * ci_b + one_minus_alpha * tail_b;

                    T_after = T_i;
                }
            }

            __syncthreads();
        }

        batch_end = batch_start;
    }
}


torch::Tensor raster_build_conic_cuda(
    torch::Tensor scale,
    torch::Tensor theta
) {
    int N = scale.size(0);
    auto conic = torch::empty(
        {N, 3},
        torch::TensorOptions()
            .device(scale.device())
            .dtype(torch::kFloat32)
    );

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_conic_kernel<<<blocks, threads, 0, stream>>>(
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        conic.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return conic;
}


std::vector<torch::Tensor> raster_grad_conic_to_scale_theta_cuda(
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor grad_conic
) {
    int N = scale.size(0);

    auto grad_scale = torch::empty_like(scale);
    auto grad_theta = torch::empty_like(theta);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    grad_conic_to_scale_theta_kernel<<<blocks, threads, 0, stream>>>(
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_scale.data_ptr<float>(),
        grad_theta.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_scale, grad_theta};
}


torch::Tensor raster_forward_cuda(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    int N = mu.size(0);

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_forward_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}



torch::Tensor raster_forward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    int N = mu.size(0);

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_forward_conic_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}


torch::Tensor raster_forward_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    int H,
    int W,
    int tile_size
) {
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int threads = tile_size * tile_size;

    raster_forward_tiled_kernel<<<total_tiles, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}


std::vector<torch::Tensor> raster_backward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor grad_out,
    int H,
    int W
) {
    int N = mu.size(0);

    auto grad_mu = torch::zeros_like(mu);
    auto grad_conic = torch::zeros_like(conic);
    auto grad_opacity = torch::zeros_like(opacity);
    auto grad_color = torch::zeros_like(color);

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_backward_conic_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_mu, grad_conic, grad_opacity, grad_color};
}


std::vector<torch::Tensor> raster_backward_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    torch::Tensor grad_out,
    int H,
    int W,
    int tile_size
) {
    auto grad_mu = torch::zeros_like(mu);
    auto grad_conic = torch::zeros_like(conic);
    auto grad_opacity = torch::zeros_like(opacity);
    auto grad_color = torch::zeros_like(color);

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    int tiles_x = (W + tile_size - 1) / tile_size;

    raster_backward_tiled_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_mu, grad_conic, grad_opacity, grad_color};
}

__global__ __launch_bounds__(256) void loss_grad_kernel(
    const float* __restrict__ render,
    const float* __restrict__ target,
    float* __restrict__ loss,
    float* __restrict__ grad_render,
    int total,
    int loss_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float diff = render[idx] - target[idx];
    float inv_total = 1.0f / (float)total;

    if (loss_type == 0) {
        // L1 mean
        float abs_diff = fabsf(diff);
        atomicAdd(loss, abs_diff * inv_total);

        float g = 0.0f;
        if (diff > 0.0f) g = inv_total;
        else if (diff < 0.0f) g = -inv_total;

        grad_render[idx] = g;
    } else {
        // MSE mean
        atomicAdd(loss, diff * diff * inv_total);
        grad_render[idx] = 2.0f * diff * inv_total;
    }
}


std::vector<torch::Tensor> raster_forward_tiled_train_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    int H,
    int W,
    int tile_size
) {
    auto options_f = mu.options();
    auto options_i = torch::TensorOptions().device(mu.device()).dtype(torch::kInt32);

    auto out = torch::zeros({H, W, 3}, options_f);
    auto final_Ts = torch::ones({H, W}, options_f);
    auto n_contrib = torch::zeros({H, W}, options_i);

    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    int threads = tile_size * tile_size;

    raster_forward_tiled_train_kernel<<<total_tiles, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        final_Ts.data_ptr<float>(),
        n_contrib.data_ptr<int32_t>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {out, final_Ts, n_contrib};
}

std::vector<torch::Tensor> raster_forward_tiled_train_loss_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    torch::Tensor target,
    int H,
    int W,
    int tile_size,
    int loss_type
) {
    auto options_f = mu.options();
    auto options_i = torch::TensorOptions().device(mu.device()).dtype(torch::kInt32);

    auto loss = torch::zeros({}, options_f);
    auto grad_render = torch::empty({H, W, 3}, options_f);
    auto final_Ts = torch::ones({H, W}, options_f);
    auto n_contrib = torch::zeros({H, W}, options_i);

    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;
    int threads = tile_size * tile_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    raster_forward_tiled_train_loss_kernel<<<total_tiles, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        target.data_ptr<float>(),
        loss.data_ptr<float>(),
        grad_render.data_ptr<float>(),
        final_Ts.data_ptr<float>(),
        n_contrib.data_ptr<int32_t>(),
        H,
        W,
        tile_size,
        tiles_x,
        loss_type
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {loss, grad_render, final_Ts, n_contrib};
}


std::vector<torch::Tensor> raster_backward_tiled_fast_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    torch::Tensor final_Ts,
    torch::Tensor n_contrib,
    torch::Tensor grad_out,
    int H,
    int W,
    int tile_size
) {
    auto grad_mu = torch::zeros_like(mu);
    auto grad_conic = torch::zeros_like(conic);
    auto grad_opacity = torch::zeros_like(opacity);
    auto grad_color = torch::zeros_like(color);

    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    int threads = tile_size * tile_size;

    raster_backward_tiled_fast_kernel<<<total_tiles, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        final_Ts.data_ptr<float>(),
        n_contrib.data_ptr<int32_t>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_mu, grad_conic, grad_opacity, grad_color};
}

// ============================================================================
// Preprocess tiled CUDA
// ============================================================================

__global__ __launch_bounds__(256) void compute_tile_counts_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ scale,
    const float* __restrict__ theta,
    int* __restrict__ counts,
    int N,
    int H,
    int W,
    int tile_size,
    float k_sigma,
    int tiles_x,
    int tiles_y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float fila = mu[i * 2 + 0];
    float col = mu[i * 2 + 1];
    float sx = scale[i * 2 + 0];
    float sy = scale[i * 2 + 1];
    float th = theta[i];

    float s, c;
    __sincosf(th, &s, &c);

    // AABB conservador para una gaussiana rotada.
    float extent_fila = k_sigma * sqrtf((c * sx) * (c * sx) + (s * sy) * (s * sy));
    float extent_col  = k_sigma * sqrtf((s * sx) * (s * sx) + (c * sy) * (c * sy));

    int min_f = (int)floorf(fila - extent_fila);
    int max_f = (int)ceilf (fila + extent_fila);
    int min_c = (int)floorf(col  - extent_col);
    int max_c = (int)ceilf (col  + extent_col);

    // Fase 1: culling temprano.
    // Si la AABB completa esta fuera de la imagen, no debe tocar ningun tile.
    // Antes se hacia clamp y una gaussiana fuera de pantalla podia caer en un tile de borde.
    if (max_f < 0 || min_f >= H || max_c < 0 || min_c >= W) {
        counts[i] = 0;
        return;
    }

    min_f = max(0, min(H - 1, min_f));
    max_f = max(0, min(H - 1, max_f));
    min_c = max(0, min(W - 1, min_c));
    max_c = max(0, min(W - 1, max_c));

    int tile_y_min = min_f / tile_size;
    int tile_y_max = max_f / tile_size;
    int tile_x_min = min_c / tile_size;
    int tile_x_max = max_c / tile_size;

    tile_y_min = max(0, min(tiles_y - 1, tile_y_min));
    tile_y_max = max(0, min(tiles_y - 1, tile_y_max));
    tile_x_min = max(0, min(tiles_x - 1, tile_x_min));
    tile_x_max = max(0, min(tiles_x - 1, tile_x_max));

    int num_y = tile_y_max - tile_y_min + 1;
    int num_x = tile_x_max - tile_x_min + 1;
    counts[i] = num_y * num_x;
}

__global__ __launch_bounds__(256) void duplicate_with_keys_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ scale,
    const float* __restrict__ theta,
    const float* __restrict__ depth,
    const float* __restrict__ depth_min,
    const float* __restrict__ depth_max,
    const int* __restrict__ offsets,
    int32_t* __restrict__ keys,
    int32_t* __restrict__ gaussian_ids,
    int N,
    int H,
    int W,
    int tile_size,
    float k_sigma,
    int tiles_x,
    int tiles_y
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float fila = mu[i * 2 + 0];
    float col = mu[i * 2 + 1];
    float sx = scale[i * 2 + 0];
    float sy = scale[i * 2 + 1];
    float th = theta[i];

    float s, c;
    __sincosf(th, &s, &c);

    float extent_fila = k_sigma * sqrtf((c * sx) * (c * sx) + (s * sy) * (s * sy));
    float extent_col  = k_sigma * sqrtf((s * sx) * (s * sx) + (c * sy) * (c * sy));

    int min_f = (int)floorf(fila - extent_fila);
    int max_f = (int)ceilf (fila + extent_fila);
    int min_c = (int)floorf(col  - extent_col);
    int max_c = (int)ceilf (col  + extent_col);

    // Fase 1: culling temprano tambien en duplicacion.
    // Debe coincidir con compute_tile_counts_kernel para no escribir fuera del arreglo.
    if (max_f < 0 || min_f >= H || max_c < 0 || min_c >= W) {
        return;
    }

    min_f = max(0, min(H - 1, min_f));
    max_f = max(0, min(H - 1, max_f));
    min_c = max(0, min(W - 1, min_c));
    max_c = max(0, min(W - 1, max_c));

    int tile_y_min = min_f / tile_size;
    int tile_y_max = max_f / tile_size;
    int tile_x_min = min_c / tile_size;
    int tile_x_max = max_c / tile_size;

    tile_y_min = max(0, min(tiles_y - 1, tile_y_min));
    tile_y_max = max(0, min(tiles_y - 1, tile_y_max));
    tile_x_min = max(0, min(tiles_x - 1, tile_x_min));
    tile_x_max = max(0, min(tiles_x - 1, tile_x_max));

    // Fase 2: key int32.
    // Usamos 16 bits para depth cuantizado y el resto para tile_id.
    // Esto reduce memoria y ancho de banda frente a int64.
    const int DEPTH_SCALE = 65536;
    float dmin = depth_min[0];
    float dmax = depth_max[0];
    float denom = dmax - dmin + 1e-8f;
    float dn = (depth[i] - dmin) / denom;
    dn = fminf(fmaxf(dn, 0.0f), 1.0f);
    int depth_q = (int)(dn * (float)(DEPTH_SCALE - 1));
    depth_q = max(0, min(DEPTH_SCALE - 1, depth_q));

    int base = offsets[i];
    int local = 0;
    for (int ty = tile_y_min; ty <= tile_y_max; ++ty) {
        for (int tx = tile_x_min; tx <= tile_x_max; ++tx) {
            int tile_id = ty * tiles_x + tx;
            int out_idx = base + local;
            keys[out_idx] = tile_id * DEPTH_SCALE + depth_q;
            gaussian_ids[out_idx] = (int32_t)i;
            local++;
        }
    }
}

__global__ __launch_bounds__(256) void identify_tile_ranges_kernel(
    const int32_t* __restrict__ sorted_keys,
    int32_t* __restrict__ ranges,
    int total_instances,
    int total_tiles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_instances) return;

    const int DEPTH_SCALE = 65536;
    int tile = sorted_keys[idx] / DEPTH_SCALE;
    if (tile < 0 || tile >= total_tiles) return;

    if (idx == 0) {
        ranges[tile * 2 + 0] = 0;
    } else {
        int prev_tile = sorted_keys[idx - 1] / DEPTH_SCALE;
        if (prev_tile != tile) {
            ranges[prev_tile * 2 + 1] = idx;
            ranges[tile * 2 + 0] = idx;
        }
    }

    if (idx == total_instances - 1) {
        ranges[tile * 2 + 1] = total_instances;
    }
}

std::vector<torch::Tensor> raster_preprocess_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor depth,
    int H,
    int W,
    int tile_size,
    double k_sigma
) {
    int N = mu.size(0);
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    // Con key int32: key = tile_id * 65536 + depth_q.
    // Esto soporta hasta 32768 tiles con int32 signed.
    TORCH_CHECK(total_tiles <= 32768, "demasiados tiles para key int32: sube tile_size o usa version int64");

    auto opts_i32 = torch::TensorOptions().device(mu.device()).dtype(torch::kInt32);
    auto counts = torch::empty({N}, opts_i32);
    auto offsets = torch::empty({N}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_tile_counts_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        counts.data_ptr<int>(),
        N,
        H,
        W,
        tile_size,
        (float)k_sigma,
        tiles_x,
        tiles_y
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    // Prefix sum para obtener offsets por gaussiana.
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr,
        scan_bytes,
        counts.data_ptr<int>(),
        offsets.data_ptr<int>(),
        N,
        stream
    );
    auto scan_temp = torch::empty({(long long)scan_bytes}, torch::TensorOptions().device(mu.device()).dtype(torch::kUInt8));
    cub::DeviceScan::ExclusiveSum(
        scan_temp.data_ptr<unsigned char>(),
        scan_bytes,
        counts.data_ptr<int>(),
        offsets.data_ptr<int>(),
        N,
        stream
    );

    // Se sincroniza para conocer el tamano de salida dinamico.
    auto total_tensor = counts.sum(torch::kInt64);
    long long total_instances_ll = total_tensor.item<long long>();
    TORCH_CHECK(total_instances_ll <= INT_MAX, "demasiadas instancias tile-gaussiana para int32");
    int total_instances = (int)total_instances_ll;

    auto keys = torch::empty({total_instances}, opts_i32);
    auto gaussian_ids = torch::empty({total_instances}, opts_i32);
    auto keys_sorted = torch::empty({total_instances}, opts_i32);
    auto gaussian_ids_sorted = torch::empty({total_instances}, opts_i32);
    auto ranges = torch::full({total_tiles, 2}, -1, opts_i32);

    if (total_instances == 0) {
        return {gaussian_ids_sorted, ranges};
    }

    auto depth_min = depth.min();
    auto depth_max = depth.max();

    duplicate_with_keys_kernel<<<blocks, threads, 0, stream>>>(
        mu.data_ptr<float>(),
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        depth.data_ptr<float>(),
        depth_min.data_ptr<float>(),
        depth_max.data_ptr<float>(),
        offsets.data_ptr<int>(),
        keys.data_ptr<int32_t>(),
        gaussian_ids.data_ptr<int32_t>(),
        N,
        H,
        W,
        tile_size,
        (float)k_sigma,
        tiles_x,
        tiles_y
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    // Ordena por tile y depth usando radix sort.
    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_bytes,
        keys.data_ptr<int32_t>(),
        keys_sorted.data_ptr<int32_t>(),
        gaussian_ids.data_ptr<int32_t>(),
        gaussian_ids_sorted.data_ptr<int32_t>(),
        (int)total_instances,
        0,
        32,
        stream
    );
    auto sort_temp = torch::empty({(long long)sort_bytes}, torch::TensorOptions().device(mu.device()).dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairs(
        sort_temp.data_ptr<unsigned char>(),
        sort_bytes,
        keys.data_ptr<int32_t>(),
        keys_sorted.data_ptr<int32_t>(),
        gaussian_ids.data_ptr<int32_t>(),
        gaussian_ids_sorted.data_ptr<int32_t>(),
        (int)total_instances,
        0,
        32,
        stream
    );

    int range_threads = 256;
    int range_blocks = ((int)total_instances + range_threads - 1) / range_threads;
    identify_tile_ranges_kernel<<<range_blocks, range_threads, 0, stream>>>(
        keys_sorted.data_ptr<int32_t>(),
        ranges.data_ptr<int32_t>(),
        (int)total_instances,
        total_tiles
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {gaussian_ids_sorted.contiguous(), ranges.contiguous()};
}
