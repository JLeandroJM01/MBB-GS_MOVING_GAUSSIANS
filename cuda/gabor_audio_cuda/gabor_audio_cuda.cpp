// Bindings PyTorch para el Gabor splatting de audio (waveform 1D).
//
// Expone:
//   forward(mu, sigma, amp, fnorm, phi, T, k_sigma) -> x_hat [T]
//   backward(mu, sigma, amp, fnorm, phi, grad_out, k_sigma)
//        -> [grad_mu, grad_sigma, grad_amp, grad_fnorm, grad_phi]
//
// La logica de activacion (sigma = exp(.), fnorm = sigmoid(.) * 0.5, etc.)
// se hace en PyTorch antes de llamar a estas funciones, igual que en el
// rasterizador de video. Aqui solo se reciben los parametros ya activados.

#include <torch/extension.h>
#include <vector>


torch::Tensor gabor_forward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    int T,
    float k_sigma
);

std::vector<torch::Tensor> gabor_backward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    torch::Tensor grad_out,
    float k_sigma
);


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " debe ser un tensor CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " debe ser contiguo")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " debe ser float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)


torch::Tensor forward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    int64_t T,
    double k_sigma
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);
    CHECK_INPUT(fnorm);
    CHECK_INPUT(phi);

    TORCH_CHECK(mu.dim() == 1, "mu debe ser 1D [N]");
    TORCH_CHECK(sigma.size(0) == mu.size(0), "sigma y mu deben tener el mismo N");
    TORCH_CHECK(amp.size(0) == mu.size(0), "amp y mu deben tener el mismo N");
    TORCH_CHECK(fnorm.size(0) == mu.size(0), "fnorm y mu deben tener el mismo N");
    TORCH_CHECK(phi.size(0) == mu.size(0), "phi y mu deben tener el mismo N");
    TORCH_CHECK(T > 0, "T debe ser > 0");

    return gabor_forward_cuda(mu, sigma, amp, fnorm, phi, (int)T, (float)k_sigma);
}


std::vector<torch::Tensor> backward(
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor amp,
    torch::Tensor fnorm,
    torch::Tensor phi,
    torch::Tensor grad_out,
    double k_sigma
) {
    CHECK_INPUT(mu);
    CHECK_INPUT(sigma);
    CHECK_INPUT(amp);
    CHECK_INPUT(fnorm);
    CHECK_INPUT(phi);
    CHECK_INPUT(grad_out);

    TORCH_CHECK(grad_out.dim() == 1, "grad_out debe ser 1D [T]");

    return gabor_backward_cuda(mu, sigma, amp, fnorm, phi, grad_out, (float)k_sigma);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Gabor audio render forward (CUDA)");
    m.def("backward", &backward, "Gabor audio render backward (CUDA)");
}
