#include <torch/extension.h>

torch::Tensor raster_forward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
);

torch::Tensor raster_forward_cuda(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
);


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
);

std::vector<torch::Tensor> raster_backward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor grad_out,
    int H,
    int W
);
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
);

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
);

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
);

std::vector<torch::Tensor> raster_preprocess_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor depth,
    int H,
    int W,
    int tile_size,
    double k_sigma
);

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
);

torch::Tensor raster_build_conic_cuda(
    torch::Tensor scale,
    torch::Tensor theta
);

std::vector<torch::Tensor> raster_grad_conic_to_scale_theta_cuda(
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor grad_conic
);

torch::Tensor raster_forward(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale debe estar en CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale debe ser contiguous");
    TORCH_CHECK(theta.is_contiguous(), "theta debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale debe ser float32");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu debe tener shape (N, 2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale debe tener shape (N, 2)");
    TORCH_CHECK(theta.dim() == 1, "theta debe tener shape (N)");
    TORCH_CHECK(opacity.dim() == 1, "opacity debe tener shape (N)");
    TORCH_CHECK(color.dim() == 2 && color.size(1) == 3, "color debe tener shape (N, 3)");

    return raster_forward_cuda(mu, scale, theta, opacity, color, H, W);
}



torch::Tensor raster_forward_conic(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu debe tener shape (N, 2)");
    TORCH_CHECK(conic.dim() == 2 && conic.size(1) == 3, "conic debe tener shape (N, 3)");
    TORCH_CHECK(opacity.dim() == 1, "opacity debe tener shape (N)");
    TORCH_CHECK(color.dim() == 2 && color.size(1) == 3, "color debe tener shape (N, 3)");

    return raster_forward_conic_cuda(mu, conic, opacity, color, H, W);
}


torch::Tensor raster_forward_tiled(
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
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(gaussian_ids.is_cuda(), "gaussian_ids debe estar en CUDA");
    TORCH_CHECK(ranges.is_cuda(), "ranges debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(gaussian_ids.is_contiguous(), "gaussian_ids debe ser contiguous");
    TORCH_CHECK(ranges.is_contiguous(), "ranges debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(gaussian_ids.dtype() == torch::kInt32, "gaussian_ids debe ser int32");
    TORCH_CHECK(ranges.dtype() == torch::kInt32, "ranges debe ser int32");

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu debe tener shape (N, 2)");
    TORCH_CHECK(conic.dim() == 2 && conic.size(1) == 3, "conic debe tener shape (N, 3)");
    TORCH_CHECK(color.dim() == 2 && color.size(1) == 3, "color debe tener shape (N, 3)");
    TORCH_CHECK(ranges.dim() == 2 && ranges.size(1) == 2, "ranges debe tener shape (num_tiles, 2)");

    TORCH_CHECK(tile_size > 0, "tile_size debe ser positivo");
    TORCH_CHECK(tile_size * tile_size <= 256, "tile_size demasiado grande para esta version optimizada");

    return raster_forward_tiled_cuda(
        mu, conic, opacity, color, gaussian_ids, ranges, H, W, tile_size
    );
}


std::vector<torch::Tensor> raster_backward_conic(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor grad_out,
    int H,
    int W
) {
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out debe ser float32");

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu debe tener shape (N, 2)");
    TORCH_CHECK(conic.dim() == 2 && conic.size(1) == 3, "conic debe tener shape (N, 3)");
    TORCH_CHECK(opacity.dim() == 1, "opacity debe tener shape (N)");
    TORCH_CHECK(color.dim() == 2 && color.size(1) == 3, "color debe tener shape (N, 3)");
    TORCH_CHECK(grad_out.dim() == 3 && grad_out.size(2) == 3, "grad_out debe tener shape (H, W, 3)");

    return raster_backward_conic_cuda(mu, conic, opacity, color, grad_out, H, W);
}

std::vector<torch::Tensor> raster_backward_tiled(
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
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(gaussian_ids.is_cuda(), "gaussian_ids debe estar en CUDA");
    TORCH_CHECK(ranges.is_cuda(), "ranges debe estar en CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(gaussian_ids.is_contiguous(), "gaussian_ids debe ser contiguous");
    TORCH_CHECK(ranges.is_contiguous(), "ranges debe ser contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out debe ser float32");

    TORCH_CHECK(gaussian_ids.dtype() == torch::kInt32, "gaussian_ids debe ser int32");
    TORCH_CHECK(ranges.dtype() == torch::kInt32, "ranges debe ser int32");

    TORCH_CHECK(tile_size > 0, "tile_size debe ser positivo");
    TORCH_CHECK(tile_size * tile_size <= 256, "tile_size demasiado grande para esta version optimizada");

    return raster_backward_tiled_cuda(
        mu,
        conic,
        opacity,
        color,
        gaussian_ids,
        ranges,
        grad_out,
        H,
        W,
        tile_size
    );
}




std::vector<torch::Tensor> raster_forward_tiled_train(
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
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(gaussian_ids.is_cuda(), "gaussian_ids debe estar en CUDA");
    TORCH_CHECK(ranges.is_cuda(), "ranges debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(gaussian_ids.is_contiguous(), "gaussian_ids debe ser contiguous");
    TORCH_CHECK(ranges.is_contiguous(), "ranges debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(gaussian_ids.dtype() == torch::kInt32, "gaussian_ids debe ser int32");
    TORCH_CHECK(ranges.dtype() == torch::kInt32, "ranges debe ser int32");

    TORCH_CHECK(tile_size > 0, "tile_size debe ser positivo");
    TORCH_CHECK(tile_size * tile_size <= 256, "tile_size demasiado grande para esta version optimizada");

    return raster_forward_tiled_train_cuda(
        mu, conic, opacity, color, gaussian_ids, ranges, H, W, tile_size
    );
}


std::vector<torch::Tensor> raster_backward_tiled_fast(
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
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(gaussian_ids.is_cuda(), "gaussian_ids debe estar en CUDA");
    TORCH_CHECK(ranges.is_cuda(), "ranges debe estar en CUDA");
    TORCH_CHECK(final_Ts.is_cuda(), "final_Ts debe estar en CUDA");
    TORCH_CHECK(n_contrib.is_cuda(), "n_contrib debe estar en CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(gaussian_ids.is_contiguous(), "gaussian_ids debe ser contiguous");
    TORCH_CHECK(ranges.is_contiguous(), "ranges debe ser contiguous");
    TORCH_CHECK(final_Ts.is_contiguous(), "final_Ts debe ser contiguous");
    TORCH_CHECK(n_contrib.is_contiguous(), "n_contrib debe ser contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(final_Ts.dtype() == torch::kFloat32, "final_Ts debe ser float32");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "grad_out debe ser float32");
    TORCH_CHECK(gaussian_ids.dtype() == torch::kInt32, "gaussian_ids debe ser int32");
    TORCH_CHECK(ranges.dtype() == torch::kInt32, "ranges debe ser int32");
    TORCH_CHECK(n_contrib.dtype() == torch::kInt32, "n_contrib debe ser int32");

    TORCH_CHECK(tile_size > 0, "tile_size debe ser positivo");
    TORCH_CHECK(tile_size * tile_size <= 256, "tile_size demasiado grande para esta version optimizada");

    return raster_backward_tiled_fast_cuda(
        mu, conic, opacity, color, gaussian_ids, ranges,
        final_Ts, n_contrib, grad_out, H, W, tile_size
    );
}


std::vector<torch::Tensor> raster_preprocess_tiled(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor depth,
    int H,
    int W,
    int tile_size,
    double k_sigma
) {
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(scale.is_cuda(), "scale debe estar en CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta debe estar en CUDA");
    TORCH_CHECK(depth.is_cuda(), "depth debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale debe ser contiguous");
    TORCH_CHECK(theta.is_contiguous(), "theta debe ser contiguous");
    TORCH_CHECK(depth.is_contiguous(), "depth debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale debe ser float32");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta debe ser float32");
    TORCH_CHECK(depth.dtype() == torch::kFloat32, "depth debe ser float32");

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu debe tener shape (N, 2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale debe tener shape (N, 2)");
    TORCH_CHECK(theta.dim() == 1, "theta debe tener shape (N)");
    TORCH_CHECK(depth.dim() == 1, "depth debe tener shape (N)");
    TORCH_CHECK(mu.size(0) == scale.size(0), "mu y scale deben tener el mismo N");
    TORCH_CHECK(mu.size(0) == theta.size(0), "mu y theta deben tener el mismo N");
    TORCH_CHECK(mu.size(0) == depth.size(0), "mu y depth deben tener el mismo N");

    TORCH_CHECK(tile_size > 0, "tile_size debe ser positivo");
    TORCH_CHECK(tile_size * tile_size <= 256, "tile_size demasiado grande para esta version optimizada");

    return raster_preprocess_tiled_cuda(
        mu, scale, theta, depth, H, W, tile_size, k_sigma
    );
}



torch::Tensor raster_build_conic(
    torch::Tensor scale,
    torch::Tensor theta
) {
    TORCH_CHECK(scale.is_cuda(), "scale debe estar en CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta debe estar en CUDA");

    TORCH_CHECK(scale.is_contiguous(), "scale debe ser contiguous");
    TORCH_CHECK(theta.is_contiguous(), "theta debe ser contiguous");

    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale debe ser float32");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta debe ser float32");

    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale debe tener shape (N, 2)");
    TORCH_CHECK(theta.dim() == 1, "theta debe tener shape (N)");
    TORCH_CHECK(scale.size(0) == theta.size(0), "scale y theta deben tener el mismo N");

    return raster_build_conic_cuda(scale, theta);
}


std::vector<torch::Tensor> raster_grad_conic_to_scale_theta(
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor grad_conic
) {
    TORCH_CHECK(scale.is_cuda(), "scale debe estar en CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta debe estar en CUDA");
    TORCH_CHECK(grad_conic.is_cuda(), "grad_conic debe estar en CUDA");

    TORCH_CHECK(scale.is_contiguous(), "scale debe ser contiguous");
    TORCH_CHECK(theta.is_contiguous(), "theta debe ser contiguous");
    TORCH_CHECK(grad_conic.is_contiguous(), "grad_conic debe ser contiguous");

    TORCH_CHECK(scale.dtype() == torch::kFloat32, "scale debe ser float32");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta debe ser float32");
    TORCH_CHECK(grad_conic.dtype() == torch::kFloat32, "grad_conic debe ser float32");

    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale debe tener shape (N, 2)");
    TORCH_CHECK(theta.dim() == 1, "theta debe tener shape (N)");
    TORCH_CHECK(grad_conic.dim() == 2 && grad_conic.size(1) == 3, "grad_conic debe tener shape (N, 3)");
    TORCH_CHECK(scale.size(0) == theta.size(0), "scale y theta deben tener el mismo N");
    TORCH_CHECK(scale.size(0) == grad_conic.size(0), "scale y grad_conic deben tener el mismo N");

    return raster_grad_conic_to_scale_theta_cuda(scale, theta, grad_conic);
}

std::vector<torch::Tensor> raster_forward_tiled_train_loss(
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
    std::string loss_name
) {
    TORCH_CHECK(mu.is_cuda(), "mu debe estar en CUDA");
    TORCH_CHECK(conic.is_cuda(), "conic debe estar en CUDA");
    TORCH_CHECK(opacity.is_cuda(), "opacity debe estar en CUDA");
    TORCH_CHECK(color.is_cuda(), "color debe estar en CUDA");
    TORCH_CHECK(gaussian_ids.is_cuda(), "gaussian_ids debe estar en CUDA");
    TORCH_CHECK(ranges.is_cuda(), "ranges debe estar en CUDA");
    TORCH_CHECK(target.is_cuda(), "target debe estar en CUDA");

    TORCH_CHECK(mu.is_contiguous(), "mu debe ser contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic debe ser contiguous");
    TORCH_CHECK(opacity.is_contiguous(), "opacity debe ser contiguous");
    TORCH_CHECK(color.is_contiguous(), "color debe ser contiguous");
    TORCH_CHECK(gaussian_ids.is_contiguous(), "gaussian_ids debe ser contiguous");
    TORCH_CHECK(ranges.is_contiguous(), "ranges debe ser contiguous");
    TORCH_CHECK(target.is_contiguous(), "target debe ser contiguous");

    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu debe ser float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic debe ser float32");
    TORCH_CHECK(opacity.dtype() == torch::kFloat32, "opacity debe ser float32");
    TORCH_CHECK(color.dtype() == torch::kFloat32, "color debe ser float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target debe ser float32");
    TORCH_CHECK(gaussian_ids.dtype() == torch::kInt32, "gaussian_ids debe ser int32");
    TORCH_CHECK(ranges.dtype() == torch::kInt32, "ranges debe ser int32");

    TORCH_CHECK(target.dim() == 3, "target debe tener shape (H, W, 3)");
    TORCH_CHECK(target.size(0) == H, "target H incorrecto");
    TORCH_CHECK(target.size(1) == W, "target W incorrecto");
    TORCH_CHECK(target.size(2) == 3, "target debe tener 3 canales");

    int loss_type = 0;
    if (loss_name == "l1") {
        loss_type = 0;
    } else if (loss_name == "mse") {
        loss_type = 1;
    } else {
        TORCH_CHECK(false, "loss_name debe ser 'l1' o 'mse'");
    }

    return raster_forward_tiled_train_loss_cuda(
        mu,
        conic,
        opacity,
        color,
        gaussian_ids,
        ranges,
        target,
        H,
        W,
        tile_size,
        loss_type
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


    // ESTAS DE ACA SON FUNCIONES ANTIGUAS QUE SE FUERON MEJORANDO POCO A POCO
    m.def("forward", &raster_forward, "Raster forward CUDA");
    m.def("forward_conic", &raster_forward_conic, "Raster forward CUDA con conic precomputado");
    m.def("forward_tiled", &raster_forward_tiled, "Raster forward CUDA tiled");
    m.def("backward_conic", &raster_backward_conic, "Raster backward CUDA conic");
    m.def("backward_tiled", &raster_backward_tiled, "Raster backward CUDA tiled");
    //



    // ESTAS FUNCIONES SON LAS QUE SE USAN AHORA
    // ESTA solo calcula el forward, el ll1 y dssim es calculado por pytorch
    m.def("forward_tiled_train", &raster_forward_tiled_train, "Raster forward CUDA tiled train");
    // Esta es igual a forward_tiled_train, solo que calcula el ll1, es más rapido pero no calcula el dssim.
    m.def("forward_tiled_train_loss", &raster_forward_tiled_train_loss, "Raster forward CUDA tiled train loss");

   // Backward principal
    m.def("backward_tiled_fast", &raster_backward_tiled_fast, "Raster backward CUDA tiled fast");

    // Preprocesa gaussianas por tiles.
    m.def("preprocess_tiled", &raster_preprocess_tiled, "Preprocess CUDA tiled con CUB");

    // Construye la representacion conic a partir de scale/theta.
    m.def("build_conic", &raster_build_conic, "Construye conic en CUDA");

    // Convierte gradientes respecto a conic en gradientes de scale/theta.
    m.def("grad_conic_to_scale_theta", &raster_grad_conic_to_scale_theta, "Convierte grad_conic a grad_scale y grad_theta en CUDA");

}