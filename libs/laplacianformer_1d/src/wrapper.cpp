#include <torch/extension.h>

void laplacian_1d_forward_cuda(
    const at::Tensor query,
    const at::Tensor key,
    at::Tensor output);

void laplacian_1d_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor query,
    const at::Tensor key,
    at::Tensor grad_query,
    at::Tensor grad_key);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("laplacian_1d_forward", &laplacian_1d_forward_cuda,
        "1D Laplacian pairwise L1 distance forward (CUDA)");
  m.def("laplacian_1d_backward", &laplacian_1d_backward_cuda,
        "1D Laplacian pairwise L1 distance backward (CUDA)");
}
