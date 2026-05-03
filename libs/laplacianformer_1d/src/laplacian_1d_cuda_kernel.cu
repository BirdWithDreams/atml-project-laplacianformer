#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

template <typename scalar_t>
struct AccT { using type = float; };
template <>
struct AccT<double> { using type = double; };

template <typename scalar_t>
__device__ __forceinline__ typename AccT<scalar_t>::type to_acc(scalar_t v) {
    return static_cast<typename AccT<scalar_t>::type>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_acc(typename AccT<scalar_t>::type v) {
    return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__device__ __forceinline__ typename AccT<scalar_t>::type sign_diff(scalar_t lhs, scalar_t rhs) {
    using acc_t = typename AccT<scalar_t>::type;
    acc_t diff = to_acc<scalar_t>(lhs) - to_acc<scalar_t>(rhs);
    return (diff > acc_t(0)) - (diff < acc_t(0));
}

template <typename T>
__device__ __forceinline__ T abs_val(T v) {
    return v < T(0) ? -v : v;
}

template <typename scalar_t>
__global__ void laplacian_1d_forward_kernel(
        const scalar_t* __restrict__ query,
        const scalar_t* __restrict__ key,
        scalar_t* __restrict__ output,
        int total,
        int N,
        int M,
        int D) {
    using acc_t = typename AccT<scalar_t>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int m = idx % M;
    int tmp = idx / M;
    int n = tmp % N;
    int bh = tmp / N;

    const scalar_t* q = query + (bh * N + n) * D;
    const scalar_t* k = key + (bh * M + m) * D;

    acc_t acc = acc_t(0);
    for (int d = 0; d < D; ++d) {
        acc_t diff = to_acc<scalar_t>(q[d]) - to_acc<scalar_t>(k[d]);
        acc += abs_val<acc_t>(diff);
    }
    output[idx] = from_acc<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void laplacian_1d_backward_query_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ query,
        const scalar_t* __restrict__ key,
        scalar_t* __restrict__ grad_query,
        int total,
        int N,
        int M,
        int D) {
    using acc_t = typename AccT<scalar_t>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int n = tmp % N;
    int bh = tmp / N;

    scalar_t q_val = query[(bh * N + n) * D + d];
    acc_t acc = acc_t(0);
    for (int m = 0; m < M; ++m) {
        scalar_t k_val = key[(bh * M + m) * D + d];
        acc_t grad = to_acc<scalar_t>(grad_output[(bh * N + n) * M + m]);
        acc += grad * sign_diff(q_val, k_val);
    }
    grad_query[idx] = from_acc<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void laplacian_1d_backward_key_kernel(
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ query,
        const scalar_t* __restrict__ key,
        scalar_t* __restrict__ grad_key,
        int total,
        int N,
        int M,
        int D) {
    using acc_t = typename AccT<scalar_t>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int m = tmp % M;
    int bh = tmp / M;

    scalar_t k_val = key[(bh * M + m) * D + d];
    acc_t acc = acc_t(0);
    for (int n = 0; n < N; ++n) {
        scalar_t q_val = query[(bh * N + n) * D + d];
        acc_t grad = to_acc<scalar_t>(grad_output[(bh * N + n) * M + m]);
        acc -= grad * sign_diff(q_val, k_val);
    }
    grad_key[idx] = from_acc<scalar_t>(acc);
}

void check_inputs(const at::Tensor query, const at::Tensor key) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda(), "query and key must be CUDA tensors");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous(), "query and key must be contiguous");
    TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "expected query/key shape (B,H,N,D)/(B,H,M,D)");
    TORCH_CHECK(query.size(0) == key.size(0), "query/key batch size mismatch");
    TORCH_CHECK(query.size(1) == key.size(1), "query/key head count mismatch");
    TORCH_CHECK(query.size(3) == key.size(3), "query/key feature dimension mismatch");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
    TORCH_CHECK(
        query.scalar_type() == at::ScalarType::Float || query.scalar_type() == at::ScalarType::Double,
        "laplacian_1d_cuda currently supports float32 and float64 tensors"
    );
}

}  // namespace

void laplacian_1d_forward_cuda(
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor output) {
    check_inputs(query, key);
    at::cuda::CUDAGuard device_guard(query.device());

    int B = query.size(0);
    int H = query.size(1);
    int N = query.size(2);
    int M = key.size(2);
    int D = query.size(3);
    int total = B * H * N * M;

    TORCH_CHECK(output.is_cuda() && output.is_contiguous(), "output must be a contiguous CUDA tensor");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({B, H, N, M}), "output shape mismatch");

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "laplacian_1d_forward_cuda", ([&] {
        laplacian_1d_forward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total,
            N,
            M,
            D);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void laplacian_1d_backward_cuda(
        const at::Tensor grad_output,
        const at::Tensor query,
        const at::Tensor key,
        at::Tensor grad_query,
        at::Tensor grad_key) {
    check_inputs(query, key);
    at::cuda::CUDAGuard device_guard(query.device());

    int B = query.size(0);
    int H = query.size(1);
    int N = query.size(2);
    int M = key.size(2);
    int D = query.size(3);

    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(), "grad_output must be contiguous CUDA");
    TORCH_CHECK(grad_query.is_cuda() && grad_query.is_contiguous(), "grad_query must be contiguous CUDA");
    TORCH_CHECK(grad_key.is_cuda() && grad_key.is_contiguous(), "grad_key must be contiguous CUDA");
    TORCH_CHECK(grad_output.sizes() == at::IntArrayRef({B, H, N, M}), "grad_output shape mismatch");
    TORCH_CHECK(grad_query.sizes() == query.sizes(), "grad_query shape mismatch");
    TORCH_CHECK(grad_key.sizes() == key.sizes(), "grad_key shape mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int query_total = B * H * N * D;
    int key_total = B * H * M * D;

    dim3 block(THREADS);
    dim3 query_grid((query_total + THREADS - 1) / THREADS);
    dim3 key_grid((key_total + THREADS - 1) / THREADS);

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "laplacian_1d_backward_cuda", ([&] {
        laplacian_1d_backward_query_kernel<scalar_t><<<query_grid, block, 0, stream>>>(
            grad_output.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            grad_query.data_ptr<scalar_t>(),
            query_total,
            N,
            M,
            D);
        laplacian_1d_backward_key_kernel<scalar_t><<<key_grid, block, 0, stream>>>(
            grad_output.data_ptr<scalar_t>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            grad_key.data_ptr<scalar_t>(),
            key_total,
            N,
            M,
            D);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
