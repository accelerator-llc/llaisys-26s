#include "linear_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

// cuBLAS handle（单 device，懒创建；C++11 保证初始化线程安全）。
// 单 device 约束；多 device 需 per-device handle。
// linear 用 cuBLAS，与 llama.cpp/vLLM/PyTorch 对 F32/F16/BF16 矩阵乘的做法一致，
// cuBLAS 自动选 GEMM（m 大）与 GEMV（m=1 decode）路径。
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// bias 加法 kernel：Y[nn, k] += bias[k]，低精度 float 域加（cuBLAS gemm 不含 bias）。
template <typename T>
__global__ void add_bias_kernel(T *y, const T *bias, size_t n, size_t out_features) {
    size_t nn = blockIdx.x;
    if (nn >= n) {
        return;
    }
    T *y_row = y + nn * out_features;
    for (size_t k = threadIdx.x; k < out_features; k += blockDim.x) {
        y_row[k] = from_float<T>(to_float(y_row[k]) + to_float(bias[k]));
    }
}

static cudaDataType_t to_cuda_dtype(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        ASSERT(false, "Linear: unsupported dtype for cuBLAS " << static_cast<int>(type));
        return CUDA_R_32F;
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t val_type, size_t n, size_t in_features, size_t out_features, bool has_bias) {
    if (n == 0 || out_features == 0) {
        return;
    }
    // Y = X @ W^T (+ b)。行主序 X(n,in), W(out,in), Y(n,out)，W 不转置（W[k] 行 = 第 k 输出特征）。
    // cuBLAS 列主序视角：Y^T(out,n) = W(out,in) * X^T(in,n)。
    //   cublasGemmEx: C(M=out, N=n, K=in) = op(A)(M,K) * op(B)(K,N)
    //   A=W, op_A=T（W 行主序 (out,in) 的内存 = W^T 列主序 (in,out)，op T -> 逻辑 (out,in)），lda=in
    //   B=X, op_B=N（X 行主序 (n,in) 的内存 = X^T 列主序 (in,n)），ldb=in
    //   C=Y, ldc=out（Y 行主序 (n,out) 的内存 = Y^T 列主序 (out,n)）
    //   不做任何多余转置，仅靠 op/leading-dim 适配行主序。
    // computeType=CUBLAS_COMPUTE_32F：低精度在 float 域累加，与 cpu::linear 一致。
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t dtype = to_cuda_dtype(val_type);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(out_features), static_cast<int>(n), static_cast<int>(in_features),
                            &alpha, weight, dtype, static_cast<int>(in_features),
                            in, dtype, static_cast<int>(in_features),
                            &beta, out, dtype, static_cast<int>(out_features),
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    if (has_bias) {
        constexpr int block = 256;
        int grid = static_cast<int>(n);
        switch (val_type) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<grid, block>>>(
                reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), n, out_features);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<llaisys::bf16_t><<<grid, block>>>(
                reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(bias), n, out_features);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<llaisys::fp16_t><<<grid, block>>>(
                reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(bias), n, out_features);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace llaisys::ops::nvidia
