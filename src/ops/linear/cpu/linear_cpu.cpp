#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t n, size_t in_features, size_t out_features, bool has_bias) {
    // TO_BE_IMPLEMENTED();
    // Y = xW^T + b，weight 未转置故 W^T[k,i] = W[k,i]。
    // 行主序下外层按样本、中层按输出通道、内层沿 in_features 连续累加，
    // 对 x 行与 w 行均缓存友好（与 PyTorch CPU addmm 三重循环顺序一致）。
    // 低精度类型在 float 域累加后 cast<T> 写回，保证数值精度（bf16/f16 必须提升累加）。
    //
    // 多线程分块：按 out_features 维度切分给多线程（不同 k 写 out 不同位置，无数据竞争）；
    // 每个线程负责所有 n 个样本的一段 k 区间，线程数仅创建一次而非每样本一次。
    // 每个 k 的内层 in_features 累加顺序不变，故结果与单线程 bit-exact 一致。
    // 小矩阵（out_features 不足）退化为单线程，避免线程创建开销大于计算。
    auto worker = [&](size_t k_begin, size_t k_end) {
        for (size_t nn = 0; nn < n; nn++) {
            const T *x_row = in + nn * in_features;
            T *y_row = out + nn * out_features;
            for (size_t k = k_begin; k < k_end; k++) {
                const T *w_row = weight + k * in_features;
                float acc = has_bias ? llaisys::utils::cast<float>(bias[k]) : 0.0f;
                for (size_t i = 0; i < in_features; i++) {
                    acc += llaisys::utils::cast<float>(x_row[i]) * llaisys::utils::cast<float>(w_row[i]);
                }
                y_row[k] = llaisys::utils::cast<T>(acc);
            }
        }
    };

    size_t hardware = static_cast<size_t>(std::thread::hardware_concurrency());
    if (hardware == 0) hardware = 1;
    const size_t nthread = std::min(out_features, hardware);
    // 阈值：每线程至少 4 个 k，否则线程创建开销不划算。
    const bool use_parallel = (nthread > 1 && out_features >= nthread * 4);

    if (!use_parallel) {
        worker(0, out_features);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(nthread);
    const size_t chunk = (out_features + nthread - 1) / nthread;
    for (size_t t = 0; t < nthread; t++) {
        size_t k_begin = t * chunk;
        size_t k_end = std::min(k_begin + chunk, out_features);
        if (k_begin >= k_end) break;
        threads.emplace_back(worker, k_begin, k_end);
    }
    for (auto &th : threads) th.join();
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t val_type,
            size_t n, size_t in_features, size_t out_features, bool has_bias) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                              reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                              n, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                        reinterpret_cast<const llaisys::bf16_t *>(in),
                                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                                        reinterpret_cast<const llaisys::bf16_t *>(bias),
                                        n, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                        reinterpret_cast<const llaisys::fp16_t *>(in),
                                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                                        reinterpret_cast<const llaisys::fp16_t *>(bias),
                                        n, in_features, out_features, has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
