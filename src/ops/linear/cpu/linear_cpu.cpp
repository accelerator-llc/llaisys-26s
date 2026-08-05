#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace llaisys::ops::cpu::detail {

// 持久线程池：worker 常驻，linear 调用时分发 out_features 分块任务，
// 避免每调用 std::thread 创建/join 的开销（Fix CR#L25 中危2）。
// 参照 vLLM CPU/llama.cpp 的常驻 worker 池思路（接口与数据结构为本项目自定义）。
class LinearPool {
public:
    LinearPool() : active_(0), stop_(false) {
        size_t n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
        nthread_ = n;
        workers_.reserve(n);
        for (size_t i = 0; i < n; i++) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }
    ~LinearPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_) w.join();
    }
    size_t size() const { return nthread_; }

    // 并行执行 n 个任务块：task(t) 处理块 t，主线程阻塞至全部完成。
    // 每个 k 的内层 in_features 累加顺序不变，结果与单线程 bit-exact 一致。
    template <class F>
    void run(size_t n, F task) {
        if (n == 0) return;
        std::unique_lock<std::mutex> lk(mtx_);
        for (size_t i = 0; i < n; i++) {
            tasks_.push([task, i] { task(i); });
            active_++;
        }
        cv_.notify_all();
        done_cv_.wait(lk, [this] { return active_ == 0; });
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lk(mtx_);
                active_--;
                if (active_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    size_t nthread_;
    size_t active_;
    bool stop_;
};

// Meyers singleton：进程内首次调用时构造，退出时析构（C++11 保证初始化线程安全）。
LinearPool &get_pool() {
    static LinearPool pool;
    return pool;
}

} // namespace llaisys::ops::cpu::detail

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t n, size_t in_features, size_t out_features, bool has_bias) {
    // TO_BE_IMPLEMENTED();
    // Y = xW^T + b，weight 未转置故 W^T[k,i] = W[k,i]。
    // 行主序下外层按样本、中层按输出通道、内层沿 in_features 连续累加，
    // 对 x 行与 w 行均缓存友好（与 PyTorch CPU addmm 三重循环顺序一致）。
    // 低精度类型在 float 域累加后 cast<T> 写回，保证数值精度（bf16/f16 必须提升累加）。
    //
    // 多线程分块：按 out_features 维度切分给持久线程池的 worker（不同 k 写 out 不同位置，
    // 无数据竞争）；每个 k 的内层 in_features 累加顺序不变，故结果与单线程 bit-exact 一致。
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

    // Fix CR#L41(中危2): 阈值改为 1024（原 nthread*4 过低，decode 小矩阵线程创建开销 > 计算收益）。
    auto &pool = llaisys::ops::cpu::detail::get_pool();
    const size_t nthread = std::min(out_features, pool.size());
    if (out_features < 1024 || nthread <= 1) {
        worker(0, out_features);
        return;
    }

    // Fix CR#L25(中危2): 持久线程池替代每调用 std::thread 创建/join。
    const size_t chunk = (out_features + nthread - 1) / nthread;
    pool.run(nthread, [&](size_t t) {
        size_t k_begin = t * chunk;
        size_t k_end = std::min(k_begin + chunk, out_features);
        if (k_begin < k_end) worker(k_begin, k_end);
    });
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
