#include "../runtime_api.hpp"

#include "nvidia_utils.cuh"

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    // TO_BE_IMPLEMENTED();
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    // TO_BE_IMPLEMENTED();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}
void streamSynchronize(llaisysStream_t stream) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    // TO_BE_IMPLEMENTED();
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    // TO_BE_IMPLEMENTED();
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    // TO_BE_IMPLEMENTED();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), static_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
