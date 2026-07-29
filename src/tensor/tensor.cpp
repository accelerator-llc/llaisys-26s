#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // TO_BE_IMPLEMENTED();
    // 零维张量（标量）始终视为连续
    if (_meta.shape.empty()) {
        return true;
    }
    ptrdiff_t expected_stride = 1;
    for (int i = (int)_meta.shape.size() - 1; i >= 0; i--) {
        // size == 1 的维度对连续性无影响（stride 可以是任意值）
        if (_meta.shape[i] != 1 && _meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= (ptrdiff_t)_meta.shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // TO_BE_IMPLEMENTED();
    ASSERT(order.size() == _meta.shape.size(),
           "permute: order must have same number of dimensions as tensor");

    // 校验 order 是 [0, 1, ..., ndim-1] 的一个合法排列，无重复、遗漏
    std::vector<bool> seen(_meta.shape.size(), false);
    std::vector<size_t> new_shape(_meta.shape.size());
    std::vector<ptrdiff_t> new_strides(_meta.shape.size());

    for (size_t i = 0; i < order.size(); i++) {
        ASSERT(order[i] < _meta.shape.size(),
               "permute: order index out of range");
        ASSERT(!seen[order[i]],
               "permute: duplicate dimension in order");
        seen[order[i]] = true;
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // TO_BE_IMPLEMENTED();
    // 1. 总元素数必须一致（带溢出检查）
    size_t new_numel = 1;
    for (size_t s : shape) {
        if (s != 0 && new_numel > SIZE_MAX / s) {
            CHECK_ARGUMENT(false, "view: new shape has too many elements (overflow)");
        }
        new_numel *= s;
    }
    CHECK_ARGUMENT(new_numel == this->numel(),
                   "view: new shape must have same number of elements");

    std::vector<ptrdiff_t> new_strides(shape.size());

    // 2. 连续张量：新 strides 直接从新 shape 算
    if (this->isContiguous()) {
        size_t stride = 1;
        for (size_t i = shape.size(); i > 0; i--) {
            new_strides[i - 1] = (ptrdiff_t)stride;
            stride *= shape[i - 1];
        }
    } else {
        // 3. 不连续张量：检查 stride 兼容性
        auto work_shape = _meta.shape;
        auto work_strides = _meta.strides;
        int old_dim = (int)work_shape.size() - 1;

        for (int new_dim = (int)shape.size() - 1; new_dim >= 0; new_dim--) {
            size_t need = shape[new_dim];

            int merge_start = old_dim;
            size_t prod = 1;

            while (prod < need && old_dim >= 0) {
                prod *= work_shape[old_dim];
                old_dim--;
            }

            CHECK_ARGUMENT(prod >= need && prod % need == 0,
                           "view: shape not compatible with strides. Use reshape() instead.");

            // 检查被合并的维度在内存中连续
            for (int j = merge_start; j > old_dim + 1; j--) {
                CHECK_ARGUMENT(_meta.strides[j - 1] == _meta.strides[j] * (ptrdiff_t)_meta.shape[j],
                               "view: merged dimensions not contiguous. Use reshape() instead.");
            }

            new_strides[new_dim] = work_strides[merge_start];

            if (prod > need) {
                old_dim++;
                work_shape[old_dim] = prod / need;
                work_strides[old_dim] = work_strides[old_dim] * (ptrdiff_t)need;
            }
        }
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // TO_BE_IMPLEMENTED();
    ASSERT(dim < _meta.shape.size(), "slice: dim out of range");
    ASSERT(start <= end && end <= _meta.shape[dim],
           "slice: invalid range (start must be <= end, and end must be <= dimension size)");

    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 计算新的字节偏移（带溢出检查）
    size_t byte_stride = (size_t)_meta.strides[dim] * this->elementSize();
    if (start != 0 && byte_stride != 0 && start > SIZE_MAX / byte_stride) {
        CHECK_ARGUMENT(false, "slice: offset calculation overflow");
    }
    size_t byte_offset = start * byte_stride;
    if (_offset > SIZE_MAX - byte_offset) {
        CHECK_ARGUMENT(false, "slice: offset calculation overflow");
    }
    size_t new_offset = _offset + byte_offset;

    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

void Tensor::load(const void *src_) {
    // TO_BE_IMPLEMENTED();
    ASSERT(src_ != nullptr, "load: source pointer must not be null");
    ASSERT(this->isContiguous(), "load: tensor must be contiguous");

    core::context().setDevice(this->deviceType(), this->deviceId());

    llaisysMemcpyKind_t copy_kind = (this->deviceType() == LLAISYS_DEVICE_CPU)
                                      ? LLAISYS_MEMCPY_H2H
                                      : LLAISYS_MEMCPY_H2D;

    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        this->numel() * this->elementSize(),
        copy_kind);
}

tensor_t Tensor::contiguous() const {
    // TO_BE_IMPLEMENTED();
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    size_t dsize = this->elementSize();
    size_t total = this->numel();
    size_t ndim = this->ndim();

    auto result = create(_meta.shape, _meta.dtype, this->deviceType(), this->deviceId());
    if (total == 0) {
        return result;
    }

    core::context().setDevice(this->deviceType(), this->deviceId());

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte *dst = result->data();
        const std::byte *src = this->data();

        // 进位传播：避免逐元素取模/除法，均摊 O(1) 每元素
        // NOTE: 当前逐元素搬运导致 O(total * ndim) 次循环开销，且每次仅搬运 dsize 字节。
        // 优化方向：检测最后一维 (或连续段) 步长是否为 1，若连续则改用 memcpy 整行/整段搬运，
        // 将 memcpy 调用次数从 total 降低至 total / 连续段长度，并利用 SIMD 提升带宽利用率。
        std::vector<size_t> indices(ndim, 0);
        for (size_t linear = 0; linear < total; linear++) {
            size_t src_offset = 0;
            for (size_t d = 0; d < ndim; d++) {
                src_offset += indices[d] * (size_t)_meta.strides[d];
            }
            std::memcpy(dst + linear * dsize,
                        src + src_offset * dsize, dsize);

            for (size_t d = ndim; d-- > 0; ) {
                if (++indices[d] < _meta.shape[d]) break;
                indices[d] = 0;
            }
        }
    } else {
        // NOTE: 复制整个 storage（_storage->size() bytes）。对于大 storage 上的
        // 小 slice/view，会浪费 PCIe 带宽。优化方向：计算实际访问的字节范围
        // [_offset, _offset + max_stride_offset + dsize)，只复制该范围。
        size_t storage_bytes = _storage->size();
        auto raw_host = core::context().runtime().allocateHostStorage(storage_bytes);
        core::context().runtime().api()->memcpy_sync(
            raw_host->memory(), _storage->memory(),
            storage_bytes, LLAISYS_MEMCPY_D2H);

        auto compact_host = core::context().runtime().allocateHostStorage(total * dsize);
        std::byte *compact_ptr = compact_host->memory();
        const std::byte *raw_ptr = raw_host->memory() + _offset;

        // 进位传播：避免逐元素取模/除法，均摊 O(1) 每元素
        std::vector<size_t> indices(ndim, 0);
        for (size_t linear = 0; linear < total; linear++) {
            size_t src_offset = 0;
            for (size_t d = 0; d < ndim; d++) {
                src_offset += indices[d] * (size_t)_meta.strides[d];
            }
            std::memcpy(compact_ptr + linear * dsize,
                        raw_ptr + src_offset * dsize, dsize);

            for (size_t d = ndim; d-- > 0; ) {
                if (++indices[d] < _meta.shape[d]) break;
                indices[d] = 0;
            }
        }

        core::context().runtime().api()->memcpy_sync(
            result->data(), compact_ptr, total * dsize, LLAISYS_MEMCPY_H2D);
    }

    return result;
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // TO_BE_IMPLEMENTED();
    size_t new_numel = 1;
    for (size_t s : shape) {
        if (s != 0 && new_numel > SIZE_MAX / s) {
            CHECK_ARGUMENT(false, "reshape: new shape has too many elements (overflow)");
        }
        new_numel *= s;
    }
    CHECK_ARGUMENT(new_numel == this->numel(),
                   "reshape: new shape must have same number of elements");

    try {
        return this->view(shape);
    } catch (const std::invalid_argument &) {
        // view 形状/步长不兼容时，fallback 到复制后重试
        return this->contiguous()->view(shape);
    }
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // TO_BE_IMPLEMENTED();
    int target_device = (device < 0) ? this->deviceId() : device;

    if (this->deviceType() == device_type && this->deviceId() == target_device) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    auto src = this->contiguous();

    auto result = create(src->_meta.shape, src->_meta.dtype, device_type, target_device);

    size_t numel = src->numel();
    size_t esize = src->elementSize();
    if (numel != 0 && esize > SIZE_MAX / numel) {
        CHECK_ARGUMENT(false, "to: tensor too large (numel * elementSize overflow)");
    }
    size_t nbytes = numel * esize;
    if (nbytes == 0) {
        return result;
    }

    llaisysMemcpyKind_t kind;
    bool src_cpu = (src->deviceType() == LLAISYS_DEVICE_CPU);
    bool dst_cpu = (device_type == LLAISYS_DEVICE_CPU);
    if (src_cpu && dst_cpu) {
        kind = LLAISYS_MEMCPY_H2H;
    } else if (src_cpu && !dst_cpu) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (!src_cpu && dst_cpu) {
        kind = LLAISYS_MEMCPY_D2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    core::context().setDevice(device_type, target_device);
    core::context().runtime().api()->memcpy_sync(
        result->data(), src->data(), nbytes, kind);

    return result;
    // return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));// 原桩代码。
}

} // namespace llaisys
