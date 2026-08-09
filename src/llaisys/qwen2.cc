// Qwen2 模型 C API 包装。
// 实现 include/llaisys/models/qwen2.h 声明的导出函数，桥接 C 不透明句柄
// LlaisysQwen2Model 与 llaisys::models::Qwen2Model。

#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2.hpp"
#include "../utils.hpp"

// C 不透明句柄的 C++ 实现：内部持有一个 Qwen2Model。
// 在全局命名空间定义，与 qwen2.h 中 struct LlaisysQwen2Model; 前置声明匹配。
struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model model;
    // 转发参数直接就地构造 Qwen2Model（Qwen2Model 不可拷贝/移动，须就地构造）。
    LlaisysQwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
        : model(meta, device, device_id) {}
};

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                           llaisysDeviceType_t device,
                                           int *device_ids, int ndevice) {
    // 校验 device_ids 非空且 ndevice>=1，避免空指针解引用。
    if (meta == nullptr || device_ids == nullptr || ndevice < 1) {
        return nullptr;
    }
    // 捕获 C++ 异常，避免穿过 extern "C" 边界触发 std::terminate 杀死进程。
    try {
        int device_id = device_ids[0];
        return new LlaisysQwen2Model(*meta, device, device_id);
    } catch (const std::exception &) {
        return nullptr;
    }
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    if (model == nullptr) return;
    // 析构兜底捕获，保证 C 边界不抛异常。
    try {
        delete model;
    } catch (const std::exception &) {
        // 析构不应抛异常，忽略以保护 C 边界。
    }
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    if (model == nullptr) return nullptr;
    try {
        return model->model.weights();
    } catch (const std::exception &) {
        return nullptr;
    }
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    // model 为空或内部异常返回 -1（token id 非负，-1 表错误，供 Python 检查）。
    if (model == nullptr) return -1;
    try {
        return model->model.infer(token_ids, ntoken);
    } catch (const std::exception &) {
        return -1;
    }
}

void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
    if (model == nullptr) return;
    try {
        model->model.reset();
    } catch (const std::exception &) {
        // reset 不应抛异常，忽略以保护 C 边界。
    }
}

}
