// Qwen2 模型 C API 包装（作业3）。
// 实现 include/llaisys/models/qwen2.h 声明的导出函数，桥接 C 不透明句柄
// LlaisysQwen2Model 与 llaisys::models::Qwen2Model。

#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2.hpp"

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
    // V1 仅使用 device_ids[0]（CPU 单设备），ndevice 暂不使用。
    (void)ndevice;
    int device_id = device_ids[0];
    return new LlaisysQwen2Model(*meta, device, device_id);
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return model->model.weights();
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    return model->model.infer(token_ids, ntoken);
}

void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
    model->model.reset();
}

}
