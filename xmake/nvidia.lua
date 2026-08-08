-- NVIDIA CUDA 设备与算子编译配置
-- 由 xmake.lua 在 has_config("nv-gpu") 时 includes，开启 --nv-gpu=y 后生效；
-- 关闭时本文件不被 includes，CUDA 代码不参与编译。
-- 参考 xmake/cpu.lua 的 target 结构；.cu 文件由 add_rules("cuda") 交由 nvcc 编译。

-- CUDA 设备运行时：Runtime API 与设备资源
target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    -- 算子 kernel 不跨翻译单元调用 device 函数，无需 -rdc=true relocatable device code。
    -- 关闭 rdc 后 .cu 完整编译，不生成需 device-link 的 __cudaRegisterLinkedBinary 符号，
    -- 避免 static target 默认不 device-link 而链入 libllaisys.so 时出现未定义引用。
    set_values("cuda.rdc", false)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- .cu 由 nvcc 编译，-Xcompiler 将 -fPIC 传给底层 host 编译器，生成位置无关代码以链入 libllaisys.so。
        -- -Xcompiler 需跟参数，xmake 自动 flag 检测会误判不支持而忽略，故 {force = true} 强制保留。
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end

    add_cugencodes("native")
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

-- CUDA 算子实现
target("llaisys-ops-nvidia")
    set_kind("static")
    add_rules("cuda")
    add_deps("llaisys-tensor")
    -- 同 device-nvidia：关闭 rdc，避免 static target 不 device-link 导致 __cudaRegisterLinkedBinary 未定义。
    set_values("cuda.rdc", false)
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        -- .cu 由 nvcc 编译，-Xcompiler 将 -fPIC 传给底层 host 编译器，生成位置无关代码以链入 libllaisys.so。
        -- -Xcompiler 需跟参数，xmake 自动 flag 检测会误判不支持而忽略，故 {force = true} 强制保留。
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end

    add_cugencodes("native")
    add_files("../src/ops/*/nvidia/*.cu")
    -- linear 用 cuBLAS（cublasGemmEx）；经 cuda 规则的 utils.inherit.links 继承到 libllaisys.so。
    add_links("cublas")

    on_install(function (target) end)
target_end()

-- xmake.lua 不可改，故在此将 NVIDIA 实现注入主设备/算子 target 的依赖链。
-- xmake 同名 target 配置会合并：此处先声明 add_deps，xmake.lua 后续对同名
-- target 的 set_kind/add_files 等配置会合并进来，最终一并链接进 libllaisys.so。
target("llaisys-device")
    add_deps("llaisys-device-nvidia")
target_end()

target("llaisys-ops")
    add_deps("llaisys-ops-nvidia")
target_end()
