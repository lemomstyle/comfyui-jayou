# 嘉游 SDRAW · Python

ComfyUI 自定义节点组件包。原生 Python 计算，不依赖 Node.js。

## 安装

1. 将目录放到 `ComfyUI/custom_nodes/comfyui-jayou`。
2. 使用 ComfyUI 的 Python 环境安装依赖：

```sh
python -m pip install -r custom_nodes/comfyui-jayou/requirements.txt
```

3. 重启 ComfyUI，搜索「嘉游SDRAW · Python」或「嘉游SDRAW · 对齐 Python」。

沿用 ComfyUI 的 PyTorch；无需重复安装。若已安装满足版本要求的 OpenCV（包括 contrib 版本），请沿用现有版本，避免同时安装多个 OpenCV 发行包。

LightGlue 为可选依赖，见 `requirements-lightglue.txt`。不使用该选项时无需安装；使用时可能首次下载权重。

## 接口

- 主节点：两个 IMAGE 输入、一个可选 MASK 输入；输出 IMAGE、MASK。
- 辅助节点：两个 IMAGE 输入；输出两个 IMAGE、一个 MASK。
- 图片输出尺寸跟随第一个输入。第二个输入与 MASK 支持单张广播。
- 批次逐张计算；不在节点内部启动多任务并发。

## 工作流

`examples/` 含两套工作流与对应 API 文件。需要另外安装 ComfyUI 原生 SAM3 支持、工作流使用的 Crystools 开关以及对应模型。模型名称以当前环境为准，包内不包含权重。

外部节点的 MASK 应与第一个 IMAGE 的坐标一致。两张图先对齐后再分别计算选区，合并并乘以有效重叠范围，再接到主节点。已外部对齐时，主节点的对齐选项应关闭。

本版不调用 Gitee；语义范围由外部 MASK 提供。没有找到对象而返回空 MASK 时，区域补偿会自然跳过，其余处理仍可执行。

## 版本说明

本仓库现已切换为原生 Python 版本，不再包含旧版运行目录。升级后安装依赖并重启 ComfyUI，使用 examples 中的新工作流，或手动替换为带 Python 后缀的节点；旧工作流不会自动切换。若已另外安装 comfyui-jayou-python，请仅保留一份 Python 节点包，避免重复注册。

当前为测试版本。核心基础运算已对照，完整流程不承诺与旧版逐像素一致；运行平台的依赖安装与节点审核需在平台验证。默认关闭额外选项，先比较输出后再用于批量任务。

## 可选输出调整

新增「质感精修」和「质感强度」（0–1，默认 0.5）。默认关闭，在最终输出阶段执行；关闭或强度为 0 时完全跳过。该选项无需额外模型或 API。现有工作流可刷新节点定义，新示例已包含两个参数。MASK 输出含义不变。
