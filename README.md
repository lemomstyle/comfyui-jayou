# 嘉游 SDRAW

ComfyUI 自定义节点组件包。

## 环境

- 使用 ComfyUI 自带的 Python 与 PyTorch 环境。
- Node.js 22 或更高版本，启动 ComfyUI 的环境中应可执行 `node`。
- 如需指定 Node.js 路径，可设置环境变量 `JAYOU_NODE_BIN`。

## 安装

将本项目放入 `ComfyUI/custom_nodes/comfyui-jayou`，在 ComfyUI 的 Python 环境执行：

```sh
python -m pip install -r custom_nodes/comfyui-jayou/requirements.txt
```

重启 ComfyUI，在节点搜索中输入「嘉游SDRAW」。

本包包含「嘉游SDRAW」和「嘉游SDRAW · 图像对齐」两个节点，无需分别安装。

可选依赖见 `requirements-lightglue.txt`；未使用对应选项时无需安装。首次使用该选项可能需要下载权重。

## 接口

- 主节点：两个 `IMAGE` 输入，可选 `MASK` 输入；输出 `IMAGE` 和 `MASK`。
- 辅助节点：两个 `IMAGE` 输入；输出两个 `IMAGE` 和一个 `MASK`。
- 参数在节点面板内设置。

不要将访问令牌或个人配置提交到仓库。
