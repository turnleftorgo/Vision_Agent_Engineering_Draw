# Qwen 图像模块盲测工具技术栈

## 1. 运行环境

### Python

- 建议版本：Python 3.10 或更高版本；
- 原因：脚本使用 `int | str`、`list[float]` 等现代类型注解语法；
- 程序形态：单文件 CLI；
- 入口：`qwen_module_blind_test.py`。

## 2. 第三方 Python 包

### `openai`

用途：

- 创建 OpenAI-compatible API client；
- 调用 `client.chat.completions.create()`；
- 向本地或远程 Qwen VLM 服务提交文字与图片混合消息；
- 配置 endpoint、API key 和 timeout。

脚本采用新版 client 写法：

```python
from openai import OpenAI

client = OpenAI(base_url=..., api_key=..., timeout=...)
client.chat.completions.create(...)
```

因此需要支持 `OpenAI` client 类和 Chat Completions 接口的 SDK 版本。

### `Pillow`

导入名称为 `PIL`，用途包括：

- 打开输入图片；
- 将图片转换为 RGB；
- 将图片编码成 PNG；
- 读取图像宽高；
- 复制图像并绘制边界框、色块和 ID；
- 保存 `visualization.png`。

使用的主要接口：

```python
from PIL import Image, ImageDraw
```

## 3. Python 标准库

| 模块 | 用途 |
|---|---|
| `argparse` | 定义和解析命令行参数 |
| `base64` | 将 PNG bytes 编码为 data URL |
| `io` | 使用 `BytesIO` 在内存中生成 PNG |
| `json` | 解析模型 JSON、序列化配置与结果 |
| `os` | 读取 `LOCAL_VLM_API_KEY` 环境变量 |
| `re` | 移除 `<think>` 和 Markdown code fence |
| `sys` | 向 stderr 输出错误信息 |
| `time` | 使用高精度计时器记录模型调用耗时 |
| `dataclasses` | 定义 `Detection` 并转换为 dict |
| `pathlib` | 解析路径、创建目录、读写文本文件 |
| `typing` | 为动态 JSON 值提供 `Any` 注解 |
| `__future__.annotations` | 延迟求值类型注解 |

## 4. 外部服务

### OpenAI-compatible Qwen VLM endpoint

脚本不在本地直接加载 Qwen 权重，而是访问一个兼容 OpenAI Chat Completions 的推理服务。

默认配置：

```text
Endpoint: http://127.0.0.1:8001/v1
Model:    Qwen3.8-27B-MLX-8bit
API key:  anything
```

服务必须支持：

- `POST /chat/completions` 对应的 SDK 调用；
- `image_url` 类型的多模态消息；
- Base64 PNG data URL；
- 指定 `max_tokens` 和 `temperature`。

具体推理服务器实现不属于该目录代码的一部分。模型名称必须与服务端注册名称一致。

## 5. 图像与数据格式

### 输入图片

- Pillow 可读取的本地图像格式；
- 运行时统一转换为 RGB；
- 发给模型时统一编码为 PNG data URL；
- 不改变原始宽高。

### 模型响应

- UTF-8 JSON 文本；
- 顶层为 object；
- 包含 `objects` array；
- bbox 使用 `0..1000` 归一化坐标。

### 输出文件

- 文本：UTF-8 `.txt`；
- 结构化结果：UTF-8 JSON，缩进为两个空格，保留非 ASCII 字符；
- 可视化：PNG。

## 6. 当前未使用的技术

为保证盲测结果不被外围算法改变，当前脚本没有使用：

- OpenCV；
- NumPy；
- OCR/Tesseract；
- PyTorch、Transformers 或 MLX Python 推理代码；
- NMS 或目标检测框合并库；
- Web 框架；
- 数据库；
- 并发或异步请求框架。

即使默认模型名称包含 `MLX`，脚本本身也不依赖 `mlx` 包；MLX 属于模型服务端的实现细节。

## 7. 安装建议

该目录目前没有 `requirements.txt`、`pyproject.toml` 或 lockfile，因此不能从仓库确认已验证的精确版本。最小第三方依赖为：

```text
openai
Pillow
```

可在隔离的 Python 环境中安装：

```bash
python -m pip install openai Pillow
```

在建立可复现环境前，应先通过实际推理服务完成兼容性验证，再锁定具体版本，而不应仅根据当前源码猜测版本号。

## 8. 运行示例

```bash
python Vision_IronMan/qwen_module_blind_test.py input.png \
  --output output_module_blind_test \
  --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen3.8-27B-MLX-8bit
```

如需从环境变量提供 API key：

```bash
export LOCAL_VLM_API_KEY="your-key"
python Vision_IronMan/qwen_module_blind_test.py input.png
```
