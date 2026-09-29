# Qwen 图像模块盲测工具技术栈

## Claw View PDF 扫描工作台

- 前端：Next.js 16、React 19、TypeScript、原生 Pointer Events 与 CSS；
- 本地 API：FastAPI、Uvicorn、`python-multipart`；
- PDF 页面渲染：Poppler `pdftoppm`，CPU 执行；
- 本地状态与结果：Python 标准库 `sqlite3`，WAL 模式；
- 人工重截：Pillow 从 PDF 渲染页按像素 bbox 生成 PNG；
- pipeline 联动：子进程 + 追加式 JSONL `refined_crop_ready` 事件；
- 本地页面地址：`http://127.0.0.1:4173/scan`；
- 本地 API 地址：`http://127.0.0.1:8002`。
- 检验截图字段识别：FastAPI 后台线程池 + OpenAI-compatible Chat Completions 多模态 API；默认识别模型为 `Qwen3.6-35B-A3B-MLX-8bit`，默认 endpoint 为 `http://127.0.0.1:8001/v1`。
- 字段识别在 `refined_crop_ready` 截图落盘、或用户保存新的人工裁剪后触发；按截图 SHA-256 幂等，运行结果和状态保存在 SQLite。
- 模型只填入有截图证据且当前为空的行字段。未出现、不可读或有歧义的信息留空；人工已有内容不覆盖。`No.`、Project、Revision、Author、Date 不从单条检验截图推断，Page/Module 使用可用的后端上下文。
- 第 1 页的 Stage 2 Qwen 27B module 识别会注入纯文字图纸元数据规则；不额外发送整页或标题栏图片。Project 从 Metric / Title 表格底行读取；Revision、Author、Date 来自最新修订记录的同一行。候选汇总一致后才填充 run 顶部空字段。
- 行字段 JSON 允许省略未出现的字段（例如 DC）；缺字段按该字段 not_found 处理，其他合法字段仍会写回。

开发启动可在项目根目录执行：

```bash
python3 run_claw_view.py
```

该命令同时启动 FastAPI 和 Next.js 本地开发服务。Qwen endpoint/model 可通过
`CLAW_VIEW_ENDPOINT`、`CLAW_VIEW_MODEL`、`CLAW_VIEW_RECOVERY_ENDPOINT`、
`CLAW_VIEW_RECOVERY_MODEL` 环境变量覆盖。
截图字段识别模型也可通过 `CLAW_VIEW_EXTRACTION_MODEL` 覆盖，服务地址复用 `CLAW_VIEW_ENDPOINT`；默认模型服务名为 `Qwen3.6-35B-A3B-MLX-8bit`。识别服务需要在本地 endpoint 上运行并兼容 Chat Completions 的 `image_url` 多模态格式。

## 1. 运行环境

### Python

- 建议版本：Python 3.10 或更高版本；
- 原因：脚本使用 `int | str`、`list[float]` 等现代类型注解语法；
- 程序形态：单文件 CLI；
- 入口：`qwen_module_blind_test.py`。

## 2. 第三方 Python 包

### `opencv-python` 与 `numpy`（Stage 2 CPU X-Ray）

用途：灰度增强、Canny、Hough 线段/圆检测、轮廓和三角箭头候选提取。所有操作
运行在 CPU，不加载 CUDA、MLX 或其他 GPU 推理框架。

### Tesseract OCR（外部可执行文件）

`fai_xray.py` 通过 `subprocess` 调用系统 `tesseract` 并读取 TSV，不依赖
`pytesseract` Python 包。Tesseract 不可用时 OCR 证据为空，但 OpenCV 路径仍可
运行。

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

### `httpx`（可选 API 代理）

`qwen_module_blind_test V2.py` 可通过 `--proxy` 或 `MODEL_API_PROXY` 为单次
OpenAI-compatible API 请求配置 HTTP、HTTPS 或 SOCKS5 代理。代理仅绑定到该脚本
创建的 HTTP client，不修改系统全局代理；写入 `run_config.json` 前会隐藏代理用户名
和密码。SOCKS5 代理需要安装 `httpx[socks]`。

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


## Claw View 工作台新增功能技术栈

### 字段编辑和持久化

- 前端：React 19 client component，受控单元格输入，调用现有 FastAPI 服务。
- API：FastAPI `PATCH` 记录字段接口，Pydantic 请求校验与字段白名单。
- 存储：SQLite 现有 `records` 表保存行级字段；`runs` 表保存 Project、Revision、Author、Date 默认值，保证扫描中后续产生的记录也继承图纸资料。不新增编辑副本表。

### Excel 文件生成

- `openpyxl` 用于在后端生成 `.xlsx`、设置单元格样式和嵌入 PNG 截图。
- FastAPI `StreamingResponse` 返回内存中的工作簿；文件名按 PDF 名称安全化。
- 新增 Python 依赖：`openpyxl>=3.1`。

### 面板与滚动

- CSS Grid、React pointer events 管理可拖分隔线。
- CSS `overflow-y` 分别管理左右滚动容器；默认弱化/隐藏滚动条，在分隔线 hover/focus 时显示。
- 表格采用固定布局和列宽策略，避免横向滚动；长文本换行，截图按单元格约束缩放。
- 不增加前端 UI 或 Excel 导出依赖。

## 精修框与局部放大实现

- React 19 负责唯一选中框状态及框内的放大镜/返回控件。
- 精修框数据仅由 FastAPI 返回的 `records` 提供；大图使用记录的有效全页 bbox，部件图从记录 bbox 和模块原点换算局部 bbox。
- Canvas 2D 在当前图像像素上提取带上下文的焦点窗口；不新增图片处理依赖或后端截图接口。
- 焦点视图坐标经窗口原点偏移还原至全图坐标后复用现有 crop 更新 API。
- 只读部件全览继续使用后端生成的 Stage 1 overview JPEG；它与精修记录 overlay 分开渲染，不进入精修框编辑/放大交互。
- 全览图上的透明绝对定位按钮复用 `stage1_modules[].bbox_pixels` 作为命中区域，切换到相应的原始 module crop 视图；按钮本身不成为检验框。
- 大图 overlay 不从全部 `records` 生成；用独立表格选中标记控制单条 selected record 的 bbox，避免表格外选择或普通全图浏览触发整页框群。

## 任务恢复与后端实例隔离

- FastAPI 进程以 UUID 生成 `backend_instance_id`；健康检查和任务创建响应提供实例 ID。
- SQLite `runs` 表保存任务所属实例；服务启动时将旧实例未完成任务标记为 `interrupted`。
- 所有任务详情、记录编辑、图片资源和 Excel 导出路由处理前校验 run 所属实例；不匹配时返回 HTTP 410。
- 浏览器 `localStorage` 保存 `{run_id, backend_instance_id}`；重新挂载后先请求 health，再恢复任务和轮询。
- 网络错误只暂停轮询并触发指数退避，不清除任务；明确发现实例变化/旧任务中断后才清理本地任务状态。
- 使用 Python `uuid`、SQLite 迁移及浏览器原生存储/Fetch API，不新增依赖。

## SPC 预览浮层与取景

- 使用 React 19 的 `createPortal` 将截图浮层挂到 `document.body`，避开左侧表格的 overflow 裁剪；通过 DOM `getBoundingClientRect` 定位。
- 浮层直接读取现有 `crop_url` PNG，CSS `object-fit: contain` 保证完整显示，并用 `updated_at` 进行缓存版本控制；不新增后端接口或依赖。
- 右侧继续使用现有 Canvas 和原图坐标系，仅将焦点窗口宽高系数调整为 3.0；局部框拖动/保存协议不变。
- 浮层垂直布局依赖当前 `<tr>` 的 `getBoundingClientRect().top` 和截图区域的动态 `max-height`；图片以 `object-fit: contain` 缩到上方可用空间，避免浮层进入当前行。

## 实时 AI 可视状态

- 复用 Pipeline JSONL 事件通道传送 Crop 2 开始、每轮实际坐标和结束事件；SQLite 独立保存当前活动候选框，进度 API 返回给前端。
- React 原图层以百分比坐标显示 `pointer-events: none` 的动画框；CSS 渐变动画表示活动状态，不参与人工拖框或坐标保存。
- 表格行动画直接由已有 `extraction_status` 驱动；模型结果持久化后状态终结，动画随轮询消失。
- Description 的自动来源限定为截图图像提取流程；Stage 2 原始描述仅供内部定位，不进入记录字段。

## 截图识别结构化输出

- 字段识别仍使用 OpenAI-compatible `/v1/chat/completions`，请求显式设置 `response_format: {"type":"json_object"}`，由 oMLX JSON mode / Guided Grammar 约束为合法 JSON；模型提示词继续定义字段结构和证据规则。
- 服务端和模型环境须支持该 OpenAI 兼容参数；客户端解析器仍负责校验根对象、`fields` 和字段白名单。

## Qwen 35B 独立诊断工具

- `tests/diagnose_qwen35b_extraction.py` 使用 Python 标准库发送模型请求；默认只读本地 SQLite 获取最近一次 35B 失败记录的截图路径，也可用 `--image` 指定图片。
- 对照 production JSON mode、无 `response_format` 和 JSON mode + `enable_thinking=false`；逐次记录 request/response 元数据、原始文本、finish_reason、usage、耗时及解析诊断。
- 输出放入 `test_output/qwen35b_diagnostics/<timestamp>/`，请求日志仅记录图片尺寸、SHA-256 和 base64 已省略标记；不向 SQLite 回写。
- 可在项目根目录通过 `python3 tests/diagnose_qwen35b_extraction.py` 启动；使用 `--help` 查看参数。
