# Qwen 图像模块盲测工具实施说明

## Claw View PDF 扫描工作台（已实现）

### Approach

网页把 PDF 处理视为一个长期运行的 scan run，而不是一次同步 HTTP 请求。上传后，
本地后端先用 Poppler 把 PDF 逐页渲染成 PNG，再按页调用现有
`qwen_module_fai_pipeline.py`。只有 Stage 3 已经生成且验证为有效的
`crop2_refined` 才写入检验表；candidate 创建、恢复中的中间状态和失败结果不会显示。

目前没有可靠来源的 `Project / Revision / Author / Date / Module / Nominal / USL /
LSL / 100% / DC / Points` 全部作为 nullable 字段保留，UI 显示完整列但不填值。未来的
“第一页检验信息抽取层”只负责更新这些字段，不改变 pipeline 的识别协议和网页表格结构。

### Algorithm

```text
PDF upload
  -> Poppler CPU render (page PNGs)
  -> page 1..N sequential orchestration
  -> existing module / FAI / recovery pipeline
  -> valid crop2_refined JSONL event
  -> SQLite record (model bbox + model crop)
  -> table polling update, sorted by numeric FAI
  -> row click selects the PDF page and shows an editable dashed crop box
  -> progress view opens the Stage 1 overview, then a clickable module FAI overview
  -> user edits the dashed bbox directly on the current large drawing
  -> crop original rendered page
  -> append crop revision + set user override
  -> effective result = user result, otherwise model result
```

### Architecture

```text
web_frontend/app/scan/page.tsx
  | POST/GET/PUT
  v
web_backend/app.py (FastAPI)
  |-- ScanStore ----------------------> web_data/claw_view.sqlite3
  |-- process_pdf_run()
        |-- pdftoppm -----------------> web_data/runs/<id>/pages
        |-- qwen_module_fai_pipeline.py per page
              |-- append refined_crop_ready JSONL event
              v
        consume_new_events() ----------> records table

manual crop PUT
  -> crop source page with Pillow
  -> web_data/runs/<id>/user_crops
  -> crop_revisions + records.user_bbox/user_crop_path
```

`records` 同时保存模型版本与用户版本。接口只计算 `effective_bbox` 和
`effective_crop_path`，优先级固定为 user > model，因此人工修改不会破坏原始模型证据，
也不会被后续刷新覆盖。

## 0. IronMan Stage 2 CPU Structural X-Ray（新增）

`qwen_module_fai_pipeline.py` 在生成 `crop2` 的第一次 27B FAI 集群识别前，
先调用 `fai_xray.py` 对每张 `crop1/module` 执行一次纯 CPU 证据编码：

```text
module image
  -> Tesseract proposes FAI/text evidence
  -> OpenCV proposes FAI bubbles, annotation frames, lines and arrowheads
  -> CPU associates A/T/L/H/G/R by spatial continuity
  -> render xray.png and save xray.json
  -> one existing 27B Stage 2 request
  -> crop2
```

颜色表示 `Cxx` 集群归属，字母表示证据类型。未能可靠归属的全量 OCR、Hough
线段和轮廓不会绘制到模型输入。每个 Stage 2 module 固定保存：

- `module_original.png`：未标记的原始 module；
- `xray.png`：给 Qwen 且供人检查的 CPU 结构 X 光图；
- `xray.json`：每个 cluster 的 A/T/L/H/G/R、缺失类别和诊断计数；
- `system_prompt_xray.txt`：实际使用的 X 光图说明。

该阶段不调用 LocateAnything，不增加 GPU 模型调用。X-ray 仅使用 Tesseract、OpenCV
和 NumPy 在 CPU 上生成；Stage 2 仍然只调用一次 Qwen 27B，Stage 3 recovery 保持原流程。

## 1. 实施范围

当前实现位于 `qwen_module_blind_test.py`，是一个单文件命令行程序。本文件描述其 Approach、Algorithm 和函数级 Architecture，不规划目录重构，也不引入未实现的多轮推理、传统视觉检测或评估指标。

## 2. Approach

### 2.1 核心方法

采用 **single-pass full-image blind test**：

1. 将未经切片的完整原图直接交给 Qwen VLM；
2. 要求模型自行发现图像中的独立内容模块；
3. 要求模型以固定 JSON 协议输出 `0..1000` 归一化边界框；
4. 客户端只做严格解析、schema 校验、坐标换算和可视化；
5. 保存原始响应和全部运行参数，确保结果可审计。

这个方法刻意减少外围工程干预，使检测结果能够代表模型在指定 prompt 和完整图输入下的原生能力。

### 2.2 设计原则

#### 原始答案优先

模型响应先原样落盘，再进入解析。解析失败不能覆盖或改写原始回答。

#### 最小后处理

允许的后处理只有：

- 移除常见 `<think>...</think>` 包装；
- 移除包围整个回答的 Markdown JSON fence；
- JSON 解析；
- schema 和坐标合法性校验；
- 从归一化坐标线性换算到像素坐标。

不允许自动修复 JSON、补全字段、NMS、框合并、框缩放或重复请求。

#### 部分结果保留

如果 `objects` 中部分对象非法，保留其他合法检测，同时把非法对象及其错误原因记录到结果中。

#### 运行可追溯

保存 prompt、输入路径、图片尺寸、模型配置、耗时、原始响应、解析响应和最终结果。

## 3. Algorithm

### 3.1 主流程

```text
解析 CLI 参数
      |
      v
检查 max_tokens 和 timeout
      |
      v
解析输入/输出路径并检查图片存在
      |
      v
读取图片并转换为 RGB
      |
      v
保存 prompt 与 run_config
      |
      v
创建 OpenAI-compatible client
      |
      v
完整原图编码为 PNG data URL
      |
      v
单次调用 Qwen Chat Completions
      |
      v
保存 raw_response.txt
      |
      v
移除 think/fence 包装并严格解析 JSON
      |
      +---- 失败 ----> 写 json_parse_error，退出 2
      |
      v
保存 parsed_response.json
      |
      v
检查 objects 顶层 schema
      |
      +---- 失败 ----> 写 schema_error，退出 3
      |
      v
逐个校验 object 和 bbox
      |
      v
归一化坐标转换为像素坐标
      |
      v
写 results.json 并绘制 visualization.png
      |
      +---- 全部合法 ----> 退出 0
      |
      +---- 部分非法 ----> 退出 4
```

任何未在上述分支内处理的异常由 `main()` 捕获，打印错误后退出 `1`。

### 3.2 图片编码算法

输入图片先转换为 RGB，再以 PNG 写入内存缓冲区：

```text
PIL Image
   -> RGB
   -> PNG bytes
   -> Base64 string
   -> data:image/png;base64,...
```

编码过程不修改原图尺寸。

### 3.3 JSON 清理与解析算法

对模型文本依次执行：

1. 删除完整的 `<think>...</think>` 内容；
2. 删除开头的 ` ```json ` 或 ` ``` `；
3. 删除结尾的 ` ``` `；
4. 调用 `json.loads()`；
5. 检查顶层必须是 object。

这里的“清理”只处理输出包装，不修复 JSON 内容。

### 3.4 Detection 校验算法

对 `objects` 中每个元素独立检查：

```text
object 是 dict？
   |
   v
bbox 是长度为 4 的数组？
   |
   v
四个值都是非 bool 数字？
   |
   v
所有坐标位于 0..1000？
   |
   v
x_min < x_max 且 y_min < y_max？
   |
   v
description 是字符串？
   |
   v
id 是允许的标量？
```

任一检查失败时，该对象进入 `invalid_objects`；全部检查通过时，构造不可变的 `Detection`。

### 3.5 坐标换算算法

设原图尺寸为 `W × H`，模型输出归一化框为：

```text
[x1, y1, x2, y2], 每个值范围为 0..1000
```

像素坐标计算为：

```text
pixel_x1 = round(x1 / 1000 × W)
pixel_y1 = round(y1 / 1000 × H)
pixel_x2 = round(x2 / 1000 × W)
pixel_y2 = round(y2 / 1000 × H)
```

换算后不执行 NMS、clamp、padding 或几何修正。

### 3.6 可视化算法

1. 复制 RGB 原图；
2. 根据图像尺寸计算最小线宽；
3. 按预设颜色表循环选择框颜色；
4. 绘制每个合法 `bbox_pixels`；
5. 在框顶部绘制 ID 标签；
6. 如果标签会超出底边，则尝试放到框的上方；
7. 保存为 `visualization.png`。

## 4. Architecture

### 4.1 函数调用总览

```text
__main__
  |
  v
main()
  |
  +-- build_parser()
  |
  +-- parser.parse_args()
  |
  +-- 参数范围检查
  |
  +-- run(args)
        |
        +-- Path / PIL.Image.open()
        |
        +-- write_json(run_config.json)
        |
        +-- OpenAI(...)
        |
        +-- call_qwen(...)
        |     |
        |     +-- image_to_data_url(image)
        |     |
        |     +-- client.chat.completions.create(...)
        |
        +-- parse_model_json(raw_response)
        |     |
        |     +-- remove_thinking_and_fences(raw_response)
        |     |
        |     +-- json.loads(...)
        |
        +-- write_json(parsed_response.json)
        |
        +-- validate_detections(parsed, width, height)
        |     |
        |     +-- Detection(...)
        |
        +-- write_json(results.json)
        |     |
        |     +-- dataclasses.asdict(Detection)
        |
        +-- draw_detections(image, detections)
              |
              +-- PIL.ImageDraw
```

### 4.2 函数职责与连接

| 函数/类型 | 输入 | 输出 | 上游调用者 | 下游依赖 |
|---|---|---|---|---|
| `Detection` | ID、描述、两套坐标 | 不可变检测记录 | `validate_detections()` | `asdict()`、`draw_detections()` |
| `image_to_data_url()` | Pillow Image | PNG Base64 data URL | `call_qwen()` | `io.BytesIO`、`base64`、Pillow |
| `remove_thinking_and_fences()` | 模型原始文本 | 去除包装后的文本 | `parse_model_json()` | `re` |
| `parse_model_json()` | 模型原始文本 | 顶层 JSON object | `run()` | `remove_thinking_and_fences()`、`json.loads()` |
| `validate_detections()` | JSON object、图像宽高 | 合法 Detection 列表、非法对象列表 | `run()` | `Detection` |
| `draw_detections()` | 原图、合法 Detection | 带框 Pillow Image | `run()` | `PIL.ImageDraw` |
| `call_qwen()` | client、模型、图片、生成参数 | 模型响应文本 | `run()` | `image_to_data_url()`、OpenAI client |
| `write_json()` | 文件路径、Python value | JSON 文件 | `run()` | `json.dumps()`、`Path.write_text()` |
| `run()` | CLI namespace | 业务退出码 | `main()` | 上述所有核心函数 |
| `build_parser()` | 无 | ArgumentParser | `main()` | `argparse` |
| `main()` | 命令行环境 | 最终退出码 | `__main__` | `build_parser()`、`run()` |

### 4.3 控制职责

`main()` 负责进程级控制：

- 创建和解析 CLI；
- 校验数值型参数；
- 捕获未处理异常；
- 将异常统一转换为退出码 `1`。

`run()` 负责一次测试运行的业务编排：

- 文件和输出目录处理；
- 配置落盘；
- 模型调用；
- 解析与校验分支；
- 结果与可视化落盘；
- 返回精确业务退出码。

其他函数均保持单一职责，不直接控制完整运行生命周期。

### 4.4 数据流

```text
Image path
   -> PIL Image
   -> PNG data URL
   -> Qwen raw text
   -> parsed dict
   -> valid Detection[] + invalid object[]
   -> results.json + visualization.png
```

旁路审计数据为：

```text
固定 prompt -----------------> system_prompt.txt / user_prompt.txt
CLI 参数 + 图片信息 ---------> run_config.json
Qwen 原始文本 ---------------> raw_response.txt
成功解析的原始 JSON ---------> parsed_response.json
```

## 5. 验证计划

### 5.1 正常路径

- 模型返回空 `objects` 数组；
- 模型返回一个合法对象；
- 模型返回多个合法对象；
- 检查归一化坐标到像素坐标的换算；
- 检查可视化框和 ID。

### 5.2 格式错误

- 非 JSON 文本；
- JSON array 作为顶层；
- 缺少 `objects`；
- `objects` 不是数组；
- 混合合法与非法对象；
- bbox 长度错误、包含 bool、越界或坐标倒置；
- description 不是字符串；
- id 是 object、array 或 null。

### 5.3 运行错误

- 输入文件不存在；
- Pillow 无法读取图片；
- endpoint 无法连接；
- API 请求超时；
- 无权限创建输出目录或写入结果。

## 6. 当前限制与后续边界

当前保持单文件结构有利于盲测逻辑审计。如果未来增加批量评估、ground truth、IoU 统计或多模型实验，应另行设计评估层，避免改变本脚本的单次、完整图、最小后处理基准语义。

---

# `qwen_module_fai_pipeline.py` 双模型流式精修变更

## 7. 变更目标

现有两阶段 pipeline 保留 27B 的前置识别能力，并在每个 `crop2` 生成后立即启动一个独立的精修任务：

```text
Qwen3.8-27B-MLX-8bit
  Stage 1：完整原图 -> module/crop1
  Stage 2：crop1 -> FAI cluster/crop2
                         |
                         | crop2 一旦保存立即提交
                         v
Qwen3.8-27B-MLX-8bit（Recovery role）
  Stage 3：单个 agent 判断完整性，并可同时要求扩张多个方向
```

原始 `crop2` 不覆盖。精修结果保存到 `crop2_refined`，完整决策记录保存到 `stage3_recovery`。

## 8. Approach

### 8.1 双模型职责分离

- 27B 继续负责完整图 module 检测和每个 module 的一次 FAI 集群判断；
- Recovery模型不重新发现或验证 FAI、SPC、描述文字和 leader 起点，只判断箭头实际指向的目标部件是否足够完整；
- Recovery模型只允许输出 `finish` 或 `expand`；`expand` 携带一个或多个 `left/right/up/down`；
- 方向只表示缺失内容位于当前 crop 的哪一侧，不能解释为箭头自身的视觉朝向；
- 模型只做决策，Python 是唯一可以修改 crop 坐标的组件。

### 8.2 流式生产者/消费者

Stage 2 是 crop2 生产者。每次 `save_stage2_crops()` 保存一张新图，立即通过 callback 提交 recovery Future。默认只有一个 crop recovery worker，每个 crop 的每一轮只发出一次 Recovery模型请求。

27B 提交 recovery 后不等待该 crop 完成，继续处理下一个 module。全部 Stage 2 工作提交完毕后，主线程等待剩余 recovery Future 收敛。

### 8.3 单 Agent 多方向决策

单个 Recovery agent 从红圈 FAI 的箭头端点开始追踪目标轮廓。目标完整时输出 `finish`；目标越过洋红框时输出 `expand`，并在 `expand_sides` 中一次列出所有缺失方向，例如 `left + down`。非法响应不修改 crop，下一轮重新观察。

判断规则按通用工程图标注拓扑分为三类，而不是针对某个 FAI 编号或某张图设置特例：

1. direct leader/callout：箭头直接接触物理目标，从接触点追踪目标轮廓；
2. linear/angular/distance dimension：尺寸箭头落在 extension/witness line 或角度射线上，必须继续沿这些线找到被测物理表面；
3. diameter/radius dimension：识别 `Ø`、`⌀`、`R`，继续找到对应圆、孔、圆柱或圆弧特征。

尺寸线、extension line、centerline、leader、箭头和文字均属于 annotation geometry，不等于 physical target geometry。未找到物理目标时禁止 `finish`；标注路径离开观察范围时，沿其离开方向继续扩张。

## 9. Algorithm

### 9.1 全局坐标恢复

Stage 2 的 cluster bbox 是 crop1/module 局部坐标。精修前转换为完整原图坐标：

```text
global_x1 = module_x1 + cluster_x1
global_y1 = module_y1 + cluster_y1
global_x2 = module_x1 + cluster_x2
global_y2 = module_y1 + cluster_y2
```

每一轮观察和最终裁图都从完整原图生成，不能从上一轮已经裁剪的图片继续裁，以免累积缩放和编码损失。

### 9.2 单张带框 Observation

每轮 agent 看见一张观察图。Python 以当前 crop 为中心，从完整原图向上、下各增加当前 crop 高度的 50%，向左、右各增加当前 crop 宽度的 50%，用洋红色矩形框出扩张前的当前 crop，并用红色椭圆圈出当前候选对应的 FAI 圆形标记。只把这一张带框、带红圈的扩展观察图发送给 Recovery模型。

模型只追踪红圈 FAI 自己的标注、leader 和箭头，不追踪其他 FAI。它判断该箭头目标在洋红框内部是否足够完整；如果目标轮廓越过洋红框，则在 `expand_sides` 中输出所有缺失方向。Python 在同一轮同时更新这些边，下一轮围绕新的 crop 再构造观察图。观察图最长边默认限制为 2400 像素；该缩放只影响模型输入，不改变全局坐标或最终输出分辨率。

### 9.3 严格决策协议

agent 必须只返回：

```json
{
  "action": "finish|expand",
  "expand_sides": ["left|right|up|down"],
  "reason": "complete|target_clipped|uncertain",
  "confidence": 0.0
}
```

`finish` 必须对应空的 `expand_sides` 和 `reason=complete`；`expand` 必须至少包含一个方向，并允许同时包含多个不重复方向。字段缺失、多余字段、非法枚举或非法 confidence 时，该轮决策无效，crop 保持不变。

Recovery 请求通过 OpenAI-compatible `response_format.type=json_schema` 发送 `RECOVERY_DECISION_SCHEMA`，在解码阶段约束最终内容只能是上述 JSON 对象。该约束仅添加到 `call_recovery_agent()`，不影响 Stage 1 module detection 或 Stage 2 FAI detection。Recovery 默认 `max_tokens=2048`，为模型完成标注拓扑判断保留足够输出预算。

### 9.4 六轮状态机

```text
current = initial crop2 global bbox

for round in 1..6:
    observation = 当前 crop 四边各扩 50%，框出当前 crop，并红圈标出候选 FAI
    decision = 调用一次 Recovery agent

    decision 无效:
        crop 不变
        下一轮重新观察

    decision.action == finish:
        status = finished
        停止

    decision.action == expand:
        Python 同时向 expand_sides 中的所有方向扩图
        下一轮重新观察

    扩张因原图边界没有产生像素变化:
        status = boundary_exhausted
        停止

六轮结束仍未 finish:
    status = max_rounds_exhausted
```

每轮最多调用一次 Recovery模型。最坏情况为每个 crop2 调用 6 次。

### 9.5 扩图

默认 `step_norm=250`：每个被选中的水平边扩大当前宽度的 25%，每个被选中的垂直边扩大当前高度的 25%。例如 `left + down` 会在同一轮同时修改 `x1` 和 `y2`。所有新坐标 clamp 到完整原图。模型不输出坐标或扩张量。

`finish` 使用物理目标的四边审计：可识别的局部目标不等于完整目标；从箭头落点或被测表面继续追踪同一物理轮廓/结构单元。如果相关物理轮廓、表面、壁、剖面线区域或连接结构接触或穿过洋红框任意一边，并在框外继续，则禁止 `finish`，且必须一次返回所有对应方向。标注线穿框不等同于物理目标穿框。

## 10. Architecture

```text
run()
  |
  +-- stage1_detect_modules()                    # 27B
  |
  +-- stage2_detect_fai_for_module()             # 27B
  |     |
  |     +-- save_stage2_crops()
  |            |
  |            +-- local_box_to_global()
  |            +-- on_crop(RecoveryCandidate)
  |                    |
  |                    +-- recovery_executor.submit()
  |
  +-- recover_crop_candidate()                   # Recovery worker
        |
        +-- build_recovery_observation()
        +-- request_recovery_decision()
        |      |
        |      +-- call_recovery_agent() x 1
        |      +-- parse_recovery_decision()
        |
        +-- finish
        |      or
        +-- expand_crop_sides()
               |
               +-- next observation
```

## 11. 输出与状态

```text
output/
├── crop2/                    # 27B 原始结果
├── crop2_refined/            # 精修图；目标 FAI marker 由红色矩形框标出
└── stage3_recovery/
    └── <candidate_key>/
        ├── initial_crop.png
        ├── round_01/
        │   ├── observation.png
        │   ├── decision_raw.txt
        │   ├── decision.json
        │   └── decision_result.json
        └── result.json
```

只有 `finished` 表示 agent 已确认完整；`boundary_exhausted`、`max_rounds_exhausted` 和 `error` 都保持 `valid=false`，pipeline 返回 partial error 状态，不能把未确认 crop 当作成功结果。

## 12. 验证计划

- `expand_sides=[left,down]` 在同一轮同时扩张左边和下边；
- `finish` 携带非空 `expand_sides` 时判定为非法；
- 非法响应不修改 crop 并触发下一轮；
- crop1 局部坐标正确转换为完整原图坐标；
- left/right/up/down 扩张方向和 clamp 正确；
- 第六轮没有 finish 时返回 `max_rounds_exhausted`；
- 每轮恰好启动一次 agent 请求；
- crop2 保存 callback 在 Stage 2 继续下一个 module 前提交 recovery；
- 原始 crop2 不被精修结果覆盖；
- mock client 集成测试不依赖真实模型服务。


# Claw View 工作台编辑、导出与滚动体验实施计划

## 目标与范围

在既有 PDF 扫描工作台上增加检验记录字段编辑、Excel 导出以及左右面板的宽度和垂直滚动体验。保留已有记录识别、截图重截和进度轮询流程。

## 实施方案

### 1. 更新文档

- 在 `prd.md` 现有内容末尾增加工作台产品需求；仓库不存在 `pre.md`。
- 在本实施计划末尾记录本次功能架构、数据流和验收点。
- 在 `tech_stack.md` 增加 xlsx 生成依赖和浏览器端交互实现说明。

### 2. 可编辑记录

- 后端增加受字段白名单约束的记录更新请求模型和 `PATCH /api/runs/{run_id}/records/{record_id}`。
- SQLite 更新接口只写入被允许的业务字段并更新 `updated_at`；不允许修改 id、序号、候选键、bbox、截图路径等系统/审计字段。
- Project、Revision、Author、Date 同时保存在 scan run 和已有记录中；扫描尚未完成时新增的记录也继承 run 级图纸资料。
- 前端单元格切换为受控输入，失焦或 Enter 保存、Escape 取消；保存状态与失败提示可见。
- API 返回更新后的记录，前端合并至当前任务状态。

### 3. Excel 导出

- 后端新增 `GET /api/runs/{run_id}/export.xlsx`，用 openpyxl 创建工作簿并通过 `StreamingResponse` 下载。
- 列顺序固定为 No.、Project、Revision、Author、Date、Module、Page、FAI、SPC、Description、Nominal、USL、LSL、100%、DC、Points、SPC截图。
- 按记录当前有效截图路径插入 PNG，按列设置宽度、表头填充、边框、冻结首行和筛选。
- 前端在左侧结果区底部放置导出按钮；无任务/无记录时禁用。

### 4. 面板尺寸与滚动

- 用 CSS grid 的可拖拽列尺寸控制左右面板；维持 30%–75% 左栏边界并处理极窄视口。
- 左侧 table 使用固定布局和 `width: 100%`，避免 `max-content` 引入横向滚动；长内容换行/省略，截图列按空间缩放。
- 左面板本身承担垂直滚动，右侧图纸的 `.drawing-scroll` 保持自己的纵向滚动；不允许横向滚动。
- 分隔条设为 hover/focus 的交互热点，通过父级状态控制左右各自 scrollbar 可见；鼠标进入表格或绘图区时滚动归属保持在当前面板。
- `No.` 和表头使用 sticky 定位，确保压缩宽度时仍可快速定位。

## 数据流

```text
表格编辑 -> PATCH record -> SQLite 白名单字段更新 -> 返回记录 -> 更新 React run.records
导出按钮 -> GET export.xlsx -> 数据库读取任务记录 -> openpyxl 写表格和有效 PNG -> 浏览器下载
左右滚轮 -> 面板各自 overflow-y:auto -> 分隔线 hover/focus 显示对应滚动提示
```

## 验收点

- 修改多个字段并刷新，数据库持久化值正确；不能通过接口改写系统字段。
- xlsx 有 17 列，`No.` 第一列，四个元数据列紧随其后；截图对应记录且用户重截优先。
- 通过 Excel 软件可打开导出文件，图片不遮挡文字且行高/列宽可用。
- 拖分隔条向左压缩左栏时无水平滚动条；鼠标滚轮只滚动当前所在面板。
- 鼠标靠近中分隔区域显示垂直滚动条，移开后收起；键盘焦点也能访问面板。

# 精修框交互与局部放大实施补充

## 数据与框呈现

- 使用 `run.records` 作为唯一精修框数据源；记录由现有 Stage 3 `crop2_refined` 消费流程产生。
- 大图按记录的 `effective_bbox` 绘制普通精修框；部件图通过当前页、模块和候选 key 将记录映射到部件局部坐标。
- 移除来自 `stage1_modules` 和所有 `stage2_modules.valid_clusters` 的可见候选 overlay。
- 以 `selectedId` 作为唯一编辑框身份。未选框只可点击，选中框复用现有虚线框拖动/缩放与 crop PUT 保存逻辑。

## 局部放大视图

- 扩展 `InteractiveDrawing`，为每个普通精修框提供右上角放大镜控件。
- 点击后从当前已加载的原始视图像素生成带边界约束的上下文窗口；窗口以 bbox 为中心并留出周边空间，优先覆盖约 2.5 倍 bbox 宽高，触及图像边界时平移窗口而不越界。
- 将焦点窗口映射为局部坐标系，在现有绘图容器内 fit 显示；绘制选中框及其调整控制点。
- 局部坐标的拖动/缩放变更反算为原图坐标后调用现有保存回调，避免新增记录或改动 API 数据协议。
- 放大控件切换为返回全图状态；返回后保留选中记录及已保存 bbox。

## 验收检查

- 大图及部件图只显示记录中存在的精修框。
- 任意时刻仅 `selectedId` 对应的一个框可编辑；点击其他精修框会切换编辑目标。
- 放大镜区域可以点击而不触发框拖动、框选择以外的 overlay 行为。
- 在局部放大视图的四边和图像边角拖动、缩放，保存 bbox 仍落在原始图像坐标范围内。
- 返回全图后记录截图与表格状态保持一致。

## 兼容既有 Stage 1 部件全览

- `viewMode === 'overview'` 继续加载后端已生成的 `overview_url` 整页总览图，保留图中 Stage 1 模块框和标签。
- 部件全览是独立只读视图；精修框唯一来源和单框编辑规则只约束大图与部件编辑视图。
- 在 Stage 1 全览图上按 `stage1_modules[].bbox_pixels` 放置透明点击命中框，并叠加黑底白字 `MODULE N` 标签；点击框或标签后切换到对应模块原图，并仅用该模块的 refined records 绘制可交互框。
- “查看部件全览”与“返回原图”切换时清空局部放大状态，保留当前任务和页码。

## 大图选中框规则

- 大图编辑视图不渲染 records 全量 overlay；只有左侧表格点击明确选中的记录才显示对应黄色编辑框。
- 加入独立的表格选中来源标记，模块图内部选择记录不自动污染整页大图。
- 页码切换、进入部件全览或从部件图选择记录时清除大图选中框状态。

## 任务恢复与实例隔离实施

1. 后端模块初始化时创建进程级 UUID `BACKEND_INSTANCE_ID`；扩展 `runs` 表保存任务所属实例。
2. SQLite 初始化后，将旧实例仍处于 queued/rendering/processing 的任务更新为 `interrupted`，记录服务进程重启原因；已完成/失败历史不改写。
3. `/api/health` 与任务创建响应返回 `backend_instance_id`。
4. 实现统一 run 查验：所有 `/api/runs/{run_id}` 详情、记录编辑、图片资源和 Excel 导出端点均检查任务实例；旧实例任务返回 HTTP 410 Gone。
5. 前端使用版本化 localStorage 键保存 `{run_id, backend_instance_id}`。上传成功后先保存指针，再读取任务详情；新上传成功后替换旧指针。
6. 页面启动先显示恢复态并请求 health。实例相同则获取任务并恢复任务视图、对运行中任务继续轮询；实例不同则清理指针并回到待扫描页。
7. 轮询网络错误不清空任务，采用有上限的退避重试；成功后重置退避。遇到 410、实例变化或任务明确不存在时清理指针与页面数据，禁止自动重新上传 PDF。
8. 防止竞态：启动恢复完成前不显示待扫描页；仅当异步响应对应的 run_id 仍为当前任务时才可更新界面，防止旧响应覆盖新任务。

## 验收检查

- 前端重新挂载恢复相同 run_id，且不会重复 POST `/api/runs`。
- 短时断网期间保留已加载任务/记录，网络恢复后轮询继续。
- 重启 FastAPI 后 health 返回新实例 ID，旧未完成任务成为 interrupted，其详情及子资源 API 返回 410。
- 新任务创建、编辑记录、进度读取、图片读取、导出等路由均不能通过旧 run_id 访问。

## SPC 截图预览与右侧取景实施

1. 缩略图本身成为语义化点击目标；点击时复用 `selectRecord(record, 'page', true)` 同步右侧选框，并记录当前预览 `record.id` 与触发元素。
2. 使用 React Portal 将浮层渲染到 `document.body`。垂直位置以触发元素所在 `<tr>` 的顶边为锚点，动态限制图片最大高度，先完整缩放截图，再让浮层底边落在当前行顶边上方；不回退到当前行下方。横向仍按缩略图位置对齐并限制在视口内，避免表格 overflow 裁切。
3. 浮层直接加载 `record.crop_url` 对应的有效截图，按原比例完整显示并受视口最大宽高限制；沿用 `updated_at` 缓存版本，保存人工重截后自动更新。
4. 监听滚动与窗口尺寸改变重新定位；左侧其他区域点击、按 Esc、换页、换任务及记录不存在时关闭。右侧框编辑期间保留浮层，保存后由 `updated_at` 驱动截图更新。
5. 将 `focusWindow` 的取景系数从 2.5 改为 3.0，使右侧放大视图包含更多上下文；框的原图坐标映射及 `PUT crop` 保存逻辑保持不变。
6. 仅在未进入局部放大模式时渲染框上的放大镜；放大模式保留返回全图按钮。
7. 将右侧 PDF 视图的组件 key 纳入表格选框状态，确保同一记录从普通大图切入选中态时重建可拖动框，而不保留旧的空草稿。

## 验收检查

- 处理中的已有截图无需等待任务完成即可弹出完整预览、选中右侧框。
- 第 1 行、第 8 行及更靠下的记录预览都位于各自表格行上方，不覆盖当前行编辑控件；窗口缩放/滚动后位置仍正确，外部点击和 Esc 可关闭。
- 连续选择不同截图时无旧图闪现，人工重截后预览跟随有效截图更新。
- 框靠近图片边界时取景不会越界，拖动、缩放后的保存坐标仍正确。


# Claw View 审核模式实施补充

## 高层方案

审核模式作为 `scan/page.tsx` 的独立视图状态实现，普通模式的现有检验表和右侧图纸流程保持原样。模式开启时，左侧表格替换为审核面板；右侧当前页绘制所有可审核记录的 FAI 圆圈。所有框编辑继续使用页面像素坐标系，避免引入第二套持久化协议。

## 组件与函数职责

1. **审核模式状态与切换处理**
   - 在扫描页增加 `auditMode` 状态，以及 `hoveredRecordId`、`lockedRecordId` 和当前临时 bbox 状态。
   - `toggleAuditMode()` 负责切换布局、关闭旧的截图浮层/临时焦点，并在退出审核模式时保留扫描任务、页码和已保存记录。
   - `getActiveAuditRecord()` 按“悬停记录优先，否则点击锁定记录”的规则返回当前审核记录；悬停离开时回退到锁定项。

2. **FAI 圆圈 overlay 生成**
   - `buildAuditOverlays(records, pageIndex)` 只选择当前页可审核的 `run.records`，依据每条记录的 `effective_bbox` 计算圆心和标签。
   - 扩展 `Overlay` 支持圆形标记、`onMouseEnter`、`onMouseLeave` 与现有 `onClick`；悬停只切换临时预览，点击更新锁定记录。
   - 将选中圆圈绘制为强调态。保持 Stage 1 全览和现有普通模式的 overlay 规则独立。

3. **审核信息面板**
   - 新建 `AuditPanel`（可放在 `web_frontend/app/scan/` 下），接收当前记录、临时 bbox、指针拖动回调和保存状态。
   - 顶部字段复用 `EditableCell` 与现有字段更新函数；底部左右两图分别呈现完整 FAI 截图和 3 倍局部视图。
   - 未选择记录时显示空状态；没有截图或正在保存时分别显示对应状态，不隐藏错误反馈。

4. **双图实时 bbox 同步**
   - 扩展 `InteractiveDrawing` 的选框移动/缩放逻辑，增加 `onDraftChange(bbox)` 回调，在每次 pointer move 后报告未提交 bbox；仍由组件内部约束 bbox 不越出图像边界。
   - 将现有 pointer 坐标换算逻辑复用到审核视图。若右侧编辑图是局部窗口，先加窗口原点偏移，再换算回整页坐标；临时 bbox 始终用整页像素坐标保存。
   - 完整截图预览按临时 bbox 即时绘制当前框/新 crop 范围；Canvas 2D 用同一 bbox 和现有 `focusWindow()` 生成约 3 倍取景。两图共享状态，避免分别维护造成不同步。
   - pointer move 只更新本地草稿，不请求后端；pointer up 时调用现有 `saveRecord(record, finalBbox)` 一次提交。保存失败时保留草稿及选中记录，成功后以服务端返回记录替换旧记录并通过 `updated_at` 刷新截图缓存。

5. **布局与样式**
   - `scan/page.tsx` 根据 `auditMode` 条件渲染原检验表或 `AuditPanel`，右侧图纸沿用现有页码导航和绘图容器。
   - 在 `globals.css` 增加审核面板字段布局、双图响应式布局、圆圈默认/悬停/锁定样式、选框手柄和保存状态样式。
   - 窄屏下两张截图改为纵向排列；保留可操作的点击锁定，不依赖 hover 才能审核。

## 数据与接口边界

- 不新增 Python 包、npm 包、API 路由或 SQLite 表。
- 字段编辑沿用现有字段更新接口；框保存沿用 `saveRecord` 和现有 crop PUT。
- 截图仍由现有后端从源 PDF 页面重截；模型 bbox 与人工 bbox 的优先级、裁剪历史和 AI 字段提取触发规则不变。
- 现有普通模式仅单条显式选中框的规则继续有效；审核模式的“显示当前页全部 FAI 圆圈”是独立的只读定位层，只有当前审核项的编辑框可拖动或缩放。

## 高层验收检查

- 普通模式进出审核模式后布局及任务状态正确；右侧页码变化后只显示对应页的 FAI 圆圈。
- 鼠标悬停可预览，点击可锁定；鼠标离开后保持点击项，点击另一标记后切换。
- 编辑局部截图框时两张图随临时 bbox 同步变化；仅 pointer up 触发一次保存。
- 保存成功后完整图、局部图和表格缩略图均使用新版本；失败保留草稿且可重试。
- 无截图、空记录、窄窗口、触控/无 hover 输入等状态都有可用界面反馈。

# 审核模式扩展：部件图、手动画框与模型字段并发

本节是在既有审核模式实施思路上的增量，不替换前面已经记录的悬停/锁定和双截图联动方案。

## 高层实现思路

### 1. 锁定记录时切换到部件图

- 在 `scan/page.tsx` 继续用当前页 `run.records` 生成 FAI 圆圈；将圆心从 `effective_bbox` 中心改为优先使用 `marker_bbox` 中心，缺失时才回退。
- 悬停态只更新左侧预览；点击态设置 `lockedAuditId` 并进入部件审核视图。悬停离开后仍恢复锁定项。
- 增加 `AuditComponentView`，读取锁定记录的 `module_index`，复用 `progressPage.stage1_modules[module_index - 1]` 的原始部件图和尺寸。
- 计算唯一选中框时把全页 `effective_bbox` 减去模块原点，转换为部件局部坐标。部件视图的 overlay 列表只传当前锁定记录，避免把同部件的其他 FAI 框绘出。
- 增加 `AuditFaiSelector`，从当前页记录生成 FAI 小圆圈按钮；点击后更新锁定记录、部件图、框和左侧面板。若记录所属模块不同，自动加载新模块图。

### 2. 复用绘图组件支持新建矩形

- 扩展 `InteractiveDrawing` 的 props，增加 `drawMode` 和 `onDrawComplete(box)`。已有框的 move/resize 行为保留；画框模式下在空白图面 pointer down 时记录起点，pointer move 更新临时矩形，pointer up 后 clamp 到图像边界并回调最终框。
- 页面图的框直接是全页像素坐标；部件图的回调通过模块原点转换成全页坐标。新建、保存和模型返回结果均使用全页像素坐标，避免持久化两套坐标格式。
- `AuditPanel` 和 `AuditComponentView` 下方分别提供画框按钮。画框完成后前端立即建立 provisional audit record，显示手绘选区的小圆圈、顶部字段状态、截图和 3× 局部放大预览。

### 3. 手绘框交给 Stage 2 判断 FAI

- 新增后端请求模型 `ManualAuditDetectRequest`，载荷包含 `page_index`、全页 bbox、来源视图/模块信息。
- 新增 `POST /api/runs/{run_id}/pages/{page_index}/audit/detect`，校验任务、页面、框大小与坐标；从源页面按选框取上下文图像，并明确标注用户选区是目标种子。
- 在 `qwen_module_fai_pipeline.py` 抽出可复用的 Stage 2 focused-detection helper，调用现有 `call_vision_once`、`fai.parse_response` 和 `fai.validate_clusters`。用户选区只限制本次识别目标，不改变正常整模块扫描的 Stage 2 行为。
- 将有效 cluster 的整体 bbox 与 marker bbox 从模型输入局部坐标映射回全页坐标；用模型返回的 FAI 编号、SPC 和描述创建正式审核记录，并返回记录及截图 URL。随后复用截图 extraction 队列填写公差、Points 等字段。若模型结果缺失/无效，保留 provisional 框并返回可人工确认的状态。
- 复用 `_queue_extraction_safely` 处理新截图中的字段提取，避免再造另一条字段识别队列。新建记录需有稳定的 `candidate_key`，保证重试不会重复创建记录。

### 4. 字段处理状态和人工编辑保护

- 前端按字段状态给正在自动填写的单元格添加蓝色光效；识别状态在提交手绘框后立即开始，字段提取状态沿用 `extraction_status`。
- 扩展 records 表的 SQLite 迁移，增加已由用户明确编辑的字段集合。`update_record_fields` 每次写字段时同步登记来源。
- 修改 `ScanStore.apply_extraction`：模型结果只允许写入没有被用户锁定的字段；其他字段照常合并。即使用户明确清空字段，后续识别也不覆盖它。
- API 返回每字段的模型/用户/空值状态，前端按该状态决定蓝光、显示值及完成态，不能只依赖整条记录的单一 extraction 状态。

## 计划新增或调整的函数

| 文件 | 函数/组件 | 职责 |
| --- | --- | --- |
| `web_frontend/app/scan/page.tsx` | `InteractiveDrawing` 新增 draw props | 绘制临时框并回调全图坐标 |
| `web_frontend/app/scan/page.tsx` | `AuditComponentView` | 展示锁定记录所属部件和唯一选中框 |
| `web_frontend/app/scan/page.tsx` | `AuditFaiSelector` | 切换当前页锁定 FAI |
| `web_frontend/app/scan/page.tsx` | `submitManualAuditBox` | 建立 provisional UI、提交后端并替换正式结果 |
| `web_backend/app.py` | `detect_manual_audit_record` | 校验请求、调用 focused Stage 2 并返回新记录 |
| `qwen_module_fai_pipeline.py` | `detect_fai_in_user_region` | 复用 Stage 2 模型调用/解析/校验，返回局部坐标 cluster |
| `web_backend/store.py` | `create_manual_record` | 幂等写入手动画框识别出的新记录 |
| `web_backend/store.py` | `update_record_fields` / `apply_extraction` | 记录用户字段来源并阻止模型覆盖 |

## 验收检查

- 点击锁定后右侧为当前记录的 Stage 1 部件图，只有该记录的框可见；切换下方 FAI 按钮后所有视图同步更新。
- 两处画框工具都能创建矩形；部件图选框换算到整页后位置正确，重载后记录框仍对齐。
- 完成手绘后 provisional 框和左右截图立即出现；Stage 2 返回的 FAI 编号、marker bbox 和字段更新同一审核项，不产生重复项。
- 模型定位失败时保留手绘框并进入人工确认状态；请求失败可重试。
- 字段模型处理中有蓝色光效；模型返回前后用户改过的字段均保持用户值，其他字段仍能由模型补全。
- 普通扫描、已有记录的 crop PUT、Stage 3 recovery 和非审核模式视图不受手动识别链路影响。

# 审核模式增量：即时待识别圈与部件内 FAI 列表

本节仅补充两个审核交互细节，沿用上文的 provisional record、圆圈 overlay、模块坐标和 FAI 选择器设计。

## 高层实现

- 在 `submitManualAuditBox()` 创建临时记录并提交网络请求前，继续同步设置 `manualPendingRecord`、`lockedAuditId` 和 `auditDraft`；将该 provisional record 合并进当前页 `auditDisplayRecords`，让 overlay 在本地状态更新后立即渲染，无需等待 Stage 2 响应。
- provisional marker 的中心使用临时 `effective_bbox` 中心，`fai` 未识别时标签显示“—”。等待期间保持黑色未选中样式；服务端返回后，以正式 record 更新当前记录和圆圈，不保留重复临时圈。失败时保留临时圈与选框，支持原有重试操作。
- 页面全图的 audit overlay 可以显示当前页所有正式圆圈及 provisional 圆圈；部件图 overlay 仅加入当前 `module_index` 匹配的 provisional 圈，继续保持部件图只显示当前审核框的约束。
- 将选择器的数据源限制为当前页且 `module_index` 等于锁定记录/当前部件的 records，并包含该模块内的 provisional record。锁定项切换或模块切换时重新计算列表；不同模块的 FAI 不出现在当前部件的列表中。

## 涉及函数与验收

- 调整 `auditDisplayRecords`、`auditOverlays` 和部件临时 overlay 生成逻辑；保持 `submitManualAuditBox()` 的立即本地更新和成功替换顺序。
- 调整 `auditSelectorRecords`：基于锁定记录的 `module_index` 过滤，而不是直接使用当前页全部记录。
- 更新 `AuditFaiSelector` 的未知编号占位显示为“—”。无需增加后端接口、数据库字段、模型调用或第三方依赖。
- 验证手动画框后响应未返回时页面/部件可见黑色“—”圆圈；Stage 2 返回后更新为编号且不重复；失败重试时临时圈保留。
- 验证部件选择器只显示当前模块 FAI，跨模块切换后成员变化；全页图仍显示当前页全部正式 FAI 标记。

# 审核部件图选框同步修复

## 实现

- 在部件图的 `InteractiveDrawing` 调用上启用已有 `syncSelection` prop。切换 FAI 时，父组件传入的新 `selection` 被同步到部件图内部的 `draft`，橙色编辑框随锁定记录更新。
- 保留稳定的 `key={page + module}`，不把选中记录 ID 加入 key，避免切换 FAI 时卸载图片组件。
- 复用 `InteractiveDrawing` 已有同步 effect：若当前正拖动/缩放审核框，则暂不覆盖 `draft`；完成手动画框使用独立的 `drawStartRef` 和 `drawBox`，选框同步不触碰这份手绘草稿。

## 验收

- 在同一部件图下连续点击不同 FAI 圆圈，选中态、左侧字段和部件图上的编辑框都切换到对应记录。
- 切换 FAI 不触发部件图重新加载；手动画框过程中其它待识别请求返回，不清除正在绘制的临时框。
- 当前审核框拖动时，外部 props 更新不会打断本次拖动；拖动结束后仍可切换到其他 FAI。
