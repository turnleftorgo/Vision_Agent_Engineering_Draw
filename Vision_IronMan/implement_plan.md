# Qwen 图像模块盲测工具实施说明

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
