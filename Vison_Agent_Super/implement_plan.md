# FAI Crop Recovery Agent 实施计划

> 目录说明：workspace 中实际存在的目录名是 `Vison_Agent_Super`，不是
> `Vision_Agent_super`。本计划写入现有目录，避免创建两个近似目录。

## 1. 实施目标

以 `Vision_Agent_Ultra/fai_DET_crop_5.py` 的 evidence-first 上游采集器为基础，
加入一个由 Python 调度的 crop recovery agent：Qwen 每轮观察当前 crop 和动态
wider context，只能建议一次扩图、确认完成，或者拒绝假 FAI；Python 是唯一能够
改变 crop 坐标的执行者。

目标流程：

```text
V5 上游证据收集
  OCR + OpenCV -> guided LocateAnything -> A/T/L/H/G/R
                         |
                         v
一次 semantic association（允许 complete=false 或零组件）
                         |
                         v
Python 构造初始最小 crop
                         |
                         v
加载 skills/fai-crop-recovery/SKILL.md
                         |
             最多 6 个 observation turn
                         |
       +-----------------+------------------+
       |                 |                  |
 expand_crop(...)   finish(valid=true)  reject_candidate
       |                 |                  |
 Python 限制并扩图      保存最佳 crop       归档为非 FAI
       |
 从完整原图生成新的动态观察
       +-----------------------> 下一轮
```

`FAI_DET_CROP_4.py` 只作为结果对照基准，不修改、不导入、不调用。当前 Ultra V5
也保持不动；正式实现放到：

```text
Vison_Agent_Super/
├── fai_DET_crop_5.py
├── implement_plan.md
├── recovery/
│   ├── __init__.py
│   ├── models.py
│   ├── skill_loader.py
│   ├── protocol.py
│   ├── crop_tool.py
│   ├── observation.py
│   ├── agents.py
│   └── engine.py
├── skills/
│   └── fai-crop-recovery/
│       ├── SKILL.md
│       └── agents/
│           └── openai.yaml
└── tests/
    ├── test_protocol.py
    ├── test_crop_tool.py
    ├── test_observation.py
    └── test_recovery_engine.py
```

## 2. 明确范围

本轮实现包含：

- 保留 Ultra V5 已有的 OCR/OpenCV-first 上游证据收集。
- 严格的 recovery action JSON Schema。
- `observe -> decide -> expand -> observe` 状态机。
- `max_turn=6`，其中第 6 轮只做最终判断，不允许继续扩图。
- 格式重试与几何 turn 分离。
- 动态 wider context。
- 唯一可变工具 `expand_crop`，独立实现于 `recovery/crop_tool.py`。
- Skill 决策层、协议校验层、状态机和受信任工具执行层相互分离。
- 最多三个只读 subagent 和一个主 agent 仲裁。
- semantic mapping 不完整或零组件时仍进入 recovery。
- 完整的每轮图片、动作、坐标、模型元数据和最终状态记录。

暂不包含：

- 不再次重写 V5 的 annotation/arrow/target 上游提案算法。
- 不让多个 agent 各自维护或修改 crop。
- 不增加缩图工具。
- 不让 Qwen 直接写文件、执行 Python 或修改坐标。
- 不以盲目四向扩张作为 JSON/协议错误的兜底。
- 不修改 V4 基准结果。

## 3. 为什么不能继续调用 V3 `process_image()`

当前 Ultra V5 通过替换 `core.create_candidate_evidence`，然后调用 V3
`process_image()`。这种接缝只能替换上游 collector；V3 的下游固定为最多两次
validation，而且 `build_validation_context()` 固定使用初始 `roi_bbox`。

Super 版本应：

1. 继续复用 V3 的基础数据类型、图像编码、FAI 初筛等稳定底层函数。
2. 保留并复制 Ultra V5 的 evidence-first collector 到 Super 脚本。
3. 在 Super 脚本中实现自己的 `process_image_v5()`。
4. 不再调用 `core.process_image()`。
5. semantic、初始 crop、recovery、落盘和 manifest 均由 Super 调度器显式控制。

这样才能保证格式重试不计 turn、动态 context、subagent 仲裁和历史最佳 crop 都由
同一个 Python 状态机管理。

Super 主脚本只负责原有图像检测 pipeline、依赖装配和 CLI。Recovery 内部逻辑不
继续堆进单文件，避免模型调用、动作校验和坐标修改共享隐式状态。

## 4. Recovery Skill

### 4.1 Skill 文件

创建 `skills/fai-crop-recovery/SKILL.md`。它是 Python 读取并注入 Qwen system
message 的 prompt bundle，不依赖 LM Studio 自动发现。

Frontmatter 只包含：

```yaml
---
name: fai-crop-recovery
description: Recover and validate incomplete FAI engineering-drawing crops by tracing annotation text, leaders, terminal arrowheads, and touched target features. Use when a candidate crop may be incomplete, semantically unmapped, clipped, contaminated by neighboring annotations, or not a true FAI marker.
---
```

正文采用低自由度 SOP，控制在约 80 行以内：

1. 先验证红框是否真的是含 `FAI + 编号` 的 marker。
2. SPC、普通孔、普通圆号、尺寸文字不能被当作 FAI。
3. 检查参数、公差、feature-control frame 和说明文字。
4. 从相关文字开始，逐段追踪所有 leader。
5. 检查每条 leader 的 terminal arrowhead。
6. 检查箭头真正接触的局部零件或表面。
7. 检查 crop 四边是否截断相关文字、线、箭头或 target。
8. 缺失内容时只建议一次 `expand_crop`，且只扩缺失方向。
9. 完整时只返回 `finish`。
10. 红框不是 FAI 时只返回 `reject_candidate`。
11. 不确定时不得假装完整，也不得因为 semantic mapping 为空就拒绝真 FAI。
12. 相邻 FAI/SPC 或无关 annotation 不是本组证据，应避免为它们扩图。

### 4.2 Skill 加载器

实现：

```python
load_recovery_skill(path: Path) -> RecoverySkill
```

行为：

- 程序启动时读取一次，而不是每轮重复读盘。
- 验证文件存在、frontmatter 的 `name` 正确、正文非空。
- 分离 frontmatter 与正文，只把正文和必要身份信息注入 system prompt。
- 计算 SHA-256，写入 manifest，确保一次运行使用同一个 Skill 版本。
- verification 开启但 Skill 不可用时直接报错；不得静默退回松散 prompt。

### 4.3 Skill 目录为什么不包含工具脚本

`skills/fai-crop-recovery/` 首版不建立 `scripts/`。其中 `SKILL.md` 是 SOP，
`agents/openai.yaml` 只是标准 Skill 的展示/触发元数据，不包含可执行逻辑：

- Skill 是供 Qwen 阅读的决策 SOP，不是 Python 执行环境。
- Qwen 不能直接运行或修改 crop 工具。
- `expand_crop` 需要接受全局面积、原图边界、历史 bbox 和 turn 状态约束，属于应用
  的受信任执行层。
- 把执行器放进 Skill 会混淆“模型建议动作”和“Python 获准执行动作”的安全边界。

因此工具固定放在 `recovery/crop_tool.py`，由 `engine.py` 在动作通过
`protocol.py` 校验后调用。未来即使替换 Skill 内容，也不能绕过工具限制。

## 5. 严格动作协议

### 5.1 统一 Action Schema

为兼容本地 OpenAI-compatible endpoint，优先使用一个扁平、无 `oneOf` 的严格
Schema。所有字段必须出现，`additionalProperties=false`：

```json
{
  "action": "expand_crop",
  "valid": false,
  "candidate_valid": true,
  "missing": ["terminal_arrowhead", "target_feature"],
  "arguments": {
    "left_norm": 250,
    "top_norm": 0,
    "right_norm": 0,
    "bottom_norm": 0
  },
  "confidence": 0.82
}
```

字段约束：

- `action`: `expand_crop | finish | reject_candidate`。
- `valid`: 仅 `finish` 时允许为 `true`。
- `candidate_valid`: 仅确认红框是真 FAI 时为 `true`。
- `missing`: 字符串数组，限制项目数和单项长度。
- 四个 `*_norm`: 整数，范围 `0..500`。
- `confidence`: 数值，范围 `0..1`。

Python 再执行跨字段校验：

- `expand_crop`: `candidate_valid=true`、`valid=false`，至少一个方向大于零。
- `finish`: `candidate_valid=true`、`valid=true`、`missing=[]`、四方向全零。
- `reject_candidate`: `candidate_valid=false`、`valid=false`、四方向全零。
- `reject_candidate` 只能表示 marker 不是 FAI，不能表示暂时找不到 target。

请求使用：

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "fai_crop_recovery_action_v1",
        "strict": True,
        "schema": RECOVERY_ACTION_SCHEMA,
    },
}
```

Qwen 可以继续启用 thinking；Python 只解析最终 `message.content` 中的动作 JSON。
若 endpoint 返回独立 `reasoning_content`，只在 debug 模式落盘，不拼回下一轮上下文。

### 5.2 Tool 语义

对 Qwen 暴露的可变能力只有 `expand_crop`。默认采用严格 action envelope，由
Python 把 `action=expand_crop` 分发给同名工具执行；`finish` 和
`reject_candidate` 是终止控制动作，不修改图像。

如果 LM Studio 当前模型确认支持原生 `tool_calls`，可以增加 transport adapter，
但原生调用和 action JSON 必须先归一化为同一个 `RecoveryAction`，不能维护两套
状态机。首版以 `response_format=json_schema` 为主，优先解决输出过长和无 JSON。

## 6. Python 状态与模块边界

### 6.1 文件职责

| 文件 | 职责 | 明确禁止 |
|---|---|---|
| `fai_DET_crop_5.py` | CLI、FAI 检测、V5 上游证据收集、调用 recovery engine、保存总 manifest | 不直接执行 recovery 扩图循环 |
| `recovery/models.py` | `RecoveryConfig`、`RecoveryAction`、`CropBox`、turn/result 数据类 | 不调用模型、不读写图片 |
| `recovery/skill_loader.py` | 读取、验证并 hash `SKILL.md` | 不执行 action |
| `recovery/protocol.py` | JSON Schema、响应解析、跨字段动作校验 | 不修改 bbox |
| `recovery/crop_tool.py` | 唯一 `expand_crop` 坐标执行器 | 不调用 Qwen、不写文件、不决定循环 |
| `recovery/observation.py` | 从完整原图生成动态 context/crop 图片 | 不采纳模型动作 |
| `recovery/agents.py` | 主 agent、三个只读 subagent、coordinator 的模型请求 | 不修改 bbox |
| `recovery/engine.py` | 单状态所有者，驱动 observe/decide/act、历史和终止 | 不自行绕过 protocol 或 tool |

依赖方向必须单向：

```text
fai_DET_crop_5.py
        |
        v
recovery/engine.py
   ├── agents.py
   ├── protocol.py
   ├── observation.py
   ├── crop_tool.py
   └── skill_loader.py

以上 recovery 模块只能依赖共同基础 recovery/models.py；
任何叶子模块都不能反向导入 engine.py 或主脚本。
```

`agents.py` 只能产生建议，`protocol.py` 只能判定建议是否合法，只有
`crop_tool.py` 能计算新 bbox，只有 `engine.py` 能提交这次状态变化。

### 6.2 数据结构

新增数据结构：

```python
@dataclass(frozen=True)
class RecoveryConfig:
    max_turns: int = 6
    max_format_retries: int = 2
    max_subagents: int = 3
    max_content_bytes: int = 4096
    max_direction_norm: int = 500
    max_crop_area_ratio: float = 0.45
    max_crop_growth: float = 12.0

@dataclass
class RecoveryObservation:
    turn: int
    crop_bbox: CropBox
    context_bbox: CropBox
    crop_image: Image.Image
    context_image: Image.Image
    boundary_locked: dict[str, bool]

@dataclass
class RecoveryTurn:
    turn: int
    format_attempts: list[dict]
    action: dict | None
    requested_expansion: dict
    applied_expansion: dict
    crop_before: list[int]
    crop_after: list[int]
    deterministic_score: float
    used_subagents: bool

@dataclass
class RecoveryResult:
    status: str
    final_crop: CropBox
    best_crop: CropBox
    turns: list[RecoveryTurn]
    rejected: bool
    valid: bool
```

### 6.3 核心函数边界

核心函数按模块归属：

```text
skill_loader.py: load_recovery_skill()
observation.py:  build_dynamic_observation()
agents.py:       build_recovery_messages()
agents.py:       request_strict_action()
protocol.py:     validate_recovery_action()
crop_tool.py:    expand_crop()
engine.py:       score_crop_state()
engine.py:       should_escalate_to_subagents()
agents.py:       request_subagent_opinions()
agents.py:       request_coordinator_decision()
engine.py:       deterministic_emergency_arbitration()
engine.py:       run_crop_recovery()
fai_DET_crop_5.py: process_image_v5()
```

模型请求、JSON 校验、坐标执行、图片生成和结果落盘必须分开，避免某个模型失败时
破坏 crop 状态。

## 7. Agentic Loop 状态机

### 7.1 Turn 定义

- 一个 turn 表示“针对一个不变 bbox 生成一次有效观察并得到一个合法几何/终止决定”。
- JSON 格式重试不改变 bbox，因此不消耗 turn。
- subagent 对同一观察的并行建议不消耗额外 turn。
- `max_turns=6` 表示最多观察六次。
- 第 1 至 5 轮可以执行 `expand_crop`。
- 第 6 轮只接受 `finish` 或 `reject_candidate`。
- 第 6 轮仍返回 expand 时，不执行，结束为 `max_turns_exhausted` 并保存历史最佳 crop。

### 7.2 每轮算法

```text
for turn in 1..max_turns:
    从 full_image + current_crop 创建不可变 observation
    请求主 Qwen 严格动作

    if 协议错误:
        对同一 observation 最多格式重试 2 次

    if 重试仍失败或满足冲突/卡住条件:
        三个只读 subagent 读取同一 observation
        主 coordinator 汇总为一个严格动作
        coordinator 仍失败时使用确定性仲裁或 protocol_failed

    if reject_candidate:
        归档并停止

    if finish(valid=true):
        标记验证通过，保存并停止

    if expand_crop and turn < max_turns:
        Python 校验并执行一次扩图
        写入 before/requested/applied/after
        如果 bbox 有变化，进入下一轮
        如果 bbox 无变化，增加 stuck_count，不做默认四向扩张

到达上限:
    选择历史最佳 crop
    标记 max_turns_exhausted 或 boundary_exhausted
```

终止状态必须区分：

- `validated`
- `validated_from_incomplete_mapping`
- `rejected_not_fai`
- `max_turns_exhausted`
- `boundary_exhausted`
- `protocol_failed`
- `verification_skipped`

## 8. 格式故障处理

以下任一条件判为协议故障：

- `finish_reason == "length"`。
- 最终 content 超过 `max_content_bytes=4096`。
- content 为空。
- JSON 解析失败。
- Schema 字段缺失、类型错误、额外字段或越界。
- action 与 `valid/candidate_valid/arguments` 的跨字段关系非法。

处理顺序：

1. 保存本次 raw content、finish reason、content/reasoning 字符数和 token usage。
2. 保持 observation 和 bbox 不变。
3. 使用更短的 repair prompt 重试，最多 `max_format_retries=2`。
4. repair prompt 只包含校验错误摘要，绝不回填整段超长输出。
5. 仍失败才触发只读 subagent。
6. 所有 agent 都无法给出合法动作时标记 `protocol_failed`。
7. 协议错误绝不转换成默认四向 8% 扩张。

## 9. 三个只读 Subagent

Python 通过三次互相隔离的 Qwen 请求实现 subagent。它们没有工具、没有对话历史、
不能写入 crop，只读取同一个 `RecoveryObservation`。

角色：

| Agent | 只读职责 | 重点输出 |
|---|---|---|
| A: Completeness auditor | 检查 FAI/SPC、文字、公差、leader、arrowhead、target 是否齐全 | missing 与是否完整 |
| B: Geometry tracer | 沿 leader 逐段判断缺失端点和扩图方向 | 四方向 expansion |
| C: Adversarial verifier | 检查假 FAI、邻组污染和 crop 过大 | reject/contamination 风险 |

每个 subagent 使用同一个严格 recommendation schema：

```json
{
  "candidate_valid": true,
  "crop_complete": false,
  "missing": ["terminal_arrowhead", "target_feature"],
  "expand": {"left": 250, "top": 0, "right": 0, "bottom": 0},
  "confidence": 0.82
}
```

调度规则：

- `max_subagents=0..3`，默认 3。
- 使用线程池并行请求，三个请求共享图片字节但不共享 message history。
- 三个结果全部落盘，任何一个失败不影响其他两个。
- 主 coordinator 获得三份 JSON 摘要和同一观察图，再输出统一 RecoveryAction。
- coordinator 只能建议，最终仍由 Python 调用一次 `expand_crop`。

触发 subagent 的条件：

- 主 agent 的初次请求加两次格式重试仍失败。
- 连续两轮没有有效 bbox 变化。
- 连续两轮扩张方向互相反转或视觉判断冲突。
- 主 agent confidence 低于 0.60 且仍报告关键组件缺失。
- 主 agent 想 `finish`，但确定性边界检查发现 leader/文字明显触边。

确定性紧急仲裁只在 coordinator 也失败时使用：

- 至少两个高置信意见认为不是 FAI，才允许 reject。
- 至少两个意见认为 complete，且 adversarial agent 未高置信 reject，才允许 finish。
- 否则对扩张建议按方向取置信度加权中位数，再交给扩图限制器。
- 没有足够合法意见时返回 `protocol_failed`，不猜测扩图方向。

## 10. 独立的唯一扩图工具

工具实现固定放在 `recovery/crop_tool.py`。它是普通、可单元测试的 Python 模块，
不是让 Qwen 直接执行的命令行脚本，也不放在 Skill 的 `scripts/` 下。

实现：

```python
expand_crop(
    current: CropBox,
    action: RecoveryAction,
    full_size: tuple[int, int],
    initial_area: float,
    config: RecoveryConfig,
) -> ExpansionResult
```

纯函数契约：相同输入必须得到相同输出；函数不读取图片、不访问模型、不写文件、
不修改传入对象，也不持有全局 crop 状态。`ExpansionResult` 同时返回新 bbox、实际
应用的四方向像素、边界锁、面积限制原因和 `changed`。

Python 负责：

- 将 norm 转成当前 crop 宽高对应的像素。
- 单方向每轮最多扩 50%，即 norm 最大 500。
- 坐标 clamp 到原图边界。
- 已到达边界的方向设为 locked，后续请求该方向时应用值为零。
- 最大面积同时受两个上限约束：原图面积的 45%，以及初始 crop 面积的 12 倍；实际硬上限取二者较小值。
- 初始 crop 过小的问题应由有最小尺寸保证的 bootstrap crop 解决，不能通过绕过面积上限解决。
- 超出面积上限时按比例缩放本轮 delta，而不是直接越界。
- 整数化后 bbox 未变化时返回 `changed=false`。
- 完整记录 requested norm、clamped norm、pixel delta、边界锁和面积变化。

注意：面积上限应实现为可配置策略并覆盖单元测试。实际允许面积明确为
`min(full_image_area * max_crop_area_ratio, initial_crop_area *
max_crop_growth)`；bootstrap crop 自身必须先满足最小观察尺寸。

调用链唯一允许为：

```text
Qwen action JSON
    -> protocol.validate_recovery_action()
    -> engine 检查 turn/当前状态
    -> crop_tool.expand_crop()
    -> engine 提交新 bbox 并记录 history
```

任何 prompt、Skill、subagent 或 coordinator 都不能绕过这条调用链直接构造下一轮
bbox。

## 11. 动态 Wider Context

废弃 recovery 阶段对固定 `roi_bbox` 的依赖。每轮直接从完整原图生成：

1. `crop_image`：当前 bbox 的干净原图。
2. `context_image`：当前 bbox 四周各扩当前宽/高的 50%，并 clamp 到完整原图。
3. 在 context 上画：
   - 当前 crop：MAGENTA。
   - selected FAI：RED。
   - 已锁定的原图边界：灰色标识。
   - 上一轮实际扩张方向：可选黄色箭头，仅用于调试版 context。
4. 高分辨率 crop 和 context 分别限制长边，禁止把整页压缩成无法读字的小图。
5. 每轮重新从 `full_image` 裁取，绝不在上一轮缩放图上继续裁切。

动态 context 的 bbox 和缩放比例必须写入每轮 JSON，使模型动作能够追溯回完整原图。

## 12. Semantic Mapping 的 Recovery 门控

新的进入规则：

```text
候选图像损坏 / F0 不存在
    -> fatal，不能进入 recovery

semantic JSON 合法，complete=true
    -> 从所选组件构建 minimum crop，再进入 recovery 验证

semantic JSON 合法，complete=false
    -> 带 semantic_missing 进入 recovery

semantic JSON 合法，但没有选择任何 A/T/L/H/G/R
    -> 从 F0 + V5 上游最近证据构造 bootstrap crop，进入 recovery

semantic JSON 格式失败或 length
    -> 记录 mapping failure；仍从 F0 + 上游证据构造 bootstrap crop，进入 recovery
```

初始 crop 的优先级：

1. Qwen 合法选择的组件最小公约图。
2. `F0 + 最近 annotation/OCR + 相连 leader + terminal/target context` 的确定性 union。
3. 只剩 F0 时，使用以 marker 为中心的受限 bootstrap context。

因此 FAI 13、FAI 5 一类 semantic 零选择候选不会在 recovery 之前被跳过。

Recovery prompt 中的 semantic result 只能作为 hypothesis，不能当作 ground truth。

## 13. 历史最佳 Crop

每轮执行扩图前后都保留 bbox 和图片引用。由于没有 shrink 工具，Python 用确定性
评分维护 `best_crop`：

- F0 是否完整位于 crop 内。
- 已选或 bootstrap evidence 被覆盖的比例。
- leader/文字/target evidence 是否触边。
- crop 边界是否穿过明显的 OpenCV 线段或 OCR box。
- 无关候选 FAI/SPC 的数量。
- 面积惩罚。
- 模型 confidence 只作为次要项，不能覆盖几何硬约束。

一旦 `finish(valid=true)`，该轮 crop 直接成为最终 crop。未验证通过而到达上限时，
返回历史最高分图片，但状态必须保持 `*_exhausted`，文件名不得伪装成 validated。

## 14. Prompt 组织

每次主 agent 请求分为：

```text
system:
  固定身份和禁止事项
  + SKILL.md 正文
  + 严格动作约束

user text:
  turn/max_turn
  当前 full-image bbox
  已锁定边界
  semantic_missing（仅假设）
  前几轮动作的短摘要
  最后一轮是否禁止 expand

images:
  IMAGE 1: 动态 wider context
  IMAGE 2: 当前 clean crop
```

历史只传结构化摘要，不重复发送旧图和旧 reasoning，防止上下文线性膨胀。

Subagent 使用同一 Skill 正文，但追加各自唯一角色约束。Coordinator 只接收三份
结构化建议，不接收 subagent reasoning。

## 15. CLI 参数

新增：

```text
--max-turns 6
--max-format-retries 2
--max-subagents 3
--recovery-max-tokens 8192
--max-content-bytes 4096
--max-direction-norm 500
--max-crop-area-ratio 0.45
--max-crop-growth 12.0
--recovery-skill skills/fai-crop-recovery/SKILL.md
--subagent-confidence-threshold 0.60
```

保留：

- `--no-verify`：完全跳过 recovery。
- `--debug`：保存每轮 observation、reasoning（如 endpoint 单独提供）、协议元数据。
- 原有 endpoint、model、tile、candidate 和 Tesseract 参数。

移除或废弃 `--validation-attempts`，避免它与 `--max-turns` 表示两个竞争循环。

参数校验：

- `max_turns >= 1`。
- `max_format_retries >= 0`。
- `0 <= max_subagents <= 3`。
- `1024 <= recovery_max_tokens`。
- `512 <= max_content_bytes <= 16384`。
- `1 <= max_direction_norm <= 500`。
- area ratio 在 `(0, 1]`。

## 16. 输出与可审计性

每个 candidate 建议输出：

```text
output_super/
├── crops/
│   ├── FAIxxx_000_validated.png
│   └── FAIxxx_000_best_unvalidated.png
├── rejected/
│   └── Candidate000_not_fai.png
├── failed/
│   └── Candidate000_protocol_failed.png
├── recovery/
│   └── candidate_000/
│       ├── skill_snapshot.md
│       ├── turn_01_context.png
│       ├── turn_01_crop.png
│       ├── turn_01_action.json
│       ├── turn_01_request_meta.json
│       ├── turn_01_format_retry_01.txt
│       ├── turn_02_context.png
│       ├── turn_02_crop.png
│       ├── turn_02_subagent_a.json
│       ├── turn_02_subagent_b.json
│       ├── turn_02_subagent_c.json
│       └── history.json
├── raw_responses/
├── debug/
└── results.json
```

`results.json` 每个 candidate 至少记录：

- 初始 mapping 及其 protocol 状态。
- bootstrap/minimum crop。
- Skill 路径和 SHA-256。
- 所有 turn 的 bbox before/after。
- requested 与 applied expansion。
- 格式重试次数。
- 是否触发 subagent、各角色状态和 coordinator 结果。
- boundary locks、面积限制和停止原因。
- best crop、final crop、validated/rejected 状态。

## 17. 实施阶段

### Phase A：建立 Super 独立入口

- 从 Ultra V5 建立 Super 工作副本。
- 保留 evidence-first collector。
- 实现独立 `process_image_v5()`，停止调用 V3 `process_image()`。
- 建立 `recovery/` package 和 `tests/`，先固定模块依赖方向。
- 保证 Ultra V5 和 V4 文件零改动。

### Phase B：数据模型与独立 Crop Tool

- 在 `models.py` 定义不可变动作、bbox、配置和结果类型。
- 在 `crop_tool.py` 单独实现纯函数 `expand_crop()`。
- 先完成 norm、边界、面积、无变化和不可变输入测试。
- 工具测试全部通过后，才允许接入 agent loop。

### Phase C：Skill 与严格协议

- 用标准 skill 初始化流程创建 `fai-crop-recovery`。
- 写低自由度 `SKILL.md`。
- 在 `skill_loader.py` 实现 Skill loader 和 hash。
- 在 `protocol.py` 定义 action/recommendation JSON Schema。
- 在 `protocol.py` 实现严格解析和跨字段校验。

### Phase D：单主 Agent Loop

- 在 `observation.py` 实现动态 observation。
- 在 `agents.py` 实现主 Qwen 请求和格式重试，不计 turn。
- 在 `engine.py` 实现六轮状态机和最终轮规则。
- `engine.py` 只能通过 `crop_tool.expand_crop()` 改变 bbox。
- 允许 incomplete/empty/failed semantic mapping 进入 recovery。

### Phase E：历史最佳与防失控

- 由 `crop_tool.py` 返回边界锁、面积上限和无变化结果。
- 在 `engine.py` 实现 deterministic crop score。
- 由 `engine.py` 保存每轮状态和历史最佳 crop。

### Phase F：只读 Subagents

- 在 `agents.py` 实现 A/B/C 三个独立 role prompt。
- 并行请求，限制最多三个。
- 实现 coordinator 严格动作。
- 在 `engine.py` 实现 deterministic emergency arbitration。

### Phase G：输出、回归和实图评测

- 完善文件命名和 manifest。
- 对比 V4、Ultra V5 和 Super V5。
- 使用 `p2.png` 的已知 failed candidates 做 targeted evaluation。
- 再用 `p5.png`、`p9.png`、`p11.png` 检查泛化。

## 18. 测试计划

### 18.1 无模型单元测试

- `crop_tool.py` 不导入 OpenAI client，也不产生文件系统副作用。
- 相同 `expand_crop()` 输入产生相同的 `ExpansionResult`。
- 输入 bbox 和 action 在调用后保持不变。
- Action Schema：三种合法动作与所有非法组合。
- 格式失败不改变 bbox、不增加 geometry turn。
- norm 到 pixel 的换算。
- 单方向 50% clamp。
- 四个原图边界 lock。
- 面积上限缩放。
- 无变化检测。
- 第六轮拒绝 expand。
- best crop 回选。
- incomplete/empty mapping 的 bootstrap crop。
- 动态 context 在 crop 超出初始 ROI 后仍覆盖新区域。

### 18.2 Fake client 状态机测试

按预设响应序列模拟：

1. `expand right -> expand right -> finish`。
2. `length -> invalid JSON -> valid expand`，确认仍是同一 turn。
3. 三次协议失败后触发三个 subagent。
4. subagent 方向冲突后 coordinator 只执行一次扩图。
5. 所有请求失败后 `protocol_failed`，确认没有默认扩图。
6. marker 为假 FAI，返回 `reject_candidate`。
7. 第六轮仍要求扩图，返回 `max_turns_exhausted`。

### 18.3 实图验收

重点回归：

- Candidate 3：能够执行此前第三轮提出但未执行的向右扩张。
- Candidate 7、16：允许多段 leader 的多轮追踪。
- Candidate 9、11、12、22：长输出先格式重试，不能浪费 geometry turn。
- Candidate 1、5：semantic 零组件仍进入 recovery。
- 假 FAI、普通孔、`0.40` 文字候选能被 adversarial verifier 拒绝。

对每个候选记录：成功率、turn 数、格式重试数、subagent 使用次数、最终面积、
是否包含 FAI/annotation/leader/arrow/target，以及与人工期望方向是否一致。

## 19. 验收标准

实现完成需同时满足：

- V4 和 Ultra V5 的文件 hash 未改变。
- Super 脚本不导入或调用 V4。
- Skill 由 Python 显式读取并注入 prompt。
- `skills/fai-crop-recovery/` 不包含可改变 crop 的执行脚本。
- 唯一坐标执行器位于 `recovery/crop_tool.py`。
- `crop_tool.py` 不调用模型、不写文件、不管理 agent loop。
- 除 `engine.py -> crop_tool.expand_crop()` 外不存在第二条 bbox 修改路径。
- Qwen 最终 content 使用严格 JSON Schema。
- thinking 不影响动作解析。
- 一次 turn 最多执行一次 crop 修改。
- 格式重试不会修改 crop 或消耗 turn。
- 三个 subagent 永远只读同一 observation。
- `max_turns=6` 最多产生五次扩图。
- wider context 每轮根据 full image 和新 bbox 动态生成。
- semantic incomplete、零选择和 mapping protocol failure 都有 recovery 路径。
- 所有坐标变化都能从 history.json 重放。
- 协议失败不会触发盲目四向扩张。
- 达到边界、面积或 turn 上限时状态明确，不把未验证 crop 命名为 validated。

## 20. 推荐的首次实现顺序

首次编码严格按以下顺序进行，避免同时调试模型协议和多智能体：

1. Super 独立 `process_image_v5()`。
2. `recovery/models.py` 数据契约。
3. `recovery/crop_tool.py` 及其完整单元测试。
4. Skill loader。
5. Action Schema 和 parser。
6. 动态 context。
7. 单 agent 六轮 loop 和 history。
8. empty semantic mapping recovery。
9. fake-client 全部测试通过。
10. 加三个只读 subagent 和 coordinator。
11. 最后运行 `p2.png` 实图回归并与 V4/Ultra V5 对照。

这个顺序把最关键的协议正确性和单状态所有权先固定下来，再增加并行判断能力；若
subagent 暂时不可用，单主 agent loop 仍应是一个完整、可测试、可运行的系统。
