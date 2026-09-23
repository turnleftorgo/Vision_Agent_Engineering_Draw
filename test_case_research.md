FAI_196_007

这个案例的成功点非常明确：不是上游 semantic association 做得好，而是 recovery agent 在上游证据严重缺失的情况
  下，通过两轮观察和定向扩图，把一个明显不完整的 minimum crop 修复成了完整结果。

  流程是：

  上游只找到 FAI196 + 一小段文字
          ↓
  semantic 明确报告 leader/arrowhead/target 缺失
          ↓
  第一次向右、向下扩图
          ↓
  重新观察，发现下方仍不完整
          ↓
  第二次只向下扩图
          ↓
  第三轮确认完整并 finish

  ## 1. 上游其实失败得很严重

  Candidate 7 一共产生了 110 个 primitive：

  FAI marker       1
  OCR text        34
  leader segment  70
  arrowhead        2
  triangle         3

  但 LocateAnything 返回：

  annotation proposal = None
  target proposal     = None

  70 条 OpenCV 线段经过压缩以后，也没有形成任何可信的 LP：

  leader_path_count = 0

  Semantic association 最后只选中了：

  F0  = FAI196
  T12 = "Ul<+»U2"

  而且 T12 的 OCR confidence 只有 0.127。

  最终 mapping 是：

  {
    "fai_number": "196",
    "spc_letter": "GJ",
    "leader_path_ids": [],
    "arrowhead_ids": [],
    "target_ids": [],
    "complete": false,
    "missing": ["leader", "arrowhead", "target"]
  }

  见 output_super_p3/crop_related/candidate_007/selection.json。

  这一阶段的成功不在于“识别全了”，而在于它没有像 FAI19 那样错误声明 complete:true。它诚实地把三类关键缺失传给
  了 recovery。

  ## 2. Minimum crop 明显不完整

  初始 crop 是：

  [4539, 2736, 5658, 3656]

  查看 output_super_p3/crop_related/candidate_007/minimum.png 可以看到：

  - FAI196/SPC GJ 完整；
  - U1 ↔ U2 和 1.6 参数完整；
  - leader 向右伸出 crop；
  - R1 的水平 leader 也在右边界被截断；
  - 右侧网格零件 target 几乎完全不可见；
  - 下方相关箭头和零件局部也没有包含。

  如果没有 recovery，这个结果肯定不合格。

  ## 3. 第一轮扩图方向判断正确

  主 agent 连续三次发现：

  missing = leader, arrowhead, target

  但都返回了无效的全零扩图，因此启动 subagent。

  第一轮只有 adversarial verifier 给出了合法建议：

  {
    "crop_complete": false,
    "missing": ["leader", "arrowhead", "target"],
    "expand": {
      "right": 300,
      "bottom": 150
    }
  }

  Python 的确定性仲裁采纳这个方向，实际执行：

  右侧：+335 px
  下方：+138 px

  坐标变化：

  Before: [4539, 2736, 5658, 3656]
  After:  [4539, 2736, 5993, 3794]

  这是正确的视觉判断，因为原图中：

  - 1.6 框的 leader 向右下连接到网格零件；
  - R1 leader 向右连接到零件边缘；
  - target 明显位于 initial crop 的右侧；
  - 下方还有相关的局部箭头和结构。

  过程可在 output_super_p3/expand/candidate_007/01_turn_01_expand_right_335px_bottom_138px_transition.png 中验
  证。

  ## 4. 第二轮没有机械重复第一次方向

  第一次扩图后，程序不是继续照搬“右下扩图”，而是从完整原图重新生成动态观察。

  第二轮三个 subagent 都认为仍然缺少：

  leader
  arrowhead
  target

  但扩图建议变成：

  Adversarial verifier: bottom 150
  Completeness auditor: bottom 100
  Geometry tracer:      bottom 200

  三者对方向达成一致：右侧已经足够，现在只需要继续向下。

  Python 取中间值：

  bottom_norm = 150

  实际继续向下扩：

  +159 px

  坐标变成：

  Before: [4539, 2736, 5993, 3794]
  After:  [4539, 2736, 5993, 3953]

  见 output_super_p3/expand/candidate_007/02_turn_02_expand_bottom_159px_transition.png。

  这一点很重要：它证明这里确实发生了：

  观察 → 行动 → 新观察 → 调整行动

  而不是预先计算一个大框或重复固定方向。

  ## 5. 第三轮知道何时停止

  第三轮使用新的 crop：

  [4539, 2736, 5993, 3953]

  主 agent 第一遍就返回合法结果：

  {
    "action": "finish",
    "valid": true,
    "candidate_valid": true,
    "missing": [],
    "confidence": 0.95
  }

  没有继续启动 subagent，也没有进行第三次扩图。

  这说明它不仅能扩，还能在证据完整后停止，没有耗尽全部 6 轮。

  最终状态被准确记录为：

  validated_from_incomplete_mapping

  而不是普通 validated。这正确表达了：

  > 初始 semantic mapping 不完整，但 recovery 成功补救。

  完整历史见 output_super_p3/crop_related/candidate_007/recovery/history.json:7。

  ## 6. 最终 crop 成功包含了什么

  最终 output_super_p3/crop/FAI196_007_validated.png 包含：

  - FAI 196
  - SPC GJ
  - U1 ↔ U2
  - 1.6 参数框
  - 从参数框连接到零件的完整 leader
  - R1 leader 和 terminal arrowhead
  - 箭头接触的网格零件边缘
  - 足够大的局部 target feature
  - 下方相关的局部引线和结构

  它没有为了捕获 target 而向左或向上盲目扩张，只扩了右侧和下侧。

  总扩图量：

  right  = 335px
  bottom = 138px + 159px = 297px
  left   = 0
  top    = 0

  这同时满足“完整”和“控制 crop 面积”。

  ## 7. Debug 体系也在这个案例中发挥了价值

  crop_related/candidate_007/ 能证明上游为什么不够：

  - output_super_p3/crop_related/candidate_007/semantic.png：没有 LP path，只有少量箭头候选
  - output_super_p3/crop_related/candidate_007/qwen_selected.png：Qwen 实际只选了 F0 和 T12
  - output_super_p3/crop_related/candidate_007/minimum.png：显示初始 crop 的确截断右侧证据
  - recovery/：保存每轮图片、主 agent、subagent、coordinator 和协议错误

  expand/candidate_007/ 则完整记录了空间变化：

  00_initial.png
  01_turn_01_expand_right_335px_bottom_138px.png
  01_turn_01_expand_right_335px_bottom_138px_transition.png
  02_turn_02_expand_bottom_159px.png
  02_turn_02_expand_bottom_159px_transition.png
  timeline.json

  见 output_super_p3/expand/candidate_007/timeline.json。

  ## 准确评价

  这个案例证明了三件重要事情：

  1. 上游 semantic mapping 不完整时，不能跳过 recovery。
  2. 动态 wider context 能逐轮暴露新的 leader 和 target。
  3. Subagent 建议加 Python 确定性仲裁，确实可以挽救主 agent 的协议失败。

  但还有两个明显问题：

  - 前两轮主 agent 和 coordinator 都持续产生 expand_crop_cross_field_violation；
  - 第一轮真正有效的方向来自 adversarial verifier，而不是 geometry tracer。

  所以这是一个非常成功的“agentic recovery + Python 仲裁”案例，但不是一个成功的上游 evidence proposal 案例，也
  不是 coordinator 成功案例。真正的核心成功点是：系统在几乎只有 FAI marker 和一小段文字的情况下，通过两次有方
  向的观察—扩图循环恢复出了完整 crop。









─ Worked for 1m 35s ──────────────────────────────────────────────────────────────────────────────────────────


› 好，现在分析vision_ultra. 看这一段日志：[1/7] LocateAnything FAI scan tile 75: origin=(8640,5400),
  size=1200x1200
  [1/7] LocateAnything FAI scan tile 76: origin=(9000,5400), size=1200x1200
  [1/7] OpenCV circle-pair proposals + Qwen magnified validation
  [1/7] Found 22 deduplicated FAI candidate(s)
  [2/7] Candidate 0: LocateAnything annotation proposal
  [2/7] Candidate 0: LocateAnything arrowhead proposal
  [2/7] Candidate 0: LocateAnything target-part proposal
  [3/7] Candidate 0: local OCR
  [3/7] Candidate 0: OpenCV line and arrow geometry
  [4/7] Candidate 0: compact semantic association (80 primitives -> compact)
  [4/7] Candidate 0: selected 1 leader segment(s), 1 arrow(s)
  [5/7] Candidate 0: Qwen crop completeness validation
  [6/7] Candidate 0: saved /Users/macsuper/Desktop/Vision_Agent/output_ultra_p3/crops/FAI172_000.png + expand
  visualization
  [2/7] Candidate 1: LocateAnything annotation proposal
  [2/7] Candidate 1: LocateAnything arrowhead proposal
  [2/7] Candidate 1: LocateAnything target-part proposal
  [3/7] Candidate 1: local OCR
  [3/7] Candidate 1: OpenCV line and arrow geometry
  [4/7] Candidate 1: compact semantic association (93 primitives -> compact)
  [4/7] Candidate 1: selected 4 leader segment(s), 0 arrow(s)
  [5/7] Candidate 1: Qwen crop completeness validation
  [5/7] Candidate 1: expanded crop [3663, 1182, 5644, 2402] -> [3663, 1182, 5763, 2402]
  [6/7] Candidate 1: saved /Users/macsuper/Desktop/Vision_Agent/output_ultra_p3/crops/FAI193_001.png + expand
  visualization
  [2/7] Candidate 2: LocateAnything annotation proposal
  [2/7] Candidate 2: LocateAnything arrowhead proposal
  [2/7] Candidate 2: LocateAnything target-part proposal
  [3/7] Candidate 2: local OCR
  [3/7] Candidate 2: OpenCV line and arrow geometry
  [4/7] Candidate 2: compact semantic association (109 primitives -> compact)
  [4/7] Candidate 2: selected 0 leader segment(s), 1 arrow(s)
  [5/7] Candidate 2: Qwen crop completeness validation
  [5/7] Candidate 2: expanded crop [2035, 2275, 3540, 2973] -> [1855, 2275, 3871, 2973]
  为什么它容易卡死？


• 从这段日志的位置判断，它不是卡在扩图计算，而是很可能卡在 Candidate 2 的第二次 validation_retry 模型调用：程
  序先打印 “expanded crop”，然后同步调用 Qwen，只有返回后才会打印 [6/7] saved。我会核对文件是否落盘、调用参
  数、图片尺寸、token 上限和超时/重试设置。

• Ran printf 'CANDIDATE 2 FILES\n'; find output_ultra_p3 -type f | rg 'candidate_002|FAI.*_002' | sort
  │ printf '\nVALIDATION FILES\n'; for f in output_ultra_p3/raw_responses/candidate_002_validation*.txt; do
  │ test -e "$f" || continue; printf '\n%s size=' "$f"; stat -f '%z bytes modified=%Sm' -t '%Y-%m-%d %H:%M:%S'
  │ … +5 lines
  └ CANDIDATE 2 FILES
    output_ultra_p3/debug/candidate_rois/candidate_002.png
    … +58 lines (ctrl + t to view transcript)
    1788505207 2026-09-04 15:00:07 302 output_ultra_p3/raw_responses/candidate_001_annotation.txt
    1788505185 2026-09-04 14:59:45 923 output_ultra_p3/expand/candidate_000_expand.json

• Explored
  └ Search def call_vision_model|def run_compact_semantic_mapping|def validate_crop|OpenAI\(|timeout|
  max_tokens|
           stream|reasoning|response_format|semantic-max|validation in FAI_DET_CROP_3_Ultra.py
    Read FAI_DET_CROP_3_Ultra.py
    Read FAI_DET_CROP_3_Ultra.py
    Read FAI_DET_CROP_3_Ultra.py
    Read FAI_DET_CROP_3_Ultra.py
    Read FAI_DET_CROP_3_Ultra.py

• Context compacted

• Ran python3 -c 'from pathlib import Path; from PIL import Image; import openai; from openai._constants
  │ import DEFAULT_MAX_RETRIES; files=[Path("output_ultra_p3/debug/candidate_rois/
  │ candidate_002.png"),Path("output_ultra_p3/debug/evidence/candidate_002_compact_semantic.png")];
  │ … +2 lines
  └ openai 2.37.0 default_max_retries 2
    output_ultra_p3/debug/candidate_rois/candidate_002.png (2401, 2401) 338739
    output_ultra_p3/debug/evidence/candidate_002_compact_semantic.png (2401, 2401) 338826

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 目前证据已经很明确：Candidate 2 的第一次 validation 已经正常返回，而且它要求左右同时扩；扩图后的第二次请求要
  重新发送一张 2401×2401 的上下文图和一张 2016×698 的 crop。当前目录没有 retry 响应文件，说明沉默发生在这次同
  步 Qwen 调用里。更关键的是，第一次判断把邻近的 FAI19/SPC S 当成了 FAI18 必须包含的内容，制造了一次本不该发生
  的扩图和重试。

• Explored
  └ Search def call_vision_model|def validate_crop|max_tokens=8192|candidate_.*validation_retry|expanded crop|
           OpenAI\(|--timeout|retries: int|chat.completions.create|def build_validation_context in
           FAI_DET_CROP_3_Ultra.py

• Ran sed -n '258,292p' Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py; sed -n '1840,1908p' Vision_Agent_Ultra/
  │ FAI_DET_CROP_3_Ultra.py; sed -n '2168,2202p' Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py; stat -f '%Sm
  │ %N' -t '%Y-%m-%d %H:%M:%S' output_ultra_p3/debug/evidence/candidate_002_compact_semantic.png
  │ … +2 lines
  └ def call_vision_model(
        client: OpenAI,
    … +138 lines (ctrl + t to view transcript)
    2026-09-04 15:27:50 output_ultra_p3/raw_responses/candidate_002_mapping.txt
    2026-09-04 15:47:46 output_ultra_p3/raw_responses/candidate_002_validation.txt

• Ran sed -n '1800,1865p' Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py; printf '\ncandidate_002_mapping:\n'; sed
  │ -n '1,220p' output_ultra_p3/raw_responses/candidate_002_mapping.txt; printf '\ncandidate_002_validation:
  │ \n'; sed -n '1,180p' output_ultra_p3/raw_responses/candidate_002_validation.txt; printf '\nretry
  │ … +2 lines
  └         if candidate is not None:
                selected.append(candidate)
    … +101 lines (ctrl + t to view transcript)
    retry existence:
    missing

• Ran for f in output_ultra_p3/debug/evidence/candidate_001_compact_semantic.png output_ultra_p3/
  │ raw_responses/candidate_001_mapping.txt output_ultra_p3/raw_responses/candidate_001_validation.txt
  │ output_ultra_p3/raw_responses/candidate_001_validation_retry.txt output_ultra_p3/debug/evidence/
  │ … +2 lines
  └ 15:00:51 output_ultra_p3/debug/evidence/candidate_001_compact_semantic.png
    15:04:40 output_ultra_p3/raw_responses/candidate_001_mapping.txt
    … +3 lines (ctrl + t to view transcript)
    15:27:50 output_ultra_p3/raw_responses/candidate_002_mapping.txt
    15:47:46 output_ultra_p3/raw_responses/candidate_002_validation.txt

• Explored
  └ Search def validation_prompt|raw_output_path.write_text|def calculate_initial_crop|return data if
  isinstance|
           content.extend|OpenAI\( in FAI_DET_CROP_3_Ultra.py

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 结论：这不是扩图算法“死循环”，而是程序阻塞在 Candidate 2 扩图后的第二次 Qwen validation 请求上。Ultra 每个
  Candidate 最多只验证两次，不存在无限扩图。

  ### 当前具体卡在哪里

  执行顺序是：

  第一次 validation 返回
      ↓
  计算扩图 bbox
      ↓
  打印 expanded crop
      ↓
  同步调用第二次 validation
      ↓
  写 candidate_002_validation_retry.txt
      ↓
  保存 crop 和 expand 图

  日志停在：

  expanded crop [...] -> [...]

  而输出目录中：

  candidate_002_validation.txt        存在
  candidate_002_validation_retry.txt  不存在
  Candidate 2 crop                    不存在
  Candidate 2 expand 图               不存在

  代码也是先调用第二次 validation，等它完整返回后才继续保存文件：Vision_Agent_Ultra/
  FAI_DET_CROP_3_Ultra.py:2188。

  所以准确来说，它正在等待：

  validation = validate_crop(
      ...,
      retry_context,
      retry_crop,
      candidate_002_validation_retry.txt,
  )

  ### 为什么 Candidate 2 特别容易卡住

  #### 1. Validation 发生了语义串组

  Candidate 2 对应的是：

  {
    "fai_number": "18",
    "spc_letter": "R"
  }

  但第一次 validation 却认为缺少：

  [
    "FAI 19 circle (clipped at right border)",
    "SPC S circle (outside right border)",
    "right-side description text 'MEASURE KEY WELL Z OFFSET ON ALL 4 SIDES'"
  ]

  也就是说，Qwen把旁边的 FAI19/SPC S 误认为 FAI18 crop 必须包含的内容。

  这导致程序执行了本不必要的向右扩图。同时它还要求向左找 B1 terminal，于是左右都扩：

  原始：1505 × 698
  扩后：2016 × 698

  根本问题是 validation prompt 只告诉模型“红框是 selected FAI marker”，但没有明确传入：

  本次目标必须是 FAI 18 / SPC R
  其他 FAI 编号均属于邻近 group，禁止纳入

  因此 wider context 里同时出现 FAI18 和 FAI19 时，Qwen很容易串组。

  #### 2. 上游 mapping 本身不完整

  日志显示：

  selected 0 leader segment(s), 1 arrow(s)

  实际 mapping 是：

  {
    "fai_number": "18",
    "text_ids": ["T6"],
    "leader_path_ids": [],
    "arrowhead_ids": ["H0"],
    "target_ids": [],
    "complete": false,
    "missing": [
      "annotation_ids",
      "leader_path_ids",
      "target_ids"
    ]
  }

  所以初始 crop 只建立在一个文字框和一个箭头上，没有完整 leader path 约束。

  Validation 因而不得不重新从大图推断“哪个文字、哪个 leader、哪个 FAI 属于同组”，既慢又容易选到 FAI19。

  #### 3. 第二次 validation 仍发送两张大图

  当前 retry 一次发送：

  - IMAGE 1：2401×2401 wider context
  - IMAGE 2：2016×698 expanded crop

  而且两张图都转换成完整 PNG data URL 发送：Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py:258。

  虽然 semantic association 已经压缩成一张 compact image，但 validation 仍然保留 V3 的两图机制。代码里甚至明确
  标注：

  # Validation (unchanged from V3)

  因此 Ultra 当前真正的速度瓶颈已经从 semantic association 转移到了 validation。

  #### 4. 一个很小的 JSON 却允许输出 8192 tokens

  Validation 只需要大约几十至一百个 token 的 JSON，却设置了：

  max_tokens=8192

  见 Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py:1858。

  同时没有：

  response_format=json_schema

  Qwen可以花很长时间 thinking，甚至生成很长的自由文本。程序又是非流式同步调用，因此在完整结果返回前终端不会出
  现任何信息。

  #### 5. 重试机制会放大等待时间

  目前有两层重试：

  - call_vision_model() 自己设置 retries=2，即最多调用3次。
  - 本地 OpenAI SDK 2.37.0 默认又有 max_retries=2。
  - 单次 timeout 是300秒。

  发生超时或可重试错误时，理论上一个逻辑调用最多可能产生：

  3次外层尝试 × 每次最多3次SDK请求 = 9次HTTP请求

  所以表面上的一次 validation，极端情况下可能沉默几十分钟。

  #### 6. 文件时间也证明是模型调用慢，不是 OpenCV 慢

  Candidate 2 的时间：

  15:13:21  compact semantic image 完成
  15:27:50  mapping 返回          约14分29秒
  15:47:46  第一次 validation 返回 约19分56秒
  之后      第二次 validation 尚未落盘

  Candidate 1 每次调用大约3～4分钟，而 Candidate 2 已经增长到14～20分钟。这说明 LM Studio/Qwen 的推理、队列、
  长输出或超时重试才是主要耗时，不是 bbox 扩展计算。

  ### 最优先应该怎么修

  建议按这个优先级：

  1. 强绑定目标 FAI：

  TARGET_MARKER = FAI 18
  TARGET_SPC = R

  Only validate the group belonging to TARGET_MARKER.
  Any other FAI marker, including FAI 19, is a neighboring group.
  Never request expansion merely to include another FAI/SPC marker.

  2. 把 mapping 结果一起传给 validation：

  {
    "target_fai": "18",
    "target_spc": "R",
    "selected_text": ["T6"],
    "selected_paths": [],
    "selected_arrows": ["H0"]
  }

  3. Validation 攓为一张动态 composite image，不再发两张2401宽的大图。
  4. 使用严格 JSON Schema，并把最终输出上限降到约512 tokens。
  5. 禁用 SDK 隐式重试：

  OpenAI(
      ...,
      timeout=args.timeout,
      max_retries=0,
  )

  只保留Python调度器的一层明确重试。

  6. 给每次调用增加开始、结束和耗时日志，例如：

  Candidate 2 validation_retry:
  images=[2401x2401, 2016x698]
  request started...
  request returned in 42.8s

  所以这次“卡死”的直接原因是第二次 Qwen validation 未返回；诱因则是 FAI18/FAI19 串组，导致不必要的双向扩图，再
  叠加两张大图、8192输出上限和多层超时重试。







─ Worked for 2m 09s ──────────────────────────────────────────────────────────────────────────────────────────


› 不用修。对比super和ultra的表现，告诉我，为什么super几乎从不卡死


• 我会把 Super 和 Ultra 的模型调用链、图片数量/尺寸、token 上限、超时重试以及 subagent 触发条件逐项对照，再用
  output_super_p3 与 output_ultra_p3 的实际时间戳判断差异。重点是区分“Super调用更多但可控”与“Ultra单次调用长期
  无返回”。

• Explored
  └ List Vison_Agent_Super
    Search def call_vision_model|def run_agentic|def validate|recovery|subagent|max_tokens|max_format_retries|
           max_turn|OpenAI\(|--timeout|response_format|json_schema|images|pil_to_data_url|starting read-only
           in
           fai_DET_crop_5.py
    List Vison_Agent_Super
    Read fai_DET_crop_5.py, engine.py, agents.py, observation.py, protocol.py, semantic.py, models.py

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +16 lines
  └
     output_super_p3
    … +39 lines (ctrl + t to view transcript)
    output_super_p3/crop_related/candidate_007/recovery/turn_01_main_attempt_01_meta.json
    output_super_p3/crop_related/candidate_007/recovery/turn_01_main_attempt_02_meta.json

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +12 lines
  └ count 89 total_serial_equiv_s 2674.8 max_s 61.3
       61.32s finish=stop prompt=7492 completion=81 contentB=158 reason=0 error=None output_super_p3/
    … +31 lines (ctrl + t to view transcript)
       46.97s finish=stop prompt=5683 completion=72 contentB=158 reason=0 error=None output_super_p3/
    crop_related/candidate_003/recovery/turn_01_subagents/subagent_geometry_tracer_meta.json

• Ran wc -c output_ultra_p3/raw_responses/candidate_000_mapping.txt output_ultra_p3/raw_responses/
  │ candidate_000_validation.txt output_ultra_p3/raw_responses/candidate_001_mapping.txt output_ultra_p3/
  │ raw_responses/candidate_001_validation.txt output_ultra_p3/raw_responses/
  │ … +6 lines
  └      323 output_ultra_p3/raw_responses/candidate_000_mapping.txt
         122 output_ultra_p3/raw_responses/candidate_000_validation.txt
    … +64 lines (ctrl + t to view transcript)
      "confidence": 0.85
    }

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 对比结果出现了一个很强的反差：Super 的 89 份模型调用元数据里，最慢一次只有约 61 秒，且最终内容通常只有 40–
  130 tokens；Ultra 虽然磁盘上的 JSON 也很短，但 Candidate 2 的 mapping 用了约14分半、validation 又用了约20分
  钟。这表明关键不在“结果字数”，而在 Super 用 JSON Schema 约束了生成路径，并把请求错误变成可解析、可终止的状
  态；Ultra 是无约束同步等待，可能在内部 thinking、超时和隐式重试阶段长时间沉默。

• Ran python3 - <<'PY'
  │ import json, statistics
  │ from pathlib import Path
  │ … +23 lines
  └ coordinator n 19 min 8.15 median 12.1 mean 15.01 max 24.62
    main n 31 min 5.73 median 22.56 mean 18.34 max 32.77
    … +16 lines (ctrl + t to view transcript)
    candidate_009 calls 20 serial_sum 727.3 max 61.3
    candidate_010 calls 10 serial_sum 250.4 max 46.2

• Explored
  └ Read FAI_DET_CROP_3_Ultra.py
    Search response_format=response_format|def call_structured_agent|ThreadPoolExecutor|
    build_dynamic_observation|
           fit_image|return AgentCall|except Exception|max_content_bytes|finish_reason == \"length\"|for
           turn_number|max_format_retries in *.py
    Read semantic.py, agents.py, engine.py, observation.py

• Ran python3 - <<'PY'
  │ from pathlib import Path
  │ from PIL import Image
  │ … +13 lines
  └ (no output)

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +15 lines
  └ (no output)

• Explored
  └ List candidate_002
    Search *_meta.json in recovery

• Ran for f in output_super_p3/crop_related/candidate_002/recovery/turn_01_main_attempt_00.txt
  │ output_super_p3/crop_related/candidate_002/recovery/turn_01_main_attempt_01.txt output_super_p3/
  │ crop_related/candidate_002/recovery/turn_01_action.json output_super_p3/crop_related/candidate_002/
  │ … +2 lines
  └ output_super_p3/crop_related/candidate_002/recovery/turn_01_main_attempt_00.txt
    {"action": "expand_crop", "valid": true, "candidate_valid": true, "missing": ["bottom_feature"],
    … +132 lines (ctrl + t to view transcript)
      "error": null
    }

• Explored
  └ Read SKILL.md, agents.py, protocol.py
    Search results.json|write_text\(json.dumps|for index, marker_box|records.append|return records in
           FAI_DET_CROP_3_Ultra.py
    Read fai_DET_crop_5.py, FAI_DET_CROP_3_Ultra.py

• Ran python3 - <<'PY'
  │ import json
  │ p='output_super_p3/results.json'; d=json.load(open(p))
  │ … +4 lines
  └ 0 172 validated turns 2 exp 1
    1 193 validated turns 1 exp 0
    … +6 lines (ctrl + t to view transcript)
    8 84 validated_from_incomplete_mapping turns 1 exp 0
    9 153 validated turns 3 exp 2

• Ran python3 - <<'PY'
  │ from PIL import Image,ImageChops,ImageStat
  │ p1=Image.open('output_super_p3/crop_related/candidate_002/semantic.png').convert('RGB')
  │ … +7 lines
  └ (2401, 2401) (2401, 2401)
    bbox (0, 0, 1877, 2343) mean [2.2852259774448416, 2.216775392593777, 2.21054152606482] rms
    … +7 lines (ctrl + t to view transcript)
    perl: warning: Falling back to the standard locale ("C").
    panic: locale.c: 4486: Could not change LC_CTYPE locale to C.UTF-8, errno=9

• Ran python3 - <<'PY'
  │ import json
  │ x=json.load(open('output_super_p3/results.json'))
  │ … +6 lines
  └ {'locate': 'LocateAnything-3B-8bit', 'semantic_and_recovery': 'Qwen3.8-27B-MLX-8bit', 'endpoint':
    'http://127.0.0.1:8001/v1'}
    … +12 lines (ctrl + t to view transcript)
    Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py:2270:    parser.add_argument("--qwen-model",
    default=DEFAULT_QWEN_MODEL)

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 结论：Super 几乎不卡住，首要原因不是 subagent，而是它把 Qwen 从“自由生成文本”改成了“受 JSON Schema 约束的短
  动作生成”，再用有边界、可失败退出的 Python 状态机调度。Ultra 仍然是 V3 式的无约束同步调用。

  ### Candidate 2 是很好的对照实验

  两边使用相同模型和端点：

  Qwen3.8-27B-MLX-8bit
  http://127.0.0.1:8001/v1

  Semantic 图片也几乎一样：

   项目                               Super                             Ultra
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   semantic image                 2401×2401                         2401×2401
  ──────────────────  ──────────────────────  ────────────────────────────────
   PNG 大小                       338,823 B                         338,826 B
  ──────────────────  ──────────────────────  ────────────────────────────────
   semantic mapping                28.70 秒                        约14分29秒
  ──────────────────  ──────────────────────  ────────────────────────────────
   第一次 crop 判断                23.31 秒                        约19分56秒
  ──────────────────  ──────────────────────  ────────────────────────────────
   后续                11.61 秒完成格式修复    第二次 validation 长时间未返回

  Super 的 Candidate 2 总 Qwen 时间大约63.6秒；Ultra 已经超过34分钟，第二次 validation 还没有完成。

  因此，图片大小不是主要解释。两者最核心的差别是输出协议。

  ## 1. Super 真正使用了严格 JSON Schema

  Super 调用中有：

  response_format=response_format(schema_name, schema)

  见 Vison_Agent_Super/recovery/agents.py:72。

  这不是普通 prompt 里的“Return JSON only”，而是让 LM Studio 使用 guided decoding：模型生成每一个 token 时，都
  只能选择符合 Schema 的 token。

  所以 Super 的实际调用数据是：

  completion_tokens：通常 41～150
  content：通常 140～300 bytes
  finish_reason：全部 stop
  独立 reasoning_content：0 字符
  最长单次调用：61.32 秒

  也就是说，max_tokens=8192 虽然仍然很大，但模型实际上在约100 tokens 就被结构化协议收束了。

  Ultra 则只是文字要求：

  Return JSON only

  调用本身没有 response_format：Vision_Agent_Ultra/FAI_DET_CROP_3_Ultra.py:258。

  因此 Qwen 仍然可以：

  - 进行很长的 thinking；
  - 在内部反复分析图片；
  - 生成自由格式后再收敛到短 JSON；
  - 直到8192 tokens上限附近才结束；
  - 或在300秒 timeout 后触发重试。

  Ultra 最后保存的 JSON 很短，并不代表生成过程很短。长时间消耗很可能发生在没有被 Ultra 保存的 reasoning 或重试
  阶段。

  ## 2. Super 遇到错误时，先做“格式修复”，不会立即扩图

  Super Candidate 2 第一次返回的是：

  {
    "action": "expand_crop",
    "valid": true,
    "missing": ["bottom_feature"],
    "arguments": {
      "left_norm": 0,
      "top_norm": 0,
      "right_norm": 0,
      "bottom_norm": 0
    }
  }

  这是自相矛盾的：

  - expand_crop 却 valid=true
  - 声称缺东西，却四个方向都扩0

  Super 的协议检查立刻识别为：

  expand_crop_cross_field_violation

  然后保持图片和 crop 不变，把具体错误反馈给模型。第二次在11.61秒后返回：

  {
    "action": "finish",
    "valid": true,
    "missing": [],
    "arguments": {
      "left_norm": 0,
      "top_norm": 0,
      "right_norm": 0,
      "bottom_norm": 0
    }
  }

  因此 FAI18 没有发生错误扩图。

  这套字段一致性检查位于 Vison_Agent_Super/recovery/protocol.py:163，格式修复循环位于 Vison_Agent_Super/
  recovery/engine.py:245。

  Ultra 只要能解析出一个字典，就基本相信里面的内容。于是它把这类视觉或语义错误直接转化成：

  扩大图片
  → 重新发送更大的图片
  → 再做一次昂贵 validation

  Candidate 2 正是这样被引向第二次慢调用的。

  ## 3. Super 明确告诉 Qwen 当前目标是谁

  Super recovery prompt 会注入：

  {
    "semantic_hypothesis": {
      "fai_number": "18",
      "spc_letter": "R",
      "complete": true/false,
      "missing": [...]
    }
  }

  见 Vison_Agent_Super/recovery/agents.py:122。

  Skill 又明确规定：

  Do not borrow content from a neighboring group.
  Never request expansion for unrelated FAI/SPC groups.

  见 Vison_Agent_Super/skills/fai-crop-recovery/SKILL.md:1。

  所以 Super Candidate 2 能把 FAI18 和旁边的 FAI19 分开。

  Ultra 的 validation prompt 没有传入：

  target_fai = 18
  target_spc = R

  它只告诉模型红框是 selected marker。因此 Qwen把 FAI19/SPC S 串入 FAI18，要求向右扩图。这个错误不但影响准确
  度，也额外触发了一次大图请求。

  ## 4. Super 的观察图更聚焦

  Ultra validation 的 IMAGE 1 固定使用整个 candidate ROI：

  2401×2401

  无论当前 crop 多大，都会重新发送这个方形大区域。

  Super 根据当前 crop 动态产生 wider context，并把最长边限制在2400：Vison_Agent_Super/recovery/
  observation.py:12。

  例如 FAI18：

  Super context：2400×1463
  Super crop：   1682×1025

  Ultra context：2401×2401
  Ultra crop：   1505×698

  Super 两张图总像素大约少23%，而且动态 context 里邻近 FAI 的干扰更少。

  这个差异能改善速度，但不足以单独解释“23秒和20分钟”的巨大差距；严格 Schema 仍是主因。

  ## 5. Super 会失败退出，不会无限等待“正确答案”

  Super 的状态机有明确终止条件：

  - 最多6个观察 turn；
  - 最后一个 turn 禁止继续扩图；
  - 最多2次格式修复；
  - 到达图像边界后停止；
  - 无有效边界变化后停止；
  - 协议持续失败则标记 protocol_failed；
  - 保存历史最优 crop。

  循环位置在 Vison_Agent_Super/recovery/engine.py:424。

  当前 output_super_p3 并不是所有 Candidate 都成功：

  validated                         5
  validated_from_incomplete_mapping 2
  protocol_failed                   3

  所以更准确地说，Super不是“永远成功”，而是“遇到解决不了的问题会快速、可追踪地失败，不会一直悬在那里”。

  Ultra 遇到一次慢 validation 时，整个顺序处理链就停在当前 Candidate。

  ## 6. Subagent 不是不卡死的主要原因

  Super 的三个 subagent 确实并行执行：

  ThreadPoolExecutor(max_workers=len(roles))

  因此三个各需约50秒的判断，墙上时间接近最慢的那个约50～61秒，而不是150秒。

  但简单 Candidate 根本不需要 subagent。FAI18 Candidate 2 就是：

  semantic mapping
  → main agent
  → 一次格式修复
  → finish

  没有启动 subagent。

  所以“Super不卡”的主要贡献顺序是：

  1. 严格 JSON Schema/guided decoding；
  2. 动作字段一致性检查；
  3. 格式重试与扩图轮次分离；
  4. 明确传入目标 FAI/SPC 和 Skill 规则；
  5. 动态、较小的观察区域；
  6. 有限状态机与快速失败；
  7. subagent 并行只负责疑难情况。

  还有一个现实限制：Super 仍然使用300秒 timeout，并保留 OpenAI SDK 默认重试。因此它理论上仍可能被某个底层请求
  卡住几分钟，只是 output_super_p3 的89次有记录调用中没有触发这种情况。

  一句话概括：Ultra把“JSON”当作提示词；Super把“JSON动作”做成了模型无法绕开的协议。这个差别才是 Super 几乎不卡
  死的核心。





  

─ Worked for 3m 42s ──────────────────────────────────────────────────────────────────────────────────────────


› [Image #1] 为什么这个fai52截图截的这么清晰，还会是protocol failed ? Oridinator和subagent集体抽风了？查一查
  log告诉我原因


• 我会直接追踪 Candidate 11 的 history.json、main/subagent/coordinator 原始响应和每次协议错误，区分它是“视觉判
  断失败”还是“JSON动作协议失败”。这张最终图清晰，只能说明历史最优 crop 保存得好，并不自动代表最后一次 action
  通过协议。

• Explored
  └ List candidate_011
    Read result.json
    Read history.json

• Ran for f in output_super_p3/crop_related/candidate_011/recovery/turn_01_main_attempt_*.txt output_super_p3/
  │ crop_related/candidate_011/recovery/turn_01_subagents/subagent_*.txt; do printf '\n===== %s =====\n'
  │ "$f"; cat "$f"; done; printf '\n===== semantic mapping =====\n'; cat output_super_p3/crop_related/
  │ … +3 lines
  └
    ===== output_super_p3/crop_related/candidate_011/recovery/turn_01_main_attempt_00.txt =====
    … +224 lines (ctrl + t to view transcript)
      "final_turn": false
    }

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 日志已经定位：不是 coordinator 和 subagent 一起随机失常。Main 连续三次都把上游 semantic 的
  missing=["annotation","leader","target"] 原样照抄，同时又输出“需要扩图但四边扩0、valid=true”的自相矛盾动作。
  三个 subagent 也输出了完全相同的矛盾结论，因此全部被协议层丢弃；由于零条有效 recommendation，coordinator 实
  际上根本没有被调用。接下来我再核对 crop 右边缘是否确实截断说明文字。

• Viewed Image
  └ output_super_p3/crop/FAI52_011_best_unvalidated.png

• Viewed Image
  └ output_super_p3/crop_related/candidate_011/qwen_selected.png

• Viewed Image
  └ output_super_p3/crop_related/candidate_011/recovery/turn_01_context.png

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +8 lines
  └ {"id": "F0", "type": "fai_marker", "bbox": [1149, 1150, 1251, 1250], "source": "LocateAnything"}
    {"id": "A0", "type": "annotation", "bbox": [0, 0, 2107, 2399], "source": "LocateAnything"}
    … +20 lines (ctrl + t to view transcript)
    output_super_p3/crop_related/candidate_011/recovery/history.json:7:  "status": "protocol_failed",
    output_super_p3/crop_related/candidate_011/recovery/result.json:2:  "status": "protocol_failed",

• Explored
  └ Read protocol.py, engine.py, SKILL.md

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 原因查清了：这不是图片不清晰，也不是 coordinator 判断错了——coordinator 根本没有运行。是 Main agent 和三个
  subagent 全部产生了“JSON格式正确、动作逻辑错误”的结果，最终没有一条有效 action，程序才标记为
  protocol_failed。

  而且，这张图虽然清晰，但其实并没有完全截全：右侧说明文字应是 MEASURE KBM POCKET，当前 crop 在右边界截掉了最
  后的 ET。正确动作应该是小幅向右扩图。

  ### 实际执行过程

  #### 1. 上游 semantic mapping 先给出了错误的缺失信息

  output_super_p3/crop_related/candidate_011/semantic_mapping.txt 返回：

  {
    "fai_number": "52",
    "spc_letter": "AZ",
    "text_ids": ["T14", "T15"],
    "leader_path_ids": [],
    "arrowhead_ids": ["H3"],
    "target_ids": [],
    "complete": false,
    "missing": ["annotation", "leader", "target"]
  }

  问题在于 T14、T15 的 OCR 内容实际上是：

  T14 = "eae"
  T15 = "Denen an)"

  它没有正确识别 0.43 A KB KC 和 MEASURE KBM POCKET，也没有识别 leader path。

  但是根据 F0、T14、T15、H3 的位置做 union 后，初始 crop 恰好覆盖了绝大多数真实内容，所以最终图片看起来依然很
  好。

  ### 2. Main agent 连续三次输出同一个矛盾动作

  三次输出几乎完全相同：

  {
    "action": "expand_crop",
    "valid": true,
    "candidate_valid": true,
    "missing": ["annotation", "leader", "target"],
    "arguments": {
      "left_norm": 0,
      "right_norm": 0,
      "bottom_norm": 0,
      "top_norm": 0
    },
    "confidence": 0.95
  }

  这里至少有两个冲突：

  action = expand_crop
  valid = true                 ← 要扩图就不应 valid=true
  四个方向全部为 0             ← 声称扩图但没有扩图方向

  所以三次都被协议检查拒绝：

  expand_crop_cross_field_violation

  记录见 output_super_p3/crop_related/candidate_011/recovery/history.json。

  它也没有真正观察图片缺的是右侧 ET，而是机械照抄了 semantic mapping 的：

  ["annotation", "leader", "target"]

  ### 3. 三个 subagent 也产生了完全相同的矛盾

  三个角色全部返回：

  {
    "candidate_valid": true,
    "crop_complete": false,
    "missing": ["annotation", "leader", "target"],
    "expand": {
      "left": 0,
      "top": 0,
      "right": 0,
      "bottom": 0
    },
    "confidence": 0.95
  }

  逻辑仍然冲突：

  crop_complete = false
  missing 不为空
  但 expand 全部为 0

  因此三份意见全部被判定为：

  incomplete_recommendation_without_expansion

  对应文件：

  - output_super_p3/crop_related/candidate_011/recovery/turn_01_subagents/subagent_completeness_auditor.txt
  - output_super_p3/crop_related/candidate_011/recovery/turn_01_subagents/subagent_geometry_tracer.txt
  - output_super_p3/crop_related/candidate_011/recovery/turn_01_subagents/subagent_adversarial_verifier.txt

  ### 4. Coordinator 为什么没有纠正它们

  程序只在至少有一条 subagent recommendation 通过协议后，才调用 coordinator：

  if recommendations:
      coordinator = ...

  三个 subagent 全部协议无效，所以：

  有效 recommendations = 0
  coordinator 没有启动
  deterministic arbitration 也无内容可汇总
  最终 action = None
  status = protocol_failed

  所以不是“coordinator 和 subagent 集体抽风”，而是：

  Main agent：连续三次逻辑错误
  三个 subagent：同一种逻辑错误
  Coordinator：未运行

  ### 为什么三个 subagent 会如此一致

  它们并不是真正异构、独立的智能体。它们使用：

  - 同一个 Qwen 模型；
  - 同一组图片；
  - 同一个 semantic hypothesis；
  - 同一个 Skill；
  - 近似的 prompt；
  - temperature=0.1。

  因此错误高度相关。上游写着：

  "missing": ["annotation", "leader", "target"]

  四个调用都被这个结论锚定，直接复制，而没有遵守 Skill 中“像素优先、semantic 只是 hypothesis”的规定。

  严格 JSON Schema 只保证字段和类型正确，不能自动保证：

  expand_crop ⇒ valid=false 且至少一个方向>0

  这种跨字段逻辑仍由 Python检查，所以才出现“JSON合规但动作协议不合规”。

  ### 这张图正确的判断

  从 wider context 可以看出，完整文字是：

  MEASURE KBM POCKET

  当前 crop 右边界截断了最后部分。因此合理结果应类似：

  {
    "action": "expand_crop",
    "valid": false,
    "candidate_valid": true,
    "missing": ["right-clipped description text: POCKET"],
    "arguments": {
      "left_norm": 0,
      "top_norm": 0,
      "right_norm": 50,
      "bottom_norm": 0
    }
  }

  文件名 best_unvalidated 的含义也正是：Python保存了当前历史评分最高的 crop（95.8205），所以图片质量看起来很
  好；但整个 agent 流程没有产生一条有效的最终动作，因此不能命名为 validated。

─ Worked for 1m 26s ──────────────────────────────────────────────────────────────────────────────────────────


› [Image #1] 这个156 什么箭头都没有，为什么会判定为validated ?


• 我会检查 Candidate 12 的 semantic mapping、每轮 observation/action、是否调用 subagent，以及它为什么认为“ISO-
  114X”这类无 leader 标注可以独立成立。重点也会核对最终图右侧被截断却仍 finish 的原因。

• Explored
  └ List candidate_012
    Read result.json
    Read history.json
    Read semantic_mapping.txt
    Read selection.json

• Ran for f in output_super_p3/crop_related/candidate_012/recovery/turn_01_main_attempt_*.txt output_super_p3/
  │ crop_related/candidate_012/recovery/turn_01_action.json output_super_p3/crop_related/candidate_012/
  │ recovery/turn_01_observation.json; do printf '\n===== %s =====\n' "$f"; cat "$f"; done; printf
  │ … +6 lines
  └
    ===== output_super_p3/crop_related/candidate_012/recovery/turn_01_main_attempt_00.txt =====
    … +54 lines (ctrl + t to view transcript)
    "points": [[1152, 1300], [1351, 1300]]}
    {"id": "G7", "type": "triangle", "bbox": [1155, 1202, 1248, 1248], "source": "OpenCV-contour"}

• Viewed Image
  └ output_super_p3/crop_related/candidate_012/recovery/turn_01_context.png

• Viewed Image
  └ output_super_p3/crop_related/candidate_012/semantic.png

• Viewed Image
  └ output_super_p3/crop_related/candidate_012/qwen_selected.png

──────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 结论：这是一个明显的 false-positive validation。人眼看不到真实箭头，但程序内部认为已经选到了 LP0 + G7，其中
  G7 被误当成箭头，随后 Main agent 在第二次格式修复时直接输出 finish，Python没有再做几何真实性检查，所以被标记
  为 validated。

  ### 1. Semantic 阶段实际上“认出了”一个假箭头

  原始 mapping 是：

  {
    "fai_number": "156",
    "spc_letter": "EV",
    "annotation_ids": ["T9"],
    "text_ids": ["T9"],
    "leader_path_ids": ["LP0"],
    "arrowhead_ids": ["G7"],
    "target_ids": [],
    "complete": true,
    "missing": ["target"],
    "confidence": 0.95
  }

  见 output_super_p3/crop_related/candidate_012/semantic_mapping.txt。

  也就是说，程序账本里并不是“没有箭头”，而是：

  LP0 = leader path
  G7  = arrowhead

  但实际几何坐标是：

  L43: [1161,1273] → [1351,1293]
  L45: [1152,1300] → [1351,1300]
  G7 bbox: [1155,1202,1248,1248]

  这些都位于 FAI156/SPC 气泡和相邻文字区域，并不是通往零件的真实 leader 和 terminal arrowhead。OpenCV把气泡横
  线、文字笔画或圆圈轮廓组合成了假 leader/triangle。

  ### 2. Semantic mapping 自身还是矛盾的

  它同时声称：

  "complete": true,
  "target_ids": [],
  "missing": ["target"]

  这三个字段显然不能同时成立。

  另外：

  "annotation_ids": ["T9"]

  但 T9 实际类型是 ocr_text，不是 annotation。Python后处理把这个错误 ID 从 annotation_ids 删除了，却没有重新计
  算 complete：

  {
    "annotation_ids": [],
    "leader_path_ids": ["LP0"],
    "arrowhead_ids": ["G7"],
    "target_ids": [],
    "complete": true,
    "missing": ["target"]
  }

  见 output_super_p3/crop_related/candidate_012/selection.json。

  因此 recovery 从一开始就收到了一份高度矛盾、但带有 complete=true, confidence=0.95 的上游结论。

  ### 3. Main agent 第一次其实想扩图，但动作无效

  第一次输出：

  {
    "action": "expand_crop",
    "valid": true,
    "missing": ["target"],
    "arguments": {
      "left_norm": 0,
      "right_norm": 1,
      "top_norm": 0,
      "bottom_norm": 0
    },
    "confidence": 0.7
  }

  它似乎意识到 target 不完整，想向右扩一点。

  但这个动作协议错误：

  action=expand_crop
  valid=true

  扩图时必须是 valid=false，所以被判定为：

  expand_crop_cross_field_violation

  ### 4. 第二次格式修复时，它直接改成了 finish

  第二次返回：

  {
    "action": "finish",
    "valid": true,
    "candidate_valid": true,
    "missing": [],
    "arguments": {
      "left_norm": 0,
      "right_norm": 0,
      "top_norm": 0,
      "bottom_norm": 0
    },
    "confidence": 0.95
  }

  这个 JSON 的字段关系完全合法，所以 Python接受了它。

  注意：协议验证只能证明：

  finish
  → valid=true
  → missing=[]
  → expansion=0

  它不能证明图里真的存在箭头和 target。于是一个视觉上错误、但协议上自洽的 finish 被当成了 validation。

  完整记录在 output_super_p3/crop_related/candidate_012/recovery/history.json。

  ### 5. 为什么没有启动 subagent

  Subagent 的触发条件之一是：

  Main 输出 finish
  并且 selected evidence 接触 crop 边界

  但当前 selected evidence 是错误的：

  F0、T9、L43、L45、G7

  这些假 evidence 全部位于 crop 内部，没有碰到边界，因此：

  evidence_border_warning = false
  Main confidence = 0.95
  action = finish
  → 不启动 subagents

  所以 Candidate 12 完全由 Main agent 一个人结束，没有经过三个 auditor。

  ### 6. 高分也来自假 evidence

  它的 Python score 是：

  99.7194

  这个分数计算的是“selected evidence 是否被 crop 包含”。

  由于假 LP0、假 G7 和 F0 都在 crop 内部，它自然获得接近100的分数。这个分数不知道它们是不是真实 leader/arrow，
  也不知道右侧文字是否截断。

  ### 实际图像并不满足完整条件

  从 wider context 可以看到，紫色 crop 右边界截断了：

  - 0.48 A KB KC feature-control frame 的右侧；
  - ALL HEAT STAKE POSTS 的右侧部分；
  - FAI155/156/157 的公共说明结构没有完整包含；
  - FAI156 没有建立到零件 target 的可靠 leader/arrow 关系。

  所以根据当前 Skill 的标准，它不应该是 validated。

  准确原因可以概括为：

  OpenCV假线/假三角形
          ↓
  Qwen把 LP0/G7 当成真实 leader/arrow
          ↓
  semantic 输出 complete=true，但同时 missing target
          ↓
  Main 第一次动作矛盾，被格式检查拒绝
          ↓
  Main 第二次直接改成合法 finish
          ↓
  假 evidence 没碰边界，因此不触发 subagent
          ↓
  Python只验证动作协议，不验证视觉真实性
          ↓
  错误标记为 validated

  所以这次不是“没有箭头也被允许通过”，而是程序错误地相信了一个假箭头，并在格式修复时把视觉问题抹掉了。

──────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 把我们刚才讨论的案例，写入Vision_Agent_Super文件夹里面。命名为test_case_debug_log.md. 成功的和失败的，全
  部，都要写。对每一个，要写明具体的什么因素导致了具体的情况，以及深层次的什么架构导致了这个成功或失败。并对
  每一个给出改良/保持建议


• 我会把这轮已经核验过的 Super/Ultra 案例整理成一份可复用的调试档案：逐例记录证据、直接原因、架构根因、结论与
  建议，并区分“视觉成功”“协议成功”“失败但安全退出”“Ultra长时间阻塞”。我先盘点现有结果和各 Candidate 的状态，避
  免漏掉我们讨论过的成功与失败案例。

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +10 lines
  └ results 13
    0 172 FL validated turns 2 exp 1 initial [2268, 298, 3855, 1344] final [2268, 298, 3855, 1657]
    … +11 lines (ctrl + t to view transcript)
    11 52 AZ protocol_failed turns 1 exp 0 initial [8091, 3469, 9854, 4382] final [8091, 3469, 9854, 4382]
    12 156 EV validated turns 1 exp 0 initial [3878, 3511, 5009, 4440] final [3878, 3511, 5009, 4440]

• Ran find output_ultra_p3 -maxdepth 3 -type f -print | sort | tail -160; printf '\nULTRA RAW TIMES\n'; for f
  │ in output_ultra_p3/raw_responses/candidate_000_mapping.txt output_ultra_p3/raw_responses/
  │ candidate_000_validation.txt output_ultra_p3/raw_responses/candidate_001_mapping.txt output_ultra_p3/
  │ … +5 lines
  └ output_ultra_p3/debug/tiles/tile_014.png
    output_ultra_p3/debug/tiles/tile_015.png
    … +166 lines (ctrl + t to view transcript)
    2026-09-04 15:47:46 405 output_ultra_p3/raw_responses/candidate_002_validation.txt
    MISSING output_ultra_p3/raw_responses/candidate_002_validation_retry.txt

