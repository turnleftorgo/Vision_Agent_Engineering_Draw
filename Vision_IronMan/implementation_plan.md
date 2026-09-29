# 检验截图 Qwen VL 字段提取实施计划

## 实施步骤

1. 扩展 SQLite records schema：保存当前截图哈希/版本、识别状态、结构化结果、错误和模型标识；为已有数据库提供安全迁移。
2. 增加统一异步识别服务：读取有效截图，使用 OpenAI-compatible Chat Completions 发送图文请求，默认模型为 `Qwen3.6-35B-A3B-MLX-8bit`。
3. 编写严格中文提示词和 JSON 校验：区分找到、未出现、不可读、有歧义；字段按可见原文提取，不允许推测。
4. 接入 `refined_crop_ready` 记录创建路径与人工保存裁剪路径。截图内容哈希未变化时跳过；新版本触发推理。
5. 使用线程池限制并发；推理完成后在数据库事务中验证截图版本，只对空值字段写入，保留人工录入内容。
6. 在 records API 返回识别状态和错误摘要，让前端可展示异步处理进度及人工复核提示。
7. 更新技术栈与运行配置文档，记录模型名称、endpoint、模型服务前置条件和字段映射规则。

## 字段映射

- `fai`、`spc`、`description`、`nominal`、`usl`、`lsl`、`hundred_percent`、`dc`、`points` 可由单条截图识别。
- `No.`、Project、Revision、Author、Date 不由单条截图识别；Page、Module 优先取已有后端上下文。
- `SPC截图`来自当前有效截图文件，而非视觉模型输出。

## 可靠性策略

- 识别任务绑定 `record_id + crop_sha256`；写回前再次检查当前哈希，忽略过期结果。
- 同一有效截图仅推理一次；人工裁剪内容没有实际变化时不重复调用。
- 返回值必须通过 JSON 解析与白名单校验；字段键缺失、null 或 unreadable 只影响该字段，按 not_found/unreadable 保存，其他有效字段仍可回写。
- 自动填充只更新数据库中为空的字段；保留已有值及人工编辑。
- 网络、服务或解析错误保存为失败状态，不影响扫描任务和截图操作。

## 验证

- 运行 Python 编译检查和现有后端测试/手动 API 验证；重点验证 schema 迁移、JSON 校验、幂等哈希和过期结果防护。
- 若本机 Qwen 服务未运行，应明确报告真实模型调用未完成；不以模拟输出冒充模型验证。

## 第 1 页 module 元数据增强

1. 在 `qwen_module_fai_pipeline.py` 的 Stage 2 调用处检查 `page_index == 1`，为每个 module 的 Qwen 27B 调用追加纯文字 Project/Revision/Author/Date 提示词；不附加整页或标题栏图片。
2. 扩展同一 JSON 响应根对象，返回 `page_metadata` 四项候选及 found/not_found/unreadable/ambiguous 状态和证据；Project 从 Metric / Title 表格底行读取，Revision/Author/Date 必须取自最新修订历史的同一行；不改变 `fai_clusters` schema。
3. 将 module 候选随 `stage2_module_complete` 事件保存到已有 `page_progress.stage2_modules` JSON。
4. 等第一页全部 module 完成后汇总：Project 独立比对；Revision/Author/Date 作为同一修订记录的整体候选比对。仅采纳带证据且一致的候选；冲突时不写；一致值填充 run 顶部空字段并同步 records。
5. 对用户手动设置的 metadata 字段持久化锁标记，自动提取不得覆盖，包括用户主动清空的字段。
6. 非第一页保持原 prompt 与输出行为。

## 精修进度框与表格行状态

1. Crop 2 进入精修队列时发布带候选 ID、页码和当前原图坐标框的开始事件；每轮方向决策后发布实际 `crop_after`，完成或失败时发布结束事件。
2. 后端按 run、页码、候选 ID 保存活动精修框，并随进度 API 返回；多个候选并行时互不覆盖。
3. 前端在原图坐标层绘制不可交互的蓝紫色流动框；仅活动候选显示，逐轮移动，结束即移除。
4. 表格记录在截图提取模型状态为 queued/processing 时给整行加流动边框，completed/needs_review/failed 时移除；运行主流程结束而提取仍在进行时继续轮询。
5. 新增记录时 Description 留空，精修事件不更新 Description；截图提取只写空字段并保留人工编辑。
6. 移除 SPC 缩略图下的 AI 状态与人工修改标签，不影响缩略图点击及编辑。

## 截图识别强制 JSON mode

1. 截图字段识别的 Chat Completions 请求携带 `response_format: {"type":"json_object"}`，启用本地 oMLX 支持的 JSON/Guided Grammar 输出约束。
2. 保留现有字段解析和校验，验证最终仍是 `fields` JSON 对象；识别结果标记新的 JSON mode 版本，方便和之前的纯提示词结果区分。

## Qwen 35B 请求诊断脚本

1. 在 `tests/diagnose_qwen35b_extraction.py` 新增独立 CLI，默认只读数据库定位最近一次 35B 失败截图，也支持传入其他图像、endpoint、model、API key 和输出目录。
2. 提供生产等价请求、无 JSON mode 对照、JSON mode 且关闭 thinking 对照，用于区分模型视觉识别、JSON 参数、thinking 和服务端处理问题。
3. 保存请求脱敏摘要、HTTP/网络异常、完整响应、assistant 原文、finish_reason、token usage、JSON/业务 schema 解析诊断和总耗时；不得把 image base64 写入日志。
4. 脚本只请求模型并写本地诊断文件，不触发扫描任务、不操作生产记录；报告每项结果并返回有意义的退出码。
