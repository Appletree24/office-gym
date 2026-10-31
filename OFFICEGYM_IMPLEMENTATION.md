# OfficeGym 实现与实验手册

版本：v1.0 实施基线；编写日期：2026-10-31。

本手册面向服务器上的编码 Agent 和实验执行者。目标是实现一个可重置的办公环境、一套独立验证器和长程 Harness，完成真实评测、SFT 与有限预算 GRPO，并从原始数据填写项目描述中的 X。

这里的工具数、模板数、实例数和超参数是建议的首版设计，不是已实现或已测得的结果。最终值以程序导出的 manifest 和评测产物为准。GPU 型号、显存、服务器网络和强模型 API 尚未确定，首先执行硬件及兼容性探针。

## 0 交付物与使用方式

下载包包含本手册、24 个工具的接口清单、14 个模板的设计、世界数据库 SQL、训练和评测配置草案、实例格式、结果格式，以及已实现的硬件探针、结果统计和契约检查脚本。

立即可运行的是 scripts/probe_host.py、scripts/validate_contracts.py、scripts/analyze_results.py 和配套单元测试。包内没有完成 OfficeGym 业务环境、MCP server、模型训练或实验数据。本文出现的 officegym 命令是要由编码 Agent 实现的目标 CLI，不能把它们当作现成软件运行。训练配置文件是本项目的意图配置，需要映射到锁定版本的 verl/TRL 字段，不可直接当作完整上游 YAML 使用。

建议执行：
~~~bash
python3 scripts/probe_host.py --out artifacts/host_probe.json
python3 scripts/validate_contracts.py
python3 -m unittest discover -s tests -v
~~~

然后把本手册和 IMPLEMENTATION_HANDOFF.md 交给编码 Agent，按阶段实现。每阶段独立提交代码和验收结果；不必一次完成所有模块。

## 1 固定第一版的研究问题

主问题：在多阶段任务、外部状态变化和执行中断下，工程 Harness 能否提高正确完成率；SFT 与 GRPO 能否进一步改善同一环境中的执行策略。

第一版只实现会议协调与跟进，应用为邮件、日历、待办和简单文件版本存储。不实现浏览器操作、真实邮箱发送、真实日历接入、多 Agent、用户长期偏好记忆或自由 LLM 用户模拟。

P0 对照组：
- api_simple：强 API 模型，简单 Harness。
- api_full：同一强 API 模型，完整 Harness。
- base_full：原始 4B，完整 Harness。
- sft_full：SFT 后 4B，完整 Harness。
- grpo_full：从同一个 SFT checkpoint 出发的 GRPO 模型，完整 Harness。

P1 增加 base_simple、grpo_simple、rs_sft_full。RS-SFT 与 GRPO 都从同一 SFT checkpoint 分叉，不能把已经 GRPO 的模型当 RS-SFT 起点。

P0 回答强模型上的工程整包收益，以及完整 Harness 下的训练增量。只有补齐 P1，才能讨论训练收益跨 Harness 的可迁移性。即便补齐，也不能推断“若在简单 Harness 下训练会怎样”。

## 2 技术选择与版本锁定

建议 Linux、Python 3.11 或 3.12。业务层使用 Python、SQLite、Pydantic；MCP 使用官方 Python SDK。SFT 使用 Transformers、PEFT、TRL，GRPO 使用 verl + FSDP/FSDP2 + 一个 rollout 后端，首选 vLLM。不要同时调试 vLLM 和 SGLang。

候选学生模型为 Qwen/Qwen3-4B-Instruct-2507：官方模型卡列出 4.0B 参数、非 thinking 模式和工具调用用途。它是便于首轮控制训练轨迹的候选，不意味着已经验证最优。[S1]

创建三个隔离环境：env（业务与 CPU 测试）、serve（推理）、train（SFT/verl）。避免把所有依赖安装到服务器全局环境。选择服务器可用的稳定依赖组合后记录：
- Python、操作系统、GPU/驱动、PyTorch/CUDA。
- 模型 revision、tokenizer/chat_template 哈希。
- verl commit、rollout 后端版本、TRL/PEFT/MCP 版本。
- pip freeze 或 uv.lock、容器 digest（如果使用容器）。
- OfficeGym git commit、dirty 状态、工具 schema、环境/奖励/评测版本。

本包不假定任何未验证的 CUDA 组合。服务端先跑一次工具调用、一次 SFT optimizer step、一次 GRPO rollout+update，再冻结环境。之后不自动升级依赖。

不要把 API key 写入配置或 manifest；只记录读取它的环境变量名称。模型服务监听本机回环地址，除非你明确配置了受控访问。

## 3 硬件与训练可行性探针

先运行 probe_host.py。它只收集平台、磁盘、可选 GPU 信息和已安装库版本，不读取密钥、不下载模型、不改变系统。

实际探针分两道门：
1. 基础门：2–3 个手工任务用于检查工具序列化、模型输出解析、环境运行和成本；随后在开发场景做小批量重复测试校准难度，不能凭原始模型的单次失败决定 GRPO 无效。
2. SFT 后门：在训练池抽取 20 个实例，各采样 4 条轨迹，记录可解析率、成功率、奖励类别、组内奖励方差、耗时及显存，再决定是否进入 GRPO。

资源策略（实施假设，不是显存保证）：
- CPU 可完成全部环境、验证器、参考解和统计工作。
- 4B BF16 权重本体约 8 GB，仅为参数量乘字节数，不包括 KV cache、激活、优化器、参考策略和 rollout 引擎。
- 有限显存先尝试 LoRA、较短训练任务、rollout 并发 1、group size 2，再扩至 4。能完成推理或 SFT，不代表能完成 colocated GRPO。
- 一张卡上按阶段执行 teacher/API 采样、SFT、GRPO和评测；停掉不需要的常驻推理进程。
- 仍然 OOM 时，先检查完整上下文、KV 分配、模型同步峰值与参考模型；再考虑后端支持的 offload、缩短训练任务或更大显存。每次改变配置重新跑 smoke test。
- 不把需要通信调度的大规模分布式训练纳入 P0。

每次探针至少保存峰值显存、生成 tokens/s、每条 trajectory 耗时、组生成成本和有效组比例。实际预算上限由执行者在开跑前填写，不默认允许无限 API 调用。

## 4 代码边界与模块职责

推荐目录：
~~~text
src/officegym/env/          世界状态、时钟、事件、业务操作
src/officegym/tools/        工具 schema、统一 dispatcher、MCP 适配器
src/officegym/tasks/        模板、生成器、参考执行、数据划分
src/officegym/harness/      简单与完整 Harness、上下文、journal
src/officegym/graders/      独立结果检查、过程检查、奖励映射
src/officegym/training/     SFT 数据、verl 适配、RS-SFT
src/officegym/eval/         批量运行、故障控制、结果导出
tests/                     契约、验证器、集成、故障与统计测试
artifacts/                 manifest、任务、轨迹、结果、checkpoint
~~~

环境业务实现只写一套。MCP、参考解法客户端、verl 工具适配器都调用同一 dispatcher。它们对相同状态和相同动作必须产生相同结果与状态哈希。训练 adapter 不得绕过权限、时间推进或计费逻辑。

Agent 进程不可直接访问 SQLite、事件队列、任务私有 ground truth 或评分函数。环境和 grader 是可信进程。第一版的文件权限/进程边界按这一逻辑实现，不开放 shell/SQL 工具给 Agent。

## 5 世界数据结构

schema/world.sql 给出基础 DDL。每个 episode 使用独立 world.sqlite，Harness 自己另有 runtime.sqlite。不要让多个 rollout 共用一个可写世界数据库。

核心实体：
- people：身份、时区、可见权限。
- messages：线程、收发人、类型、正文、结构化请求及附件引用。
- meetings：组织者、参与人、起止时间、业务版本、材料版本和状态。
- documents：文件 ID、不可变版本、内容哈希和访问主体。
- todos：事项、负责人、截止时间、来源和状态。
- approvals：待确认的目标及版本摘要、确认者、状态和有效期。
- scheduled_events：事件到期时间、前置条件、响应规则。
- event_log：不可由 Agent 修改的追加记录，保留 before/after 差异。
- operations：幂等键、请求哈希、业务提交状态、完整结果。
- outbox：实际发送效果，逐收件人记录；所有外发都留在这里。
- board：Agent 可读写的任务看板，内容不作为评分真值。
- meta：模拟时间、环境版本、episode 状态等。

时间统一使用 UTC 整数秒，展示时转换时区。P0 不包含夏令时切换边界；后续加入时必须明确歧义规则。所有排序指定稳定 tie-breaker，JSON canonical 化，禁止用真实当前时间或未记录的随机数决定业务状态。

## 6 工具协议与 24 个工具

完整清单在 contracts/tools.json。建议固定为 24 个，包括业务工具和运行控制工具；最终简历数字由 tools/list 实测并分别报告两类数量。

读类：office.get_context、people.search、mail.search、mail.get_thread、calendar.query、calendar.get、documents.search、documents.read、todos.search、approvals.get、board.read、operations.get、events.poll。

写类：mail.send、calendar.create、calendar.update、calendar.cancel、documents.put、todos.create、todos.update、approvals.request、board.patch。

控制类：clock.wait、task.finish。

统一结果外壳：
~~~json
{
  "ok": true,
  "data": {},
  "error": null,
  "meta": {
    "sim_time": 1800000000,
    "observation_id": "obs_001",
    "tool_schema_version": "tools-v1"
  }
}
~~~

错误使用稳定 code：INVALID_ARGUMENT、NOT_FOUND、PERMISSION_DENIED、VERSION_CONFLICT、IDEMPOTENCY_CONFLICT、BUDGET_EXHAUSTED。错误结果同样计入工具步数，避免通过无限错误调用搜索答案。

episode_id、principal_id 和运行预算由可信执行器注入，不能由模型选择其他 episode。写工具的 operation_id 由 Harness 创建并传递，工具描述向模型解释其用途，但不能要求模型随意猜测重用哪个随机 ID。

同一 operation_id + 同一参数哈希返回原结果，不重复改变世界或产生外发效果；同键不同参数直接冲突。不同 operation_id 的语义重复仍可能产生副作用，应由 grader 检出。不要在数据库上自动去掉所有语义重复，否则重复操作指标会失去意义。

业务语义：
- calendar.create 创建草稿，不自动发邀请。
- calendar.update 修改语义字段后递增版本。
- mail.send 的正式邀请引用 meeting_id/version，环境按规范渲染稳定内容；自由正文用于协调说明，不能改变真实收件人或审批事实。
- approvals.request 发起确认请求，不直接授予确认。授权绑定会议内容 digest；变更参与人、时间或关键材料后旧确认失效。
- documents.put 只增加新版本，P0 不提供富文本编辑器。
- task.finish 是 Agent 自己声明结束，不在工具响应中泄露最终评分或失败清单。
- board 内容可以是错误的；grader 只看业务状态与事件。

必须区分 API 层权限与任务规范：跨账号越权由环境拒绝；部分在应用上合法、但违背用户任务的操作允许在沙箱执行并记录，例如取消自己账号下的无关会议。否则无法测量这一类副作用。被拒绝的尝试另计，不假装真实副作用已发生。

MCP 使用公开 input/output schema。服务器日志写 stderr，stdio 通道只承载协议；结构化结果只选择一个渠道投喂模型，避免 structuredContent 与 text 重复占上下文。官方 SDK 支持以类型定义结构化返回，具体接口按选定版本固定。[S2]

## 7 模拟时钟与用户事件

建议基础规则：
- 每次非重放的工具调用消耗 30 秒模拟时间，包括查询和错误调用。
- 已成功 operation 的幂等重放返回旧结果，不再次施加业务效果；此次 API 请求仍计工具步数和 30 秒模拟时间。
- clock.wait(until) 推进到 min(下一条对当前主体可见的到期事件、until、任务截止时间)，没有事件时也能到达上限。
- 读取不可提供“下一条与成功有关的事件”；只返回主体有权看到的事件，不让调度器成为目标答案 oracle。
- 每次动作前后处理已到期事件，按 (due_at, priority, event_id) 排序。
- 模拟时钟与现实耗时分开统计。重启恢复不自动倒退时间。

用户模拟器只响应结构化请求及模板允许的事实。请求可以得到拒绝、不完整答复或延迟回复；不能通过正文“请直接确认”绕过要求。确认者只会在满足自己可见条件的候选方案上产生 ApprovalGranted 事件。

用户规则知道自己的日历和偏好，不得访问 grader 的最终答案。事件规则的触发依赖当前状态；同一配置在不同 Agent 行为下可以出现不同轨迹，这是正常的。回放要求是初始状态、动作及规则相同，而不是不同策略得到完全相同的回复文本序列。

外部事件包括 ReplyArrived、ApprovalGranted/Rejected、AttendeeChanged、MeetingRescheduleRequested、DocumentVersionPublished、MeetingEnded、MinutesAvailable。已取消会议的旧版本到期事件需要前置条件校验，不应继续产生会后任务。

## 8 模板与实例生成

采用 14 个模板的首版计划：T01–T08 训练、D01–D02 开发、U01–U04 未见模板。细节在 contracts/templates.json。

训练覆盖：基础排期、延迟回复、已有日历冲突、结构化确认、材料版本、单次改期、参会人变更、纪要到待办。
开发覆盖：忙闲冲突与回复延迟组合、改期与材料变化组合。
未见模板覆盖：协调人交接、参与人替换后的重新确认链、会前事项反过来阻塞排期、会后负责人变更及待办修正。所有必需工具和原子能力都在训练模板中存在；未见的是组合或控制流，而非无法调用的新工具。

建议启动数据量（计划）：
- train：800 个实例，8 个模板各 100。
- dev：80 个，两个模板各 40。
- id_test：60 个，训练生成分布的新实例。
- combo_test：30 个，训练生成器显式排除的事件组合。
- unseen_test：40 个，4 个隔离模板各 10。
合计 1,010 个唯一实例。冻结测试共 130 个。可以根据成本减少，但冻结后不得挑选有利实例。

先实现 2–3 个模板和 12 个 probe 实例，再扩展。probe 不进入正式测试集。

为每个实例存 task.json、initial_world.sqlite、private_spec.json、event_rules.json、reference_trace.jsonl、reference_grade.json。private_spec 仅供生成器和 grader；参考解法只拿 public task 与工具客户端。参考解不能查询隐藏事件队列、直接 SELECT 数据库或调用 grader 寻路。

参考解通过并不证明 grader 正确。必须另外对正解加入局部错误并验证 grader 拒绝，以及构造不同但合法的执行路径。

数据指纹有两层：
1. exact_fingerprint：目标、初始世界、事件规则和环境版本 canonical JSON 的 SHA-256；排除 episode UUID、生成时间、split 名和随机 seed。
2. family_fingerprint：保留实质约束与事件结构，归一化无意义人名/ID，标记同源变体，供去重与统计聚类使用。
不同 seed 不保证不同实例；生成后执行跨分区重复检查。combo holdout 列表必须进入训练生成器排除条件。

## 9 深度和扰动的操作化

深度按事先定义的目标依赖 DAG 计算，不按模型实际调用数计算。建议三档：
- D1：准备方案 → 获得确认 → 正式邀请。
- D2：在 D1 中增加参会人回复收集和最新材料核验。
- D3：在 D2 后增加会议结束/纪要获取以及待办交付。

改期不算新增固有深度，它属于外部扰动。扰动设 0、1、2 个需要重新规划的事件，具体事件强度及触发阶段固定。宽度 P0 固定为 1；若采用 2，必须整批一致或分层报告。

深度扫描固定扰动档位，扰动扫描固定深度。任务池不能简单把最复杂模板全部放在最高档；同一基础世界可生成不同深度的条件变体，并记录 scenario_family_id。

如果复用同一基础世界形成多个条件变体，统计时按 scenario_family_id 聚类，而非把它们当完全独立样本。不同档位还应列出模板构成。

高深度可能触发压缩。报告每档的压缩触发率、平均次数和上下文长度，不把曲线解读为“纯规划深度能力”。P1 可以在可容纳上下文的子集比较压缩开/关；超窗不能通过偷偷缩短任务来隐藏。

## 10 简单与完整 Harness 的精确定义

两者共享模型、工具、权限、消息格式、输出解析、步数/时间/token 上限、任务输入，以及已完成交互的追加日志。应用写 API 的幂等能力也相同。

简单 Harness：
- 基础循环：模型 → 一个工具 → 观察 → 模型。
- 不维护 durable intent journal 和结构化任务看板。
- 上下文不足时按固定规则保留任务/政策与最近完整交互，删除最早完整工具回合；不撕开 tool_call/tool_result 对。
- 正常等待后保留已有上下文继续执行。

完整 Harness：
- 额外提供 board 工具使用说明、任务状态机和持久化待执行意图。
- 超阈值时结构化压缩；状态包括目标、约束、可信证据引用、完成项、阻塞项、等待条件、未知结果的 operation。
- 使用同一个执行模型生成摘要，摘要 tokens 和延迟计入预算；摘要标注推测与已确认事实，不把 Agent 自述当事实。
- 重启后由 journal 查询未决 operation，核对已提交效果，并将恢复结果作为新工具观察追加；不能偷偷让强模型帮助 4B 恢复。

为了工具空间一致，简单 Harness 仍能看到 board 工具，但不提供完整模式的使用提示或自动恢复能力；如果基线自行学会使用，记录使用率，不人为阻止。由此主对照是系统整包差异，包括提示与运行机制，不能宣称精确归因到单个组件。

状态机建议 READY → RUNNING → WAITING → RUNNING → DECLARED_DONE；另外有 BUDGET_STOP、INFRA_ERROR 和 RECOVERING。Agent 声明 done 后由 grader 独立判断 success。等待不是结束。

首版不加入长期用户偏好；任务明确提供的限制可以进 board。开发阶段调提示与压缩规则，正式测试前冻结。

## 11 崩溃点与恢复事务

必须分开世界事务和 Harness 本地事务。

完整 Harness 的写操作流程：
1. 解析工具动作，生成 operation_id，把完整请求及哈希写入 runtime.sqlite，状态 PREPARED，并提交。
2. 发送请求至世界 dispatcher。
3. 世界在单个 SQLite 事务内写 operations、业务变化、outbox 和 event_log，并提交。
4. 故障控制器在第 3 步之后、第 5 步之前终止 Agent worker。
5. 正常路径下 Harness 将结果写入 runtime.sqlite 并追加工具观察，状态 ACKED。

恢复流程：读取 PREPARED 的未决请求 → operations.get → 若已提交则补记原结果；若不存在则使用原 operation_id 重试；如参数哈希不符则终止并标记实现错误。不得生成新 operation_id 盲目重放。

故障注入属于实验控制器，不由 Agent 自己调用。只杀 Agent worker，世界进程保留。每个 episode 首次满足“正式邀请已提交但本地未确认”的条件时注入一次，之后关闭，避免无限崩溃。

上下文重建测试：到指定工具边界清空模型短期消息缓存，重建输入。完整模式从 board/journal/压缩记录恢复；简单模式按其预先规定的日志与截断策略恢复。两者均不读取 grader 或参考解。

正式故障评测优先从同一 checkpoint 创建分支，使两种恢复方式面对相同已提交操作；checkpoint 来源与原始模型固定并记录。另报端到端注入实验，报告各组注入触发率，不能只比较成功抵达故障点的幸存轨迹。

这体现持久化执行工程，不应被描述为模型独立学会了事务语义。

## 12 验证器规格与反例集

验证器输入为初始快照、最终快照、完整事件日志、公开政策和 private_spec；输出 GradeResult。Agent 不接触 GradeResult，训练程序才将其映射为奖励。

主要字段：
- outcome_ok：目标对象及版本、参与人、时间、材料、待办均满足任务。
- soft_deviations：预先定义、可容忍的格式或非关键内容偏差。
- hard_violations：未有效确认而发送、错误收件人、无关数据破坏、超出明确消息上限等。
- unexpected_mutations：非允许变更对象/字段的差异。
- terminated_reason、within_budget、declared_done、completion_checks。
- full_success = outcome_ok 且无 soft/hard 偏差、无非预期修改且预算合法。

“确认”检查绑定确认者、目标 digest、版本、有效时间；不能依据字符串。收件人检查使用解析后的 person_id，不只比较显示名。邮件正文与正式邀请结构一致性也要检查，防止结构正确但正文时间错误。

重复外发的语义键示例：(meeting_id, business_version, message_kind, recipient_id, purpose)。同一语义键的真实发送效果超过 1 才算重复；API重试但返回缓存结果不算。合法 reminder 必须在任务政策允许且满足时间条件时使用独立 purpose，不能靠随意改 purpose 绕过计数。

副作用检查依据初始快照加允许变化集合。环境自动事件产生的变化需要标记 actor=environment，不直接算 Agent 破坏；Agent 诱发的不合法变化仍需沿 causal_parent 找到原因。

反例集起步至少覆盖：错收件人、过期确认、未经确认发送、重复邀请、重复待办、取消无关会议、错文档版本、丢失附件、错误时区、超额询问、deadline 前后边界、发错后删除可见记录。每类包含多个合法和错误变体。不存在删除工具的错误可用离线轨迹/快照 mutation 测试 grader，不能声称模型实际发现了该漏洞。

保存人工标签来源、修改位置、预期错误类型和 grader 输出。已知反例检出率只说明对该测试集的覆盖，不能宣称发现所有作弊方式。

## 13 指标分母与结果记录

正式评测前导出并冻结 run_plan.jsonl（每个计划运行一行，包含 group、split、instance_id、repeat_id 和任务属性），不要从已完成结果倒推计划。统计脚本核对全部计划运行，缺失项必须补跑或写明基础设施失败。每次运行写一行 results.jsonl，字段契约见 contracts/run_result.schema.json 和 scripts/analyze_results.py。

必须指标：
- success_rate：full_success 的 episode 数 / 计划运行数。基础设施故障在端到端口径计不成功，并另报可用性和条件成功率；不能静默删除。
- recovery_duplicate_episode_rate：发生至少一次重复业务效果的 episode / 实际注入并进入恢复阶段的 episode。报告分子分母、注入触发率；分母为 0 返回 null，不能显示 0%。
- premature_completion_rate：Agent 明确声明完成但独立完成条件不成立的 episode / 全部 episode。另报条件口径“错误声明 / 所有声明”，两者不要混用。
- negative_detection_rate：被正确拒绝的已知错误验证器样本 / 全部已知错误样本。
- legal_false_reject_rate：被错误拒绝的合法样本 / 全部合法样本。

补充：有效事件响应率、压缩触发率、失败类型、工具步数、input/output tokens、真实耗时、模拟耗时、实际费用（没有价格表则留空）。

先记录原始数值，再计算百分比和百分点。两模型成功率差使用 percentage points，不使用相对百分比替代。失败类型可以多标签；若画互斥分布，需要固定主因优先级。

运行标识必须包含 model checkpoint、harness/config hash、task/family、repeat seed、environment/grader/tool版本、预算配置和故障协议版本。

## 14 统计与曲线

每个固定实例重复 3 次。相同实例和重复编号在各组间配对；不要求不同供应商的同一个 seed 产生相同随机数。随机运行次序，避免 API时间变化与组别混杂。

主要成功率先对实例的 3 次结果求均值，再对实例求宏平均。配对 Bootstrap 重采样实例；每次抽样保留各组的完整配对结果。若存在同源条件变体，以 scenario_family_id 为 cluster。缺组、少重复或版本不一致直接拒绝计算，不通过取交集掩盖缺失。

本包统计脚本提供 95% percentile Bootstrap CI，默认 5,000 次。它不估计训练 seed 的不确定性。未见模板以逐模板结果为主；仅 4 个模板不声称具有可靠的广泛模板泛化区间。

曲线的横轴采用预设 depth_level 或 perturbation_level，每档写清样本量与构成。不同层测试分开汇报，不能混成一个总分宣传。小样本 CI 宽或跨零，只能说当前实验未得到稳定证据。

正式统计前冻结评测版本和 checkpoint选择规则。使用 dev选择 checkpoint，测试仅用于最终报告。保存全部 checkpoints 的训练曲线，不只展示最高测试分数。

## 15 冻结评测规模与成本

按 130 个冻结实例、每实例 3 次、P0五个模型/Harness组，常规完整评测为 1,950 个 episodes；其中强API两组共780个。故障专项另算。

先在 dev 上用 12–20 个实例做 smoke，确认结果 schema、预算和成本，再确定配置里的最终运行上限。不得把 probe 结果直接当正式结果。

故障专项建议 12 个 checkpoint × 3 次 × 2 种恢复方式 × 2 类故障 = 144 个恢复 episodes；可以先只在强模型上完成，若扩展到各学生 checkpoint，需要重新计算规模。

预算同时限制 max_tool_calls、max_sim_seconds、max_generated_tokens、单次生成上限、现实超时和 API费用上限（若可计费）。单步被终止的部分生成也计入消耗。

API如果提供隐藏 reasoning token 统计应保存；若无法观测，明确与本地模型费用口径的差异，不称严格等算力实验。

## 16 SFT 数据与训练

首批从训练池选择 40 个任务、每任务最多 3 条 teacher 轨迹，估算通过率与成本后再扩展；目标可以是 200–800 条已验证轨迹，而不是先承诺一个固定数据量。不足时调整训练任务或生成预算，不混入测试答案。

teacher 使用完整 Harness，但任务必须保证整条轨迹没有触发压缩或故障恢复。P0 训练上下文上限 8,192 token，计入任务提示、工具 schema、全部观察与输出；评测上下文上限 12,288，完整 Harness 在 8,192 附近触发压缩。生成前预留单次输出空间，超出训练长度的 rollout 记预算失败，不强行截断成成功样本。保留这两项的日志标志，触发者不进入 P0 SFT。多轮工具观察和用户回复保留在输入中。

序列格式：system、user、assistant tool_calls、tool observations、assistant…。只对 assistant 自身生成的 token 计算监督损失；用户、工具观察、系统提示为 loss mask 0/label -100。不要把工具返回的真实业务状态训练成模型应自己编造的输出。

TRL 支持会话式数据；assistant_only_loss 等选项取决于 chat template 是否能正确给出 assistant mask。实施时必须打印并人工检查 5 个含多轮工具调用样本的 token/role/mask 对齐，不可只打开一个配置开关就认定正确。[S3]

验证项：
- tokenization 前后工具名称、JSON参数、特殊token、EOS无损。
- assistant mask 非空，工具与用户 mask 为0。
- 输入+输出总长在训练上限内，超长样本丢弃或换短任务，不截断末尾成功轨迹。
- 16条样本能过拟合，随后做独立dev工具调用评测。
- loss下降不能替代环境成功率。

建议初始 LoRA rank16、alpha32、dropout0.05、all-linear；学习率从1e-4起做小探针，1–2 epochs，microbatch1配合梯度累积。它们是起点而非最优结论。记录训练tokens、optimizer steps、wall time和峰值显存。

为避免 RL reference 混淆，推荐把 SFT adapter 导出为独立 merged checkpoint，在导出前后校验固定样本 logits/生成及工具通过率近似一致。SFT baseline、GRPO初始actor、RS-SFT初始模型使用同一 checkpoint 哈希。

## 17 多轮 GRPO 的具体接入

verl 支持状态化工具和自定义 AgentLoop。优先使用其内置 ToolAgentLoop 加 BaseTool；若不足再实现 OfficeGymAgentLoop。不要把离线 OpenAI兼容接口的字符串日志直接当训练轨迹。[S4]

每条 rollout：
1. 从 task snapshot 克隆独立环境。
2. 初始化与评测一致的公开提示、工具schema和看板规则。
3. 使用 verl 提供的生成接口获得 token IDs。
4. 保留实际生成 IDs，解析工具调用，执行同一 dispatcher。
5. 把工具观察作为新的环境token追加，并把对应 loss/response mask设为0。
6. 直到 done、预算耗尽或失败，调用可信grader计算终局奖励。
7. 在最后一次评分完成后释放该rollout的数据库和资源。

同一 GRPO group 从同一个初始实例开始，动作引起的分支可以不同；每条轨迹使用独立环境ID。工具结果、user模板回复、恢复器输出不是actor动作，不能纳入actor loss。

关键验收：
- 原始生成token不可 decode 后再重编码替代；verl 官方指出此类变换会改变policy轨迹。[S4]
- token长度、response_mask、logprob长度一一对应。
- teacher聊天模板转换仅用于SFT；RL记录必须保留学生实际生成。
- 不给模型提供私有ground truth或终局评分明细。
- 同组 rollout 不共享 world 或board，结束后资源能清理。
- reward来自持久化的final快照，不在释放数据库后评分。

LoRA GRPO 的 FSDP 配置和 rollout后端支持已有官方文档；但必须在锁定版本上验证具体字段。不同后端的adapter同步行为可能不同。[S5]

推荐初始意图配置：group size4（资源不足可2）、每update8个有效任务组、actor learning rate1e-6、LoRA rank16/alpha32、KL约束参考为固定SFT checkpoint、max24工具调用、训练context上限8,192 token。先5个update smoke，再20个update探针，最后按资源预算运行，默认最多100个update。不要一次启动长训练。

若使用“把SFT合并为base，再新建RL LoRA”的方式，关闭RL adapter应对应SFT参考，而不是最原始基模。至少对固定输入验证 initial actor 与 reference输出一致。KL系数建议从0.001起检查尺度，不保证适合本任务。

冻结训练Harness版本。生成的summary或自动插入规划如果不属于被训练actor，必须明确mask并计成本；P0通过短任务避免压缩，从而减小实现风险。

## 18 奖励与动态采样

建议 reward-v1：
- 1.0：完全成功。
- 0.3：结果正确、预算合法，只有预定义可容忍的软偏差。
- 0.0：普通失败、无理由不行动、超时或过早宣布完成。
- -1.0：严重违规，覆盖任何其他分数。
结果未完成且仅有软偏差，仍归普通失败 0.0，不额外扣分。严重违规始终优先判定。

软偏差不能包括错误收件人、未授权外发、破坏无关数据或泄露信息。P0 默认令软偏差集合为空；若后续能可靠定义，再按版本启用。首版实际奖励使用 1/0/-1。必要澄清和有效等待是合法中间动作，不因暂时未完成每步扣分。

成功是终局定义，不对每次工具调用给正奖励。有限预算下未完成仍是普通失败；额外的不可解任务若将“正确拒绝”定义为成功，必须在task spec提前定义，不把合法拒绝归为消极不动。

同一采样组总reward相同时，组相对优势为0；这不等于整个训练损失必然无梯度（如仍有KL项）。可以跳过无区分组并补采样，最多采目标组数的4倍。达到上限仍不足时跳过本次更新并记录原因，不能无限循环或把重复轨迹当新组。

补采样必须在同一个actor版本下完成。日志至少含candidate_groups、effective_groups、effective_group_ratio、rollout_tokens、refill_attempts、reward_histogram。不要把动态采样节省的update数当作总算力节省。

P1才启用里程碑奖励，并限制在未严重违规的轨迹，累计不超过0.2，低于结果正确的软偏差档。以终局仍有效的成果为准、同一成果不重复计分。引入后更新 reward_version，重新检查所有已知反例。

## 19 RS-SFT 与训练可迁移性

RS-SFT 从相同SFT checkpoint采样、用同样训练验证器筛选成功轨迹，再做继续SFT。它对比的是同资源条件下两种改进流程，不证明纯算法因果。

预算至少记录并尽量匹配：生成token总数（含失败与补采样）、工具调用数、实际训练token、optimizer updates、GPU-hours与API费用。采样策略随GRPO更新而改变，而RS-SFT可能固定策略，这是无法完全消除的流程差异，要在报告里说明。

完整Harness下比较 SFT、RS-SFT、GRPO。跨Harness实验对相同训练checkpoint同时评估simple/full，回答“已有训练收益是否迁移到简单运行方式”。没有在simple下训练，就不回答那种训练方式会如何。

## 20 版本与 reward hacking 审计

版本分开：env_version、tool_schema_version、task_generator_version、dataset_manifest_hash、harness_version、reward_version、eval_grader_version。训练验证器可迭代，最终测试评分统一冻结。

每5–10个updates抽查高reward轨迹，也抽查随机成功、最高回报却未full_success和低回报合法轨迹。记录漏洞类别、最小复现、旧新grader输出、影响范围和修复测试。

修复训练reward后，先重评已有轨迹以定位影响，再决定从哪个checkpoint继续。不要把不同reward版本的训练分数直接拼成一条可比曲线。最终所有checkpoint都使用相同eval grader重评。

只有模型实际产生的轨迹才能计为“发现X类reward hacking”。人工注入的负样本属于“验证器反例覆盖”，不能与实际发现混写。类目预先按根因合并，避免把不同姓名/日期的同一个漏洞算多类。

## 21 分阶段 CLI 合同与验收

以下命令是待实现的OfficeGym CLI合同，并非本包已安装命令。

阶段A 环境和模板：
~~~bash
python -m officegym.cli generate --config configs/project.json --out artifacts/tasks
python -m officegym.cli reference --manifest artifacts/tasks/manifest.json
python -m officegym.cli validate-dataset --manifest artifacts/tasks/manifest.json
python -m officegym.cli test-graders --suite artifacts/grader_cases
~~~
验收：所有纳入实验的实例参考解通过；重复回放终局哈希一致；跨分区精确重复为0；已知严重错误反例全部检出，合法样本误判逐条修复。

阶段B Harness和基线：
~~~bash
python -m officegym.cli evaluate --experiment api_simple --split dev
python -m officegym.cli evaluate --experiment api_full --split dev
python -m officegym.cli evaluate --experiment base_full --split dev
python -m officegym.cli fault-evaluate --protocol after_world_commit --model strong_api
~~~
验收：结果完整、预算确实生效、故障真实触发、无silent drop；如果simple/full都接近0或1，只在dev校准难度后冻结，不能挑测试子集制造提升。

阶段C SFT：
~~~bash
python -m officegym.cli collect-teacher --split train --max-attempts-per-task 3
python -m officegym.cli build-sft --reject-compacted --verify-masks
python -m officegym.cli train-sft --config configs/sft_plan.json
python -m officegym.cli evaluate --experiment sft_full --split dev
~~~
验收：脱敏后数据可复查、loss mask正确、无测试任务、可导出checkpoint；SFT后训练池探针决定RL是否有可学习信号。

阶段D GRPO：
~~~bash
python -m officegym.cli grpo-smoke --config configs/grpo_plan.json --updates 5
python -m officegym.cli train-grpo --config configs/grpo_plan.json
python -m officegym.cli evaluate --experiment grpo_full --split dev
~~~
验收：每次更新的模型版本、有效组、奖励、KL、梯度及资源消耗可追踪；没有指标造假或以训练reward替代成功率。

阶段E 冻结评测和统计：
~~~bash
python -m officegym.cli freeze --dataset artifacts/tasks/manifest.json --grader eval-v1
python -m officegym.cli evaluate-matrix --priority P0 --splits id_test,combo_test,unseen_test
python3 scripts/analyze_results.py --input artifacts/results.jsonl --plan artifacts/run_plan.jsonl --out artifacts/report --a api_simple --b api_full --split id_test
~~~
分别运行模型对比和测试层。未完成的组、缺失repeat必须补齐或报告，不得静默平均。

上游verl启动命令在固定commit后从其官方示例迁移；记录最终Hydra展开配置。使用本项目的config检查器把意图配置映射到上游，不在手册里伪造未验证的一键完整训练命令。

## 22 现实执行顺序和止损

建议按闸门推进，而非保证某个工期：
1. 环境探针：2–3任务、参考执行、强模型与4B基础调用。
2. CPU核心：事件、版本、操作幂等、验证器与反例测试。
3. 工程实验：简单/完整Harness、强模型基线、上下文重建和一个崩溃点。
4. SFT：小数据过拟合检查、正式数据、独立dev评测。
5. SFT后RL探针：解析率、奖励差异、资源成本。
6. 有限预算GRPO、可选RS-SFT、冻结评测和报告。

停止或回退条件：
- 参考解失败/不可复现：先修环境，不进行训练。
- grader误放严重反例：先修评分，不收集teacher数据。
- SFT后仍无法产生合法工具操作：检查数据、mask、模板与任务难度，不直接堆GRPO。
- 补采样预算耗尽且有效组过少：调整训练池难度或SFT，不无限补采样。
- 训练reward涨但冻结开发成功率下降：审计reward hacking或训练分布偏移。
- 达到预定资源预算仍无稳定增益：保留checkpoint与负面结果，报告当前配置的局限。

负面结果不影响已完成的环境、验证器、Harness与评测工程价值；不要为了填正向百分比继续选择有利seed。

## 23 从真实产物填写项目中的 X

| 项目字段 | 真实来源 | 报告要求 |
| --- | --- | --- |
| 工具X个 | MCP tools/list与工具manifest | 区分业务、看板及控制工具 |
| 模板X个、实例X个 | 冻结dataset manifest | 列出split及唯一指纹数量 |
| 重复操作X%到X% | 固定故障协议的事件日志 | 明确按episode还是按业务动作，报告分母 |
| 错误检出率X% | 人工标注负例集grader结果 | 附各类计数，不宣称覆盖所有漏洞 |
| 合法误判率X% | 合法替代解法集 | 样本数小则同时报告原始计数 |
| 强模型工程提升 | api_full减api_simple | 同一模型版本，配对CI，单位百分点 |
| 4B训练提升 | grpo_full减sft_full；另报相对base | 不把SFT与RL收益混在一起 |
| 高于RS-SFT X点 | grpo_full减rs_sft_full | 列明采样和训练资源差异 |
| reward hacking X类 | 真实训练轨迹审计台账 | 排除仅人工构造案例和重复根因 |

最终报告固定包含：方法与版本、数据分层、各组指标及分母、两条曲线、故障专项、验证器反例覆盖、训练资源和失败案例。简历只引用可追溯结果，不把建议启动规模写成完成数字。

## 24 官方资料与适用范围

[S1] Qwen3-4B-Instruct-2507 model card：
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

[S2] MCP Python SDK与结构化工具结果：
https://github.com/modelcontextprotocol/python-sdk
https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/servers/structured-output.md

[S3] TRL SFTTrainer：
https://huggingface.co/docs/trl/sft_trainer

[S4] verl AgentLoop与扩展指南：
https://verl.readthedocs.io/en/latest/advance/agent_loop.html
https://verl.readthedocs.io/en/latest/extend_guide.html

[S5] verl LoRA与GRPO：
https://verl.readthedocs.io/en/latest/advance/ppo_lora.html
https://verl.readthedocs.io/en/latest/algo/grpo.html

[S6] AppWorld task generators：
https://github.com/StonyBrookNLP/appworld/blob/main/guides/developing_new_task_generators.md

以上来源支持模型和框架接口的选型；本文的任务、奖励、数据规模、预算和验收规则是OfficeGym的实施设计。正式实验使用锁定版本，不能把持续更新的latest文档当作实验版本。

## 25 关键实现伪代码与接口验收

### 25.1 工具 dispatcher

统一入口建议为 dispatch(context, tool_name, arguments, operation_id=None) → ToolResult。context 包含可信注入的 episode、主体、预算、时钟；工具 arguments 不允许覆盖这些值。

写操作顺序：
1. 检查现实超时与剩余调用预算；记录此次调用，按固定规则推进模拟时间。
2. 校验 schema、主体可访问性及幂等请求哈希；请求哈希包含工具名、规范化参数和主体，不包含传输时间戳。
3. 已提交的同键请求返回缓存结果；同键不同参数返回冲突。返回外壳可以包含当前模拟时间，缓存业务 data 保持不变。
4. 开启事务，在事务内再检查幂等键，避免并发重试产生竞态。
5. 检查业务版本及字段约束；计算业务变化、结果和需发送的 outbox 记录。
6. 同一事务提交 operation 结果、业务状态、外发效果及不可变事件。
7. 处理由此次动作触发的到期事件，返回公开观察。

P0 每个世界串行处理动作，读写都走统一时钟。clock.wait 不额外叠加固定 30 秒，而是按等待规则跳转；若 until 早于当前时刻则返回参数错误并按普通错误调用计时。非 wait 的每次合法、错误或幂等重放调用都消耗 30 秒。task.finish 也受时间和工具预算约束。

当时钟到达相同时刻：先处理已到期世界事件，再执行动作，再处理动作新产生的已到期事件；所有事件有稳定排序。deadline 为含端点的最后可操作时刻；推进后晚于 deadline 的动作不得产生新业务效果。临界规则写进单元测试。

缓存结果不能让预算计数回滚。语义重复允许产生两个真实 outbox 记录，由 grader 判断违规；幂等重放只有一个真实效果。不要把 SQLite 的物理可靠性误写成模型行为改进。

### 25.2 Harness 与上下文

建议标准接口：
- Policy.generate(messages, tool_schemas, remaining_budget) → AssistantTurn。
- Environment.call(tool, args, operation_id) → ToolResult。
- ContextManager.prepare(messages, board_snapshot, budget) → ModelInput。
- Grader.evaluate(initial, final, event_log, task_spec) → GradeResult。

一次模型响应 P0 只执行一个工具调用；如果返回多调用，先校验，再按固定顺序串行执行并分别计步。是否支持多调用在 teacher、学生和两类 Harness 中必须一致。不可解析输出有有限重试次数（建议 2），重试消耗生成 token；持续失败作为明确终止原因。

压缩后的输入由固定任务说明/政策、工具 schema、结构化摘要、最近完整交互组成。摘要必须有来源引用，例如 message_id、meeting_id/version、event_seq。保留任务版本和当前未决 operation，不允许摘要把“请求确认”写成“已确认”。

压缩前保存旧片段哈希、摘要、源引用、压缩模型/采样参数和 token 消耗。摘要损坏或缺字段时回退到最近可信状态并记录失败；不使用 grader 修正摘要。全局生成预算包含摘要生成；API 输入 tokens 另行累计，若需要严格费用上限由实际价格和用量执行。

训练长度检查必须使用实际 chat template、工具 schema 和原始 token IDs。训练中达到长度上限是预算终止，终局按失败/违规评分；不能默默启用压缩，也不能只保留成功结尾。SFT 成功示范则整条丢弃超长样本。

### 25.3 GRPO 适配器的最小 smoke test

按顺序执行并保存日志：
1. 一个实例、两条 rollout、零 optimizer update：确认数据库隔离、动作解析、token/mask 长度与奖励。
2. 相同实例人工构造成功、普通失败、严重违规的确定性轨迹：奖励严格按版本排序；这只验证软件，不能当模型结果。
3. 关闭采样或固定可复现种子，验证同策略初始 actor/reference 的 logprob 与 KL 数值合理；不要求不同推理后端逐 bit 相同。
4. 一个 optimizer update：参数确实变化，工具结果 token 对 actor loss 的权重为零，结束后无残留 world worker。
5. 五次 update：检查 actor 版本同步，下一组使用新权重，补采样仍在当前 update 的同一 actor 版本内。
6. 导出 checkpoint，经常规评测入口调用；不能只在训练器内部能推理。

配置映射必须在服务器版本锁定后生成 resolved_train_config.yaml，并保留原始上游配置与覆盖参数。至少核对：
- algorithm 是否为 GRPO，group size 是否确实作用于每个相同 task；
- rollout agent loop 与工具配置是否指向 OfficeGym adapter；
- actor、reference、rollout 的模型 checkpoint/revision；
- LoRA rank/target、初始 adapter、权重同步方式；
- prompt/response/context 的长度字段在当前后端到底表示什么；
- reward function 与最终数据库生命周期；
- 动态有效组过滤位于哪个采样循环，以及 KL 的实现方式。

这里不提供未经服务器验证的“通用 verl 命令”。完成映射后，交付一个 scripts/train_grpo.sh：只接受已锁定配置，先检查资源与预算，再执行经过 smoke test 的启动命令。[S4][S5]

## 26 生成器与运行台账的细化

id_test 的 60 个实例可按五个条件格各 12 个规划：
- 深度扫描：(D1,P1)、(D2,P1)、(D3,P1)；
- 扰动扫描：(D2,P0)、(D2,P1)、(D2,P2)。
中心格 (D2,P1) 共用，合计 5 格。固定宽度、背景邮件量、文档大小和消息上限；其他实质约束尽量配对。不可实现的模板/条件组合不要硬套，记录各格模板构成并限制解释。

combo_test 的 30 个实例可以五格各 6 个，但每格样本量较小，主要作为额外证据。unseen_test 按 4 个模板各 10 个单独报告。是否沿用同一基础世界派生变体必须在 manifest 标明，并采用 family 聚类。

组合留出先建立显式事件签名集合，例如 [参会人变更,材料更新] 的同时出现及规定的触发顺序；训练允许各原子事件，但拒绝整个留出签名。开发集可有其他组合，不能反复根据组合测试得分修改规则。细分触发顺序是否属于“同一组合”在生成器版本中写定。

所有任务都必须有合法解。深度、扰动数量只是实验参数，不能只靠文件名声称任务确实有相应依赖：生成器输出 dependency DAG，参考解产生每个阶段所需的可见证据。若存在跳过阶段也合法完成的路径，应修正任务依赖或降低深度标记。

正式计划中的 repeat_id 固定为 0、1、2。run_plan.jsonl 的每行至少包含：
~~~json
{"group":"api_full","split":"id_test","instance_id":"实际实例ID","repeat_id":0,"scenario_family_id":"实际场景族ID","template_id":"T04","depth_level":2,"perturbation_level":1}
~~~
这段只说明结构，禁止当实验输入。计划生成于测试执行之前，保存 SHA-256；断点续跑以完整 run key 匹配。

网络/基础设施重试最多建议 2 次，仅对明确的可重试错误执行；失败轨迹与成本保留。一次动作响应不确定时必须查幂等结果，不能作为全新动作盲目重试。若整个 episode 重跑，保留 attempt_id 及原始失败，不把随机重跑直到成功当作一次评测。最终结果行按提前规定的重试政策填写，不从多次结果中选最好一次。

统计脚本严格检查计划、配对、重复次数和版本；但无法证明运行计划创建在实验之前，也无法独立证明结果来自模型。需要保留模型原始请求响应、工具/事件日志、各阶段 manifest 和命令日志，支持抽样追溯。

## 27 每个阶段应交付的真实产物

| 阶段 | 产物 | 能支持的结论 |
|---|---|---|
| 环境 | tools/list 导出、DDL、重放 hash、业务测试 | 实际工具数量和确定性 |
| 数据 | task manifest、指纹报告、参考解轨迹、split 清单 | 实际模板/实例数量与隔离 |
| 验证器 | 带人工标签的正负案例、逐案例结果 | 对已知反例的检出和误判 |
| Harness | 普通/故障运行台账、operation/outbox 对账 | 工程整包收益与恢复重复率 |
| SFT | 经筛选的数据 manifest、mask 检查、checkpoint hash | 示范训练后的执行变化 |
| GRPO | rollout、reward、effective-group、KL、checkpoint | 有限资源下的策略训练结果 |
| 对照 | 冻结 run plan、完整 results、配对统计、成本 | 在本任务分布中的收益及边界 |
| 审计 | 漏洞最小复现、grader 版本差异、全基线重评 | 实际观察到的 reward hacking 类别 |

项目展示至少准备一个从中断前到恢复后可追踪的真实案例：原始任务、故障发生位置、world 已提交的效果、恢复查询、最终评分及成本。可以制作演示视频，但视频不能替代完整评测。

如果 GRPO 未优于 SFT 或 RS-SFT，保留负面结论，解释可观测证据，例如有效组不足、奖励稀疏、训练预算或分布差异。没有足够证据时写“本轮未观察到稳定增益”，不要把猜测写成已证实原因。


