# OfficeGym

按《OFFICEGYM_IMPLEMENTATION.md》持续实施的可重置办公任务研究环境。目前具备 24 个工具、14 个模板生成器、独立评分器、官方 MCP 适配器、预算化 Harness、真实 worker-kill 恢复探针，以及已通过真实双步优化器探针的 SFT 后端和待验证的 RL 训练入口。正式数据冻结、模型训练及完整评测尚未完成；没有模型效果或训练提升结论。

Python 3.11+，当前 CPU 实现仅依赖标准库。项目目录中执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-env.lock
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m officegym.cli probe-reference --out artifacts/reference_new
PYTHONPATH=src .venv/bin/python -m officegym.cli probe-fault --out artifacts/fault_new
.venv/bin/python scripts/export_versions.py --out artifacts/versions.json
```

探针使用全新输出目录，保留原始轨迹及数据库。`probe-reference` 检查基础排期、延迟审批和确定性重放；`probe-fault` 真实终止 Agent worker，保留世界服务并核对外发效果。它们均属于软件验证，不属于真实模型实验。

策略文件访问隔离探针：`PYTHONPATH=src .venv/bin/python -m officegym.eval.isolation_probe --out artifacts/NEW_ISOLATION_PROBE`。配置式 ChatPolicy 评测会启动独立 exec 进程，在空 chroot 中降为 UID/GID 65534，清空补充组和环境，启用 no_new_privs、禁止 dump，并限制继承描述符为 stdin/stdout/stderr。HTTP 请求由可信父进程代理，凭据、世界 socket、数据库、tokenizer 和评分器保留在可信侧；生成与摘要均走有界 JSON 消息通道。需要 Linux root 及 chroot/setuid/setgid 能力；隔离失败保留 INFRA_ERROR，不回退到未隔离执行。每次运行保存 policy_isolation.json，正式结果导出与教师提取校验该证据。自定义 transport 被标记为软件 fixture。此边界目前覆盖 ChatPolicy 评测/教师路径；原生 verl worker 与完整故障模式仍需单独隔离验收。

墙钟预算从 episode 入口统一计时，包含运行准备、世界／策略进程启动和恢复间隔；工具返回后再次检查，超时的 task.finish 不计正常完成。工具 RPC 与策略启动的超时上限取剩余预算。恢复日志绑定单调时钟起点与 boot／时钟命名空间身份；跨重启或身份变化拒绝续跑，不重置预算。结果中的 runner_wall_seconds 还包含策略停止后的清理和评分开销。


原始设计与命令合同见 OFFICEGYM_IMPLEMENTATION.md；阶段要求见 IMPLEMENTATION_HANDOFF.md；剩余范围见 IMPLEMENTATION_STATUS.md。configs 中 null 预算不表示无限许可，训练配置仍需按实际安装版本映射及验证。统计脚本要求完整运行计划和真实结果，禁止将示例或探针数据写成正式评测。

MCP（官方 SDK 1.30.0，依赖见 requirements-env.lock）：先以 `serve-world --world EPISODE.sqlite --socket /private/world.sock` 启动固定世界，再以 `serve-mcp --socket /private/world.sock` 启动 stdio 适配器。两者命令均使用 `PYTHONPATH=src .venv/bin/python -m officegym.cli` 前缀。写工具 operation_id 由可信 Harness 放在 MCP `_meta["officegym/operation_id"]`，不放入模型工具参数。客户端仅消费 structuredContent；stdio stdout 专用于协议。

安装后可在任意工作目录使用命令，无需设置 `PYTHONPATH`：

```bash
python -m pip install .
officegym probe-reference --out /tmp/officegym-reference-new
```

核心运行时只依赖标准库；MCP 使用 `python -m pip install '.[mcp]'`。SQL 和工具契约随 wheel 发布，仓库中的 `schema/*.sql` 与 `contracts/tools.json` 是包内资源的符号链接。wheel 已在隔离环境、仓库外验证参考探针及持久化 journal。训练和推理依赖仍须按手册在各自环境中锁定并通过真实 smoke test。

RS-SFT 分支提供 `python -m officegym.training.rs_sft collect --help` 和 `build --help`：前者要求原始 SFT merged checkpoint、导出记录、完整 Harness 采样配置和明确的采样上限；后者复用成功轨迹筛选与真实 tokenizer 对齐流程。继续训练使用现有 `train-sft`，计划中设置 `training_branch: rs_sft`，`model_path` 必须仍指向同一原始 SFT checkpoint。GRPO 阶段、不同权重或不一致分支标签会被拒绝。当前已验证分支约束，尚无真实 RS-SFT 采样或训练结果。

统计实现也可通过 `python -m officegym.eval.analysis --help` 使用，原 `scripts/analyze_results.py` 保留为兼容入口。Checkpoint 选择入口为 `python -m officegym.eval.selection freeze --help` 与 `select --help`：先冻结候选导出记录、checkpoint 内容哈希及仅含 dev 的三次重复计划，再提交所有候选的完整 dev 结果。规则按实例宏平均成功率排序，同分依次选择更早的 optimizer step 和确定性哈希顺序；保存全部候选分数，不接收测试 split。冻结时序仍需命令日志等外部证据支持。

SFT/RS-SFT 的解析训练计划还必须提供 `checkpoint_interval_steps`，取值为 1 到 `max_steps`。入口按固定步数保存且不自动删除周期 checkpoint，追加 `training_events.jsonl` 并保留 `trainer_log_history.json`；周期 checkpoint 和最终 adapter 会记录文件哈希。它们仍需 merged 导出及等价性验证后才能进入 dev 选择流程。

非 base 的 `record_export` 还要求原始 checkpoint 路径、`training_events` 和 `optimizer_step`。等价性报告须绑定导出模型哈希、原始 checkpoint 文件哈希和步数。选择规则 v2 会复核这些来源；已有日志允许继续追加，但对应 checkpoint 和已绑定的日志前缀不能修改。文件溯源检查不替代实际 logits、生成和工具行为等价性实验。

LoRA merged 导出入口：`python -m officegym.training.merge --help`。解析计划须提供本地 base/adapter、基础模型内容哈希、训练事件与步数、精确依赖版本、设备/精度、时间上限及 shard/export 字节上限。命令在独立进程中合并并保存独立模型和 tokenizer；保存前检查 tensor 字节数，保存后核对实际导出大小，超限保留失败现场。`merge_result.json` 明确保留 `passed: false`，直到固定样本 logits、生成和工具行为等价性另行验证；不能把完成合并当作导出验收通过。

数值等价性入口：`python -m officegym.training.equivalence --help`。计划绑定 merge 计划/结果和固定 token probes 的哈希，显式限制案例数、输入/输出 token、forward logits 元素数、时间以及数值/生成匹配阈值。它比较每个输入最后一个位置的完整词表 logits 和贪心生成 token IDs，重新加载 merged 模型并保存两侧原始 tensor 证据。`passed_numeric` 只表示这一层检查；`passed` 保持 false，工具行为等价性仍须另行完成。

工具行为等价性入口：`python -m officegym.training.tool_equivalence freeze --help` / `finalize --help`。先冻结数值证据、dev 计划、阈值和预定结果路径（此时结果文件不得已存在），再使用有界评测入口分别采集 adapter 与 merged 模型结果。前者的 `model_version` 绑定数值报告中的 `source_checkpoint_files_sha256`，后者绑定 `exported_checkpoint_sha256`。最终验收重导出原始运行、重新评分保存的世界，要求完整三次重复、相同 tokenizer／采样／预算、无基础设施错误、无新增重复效果或严重违规。仅数值与工具检查都通过的报告才能用于 `record_export`；部署是否确实加载对应权重仍需服务端运行证据支持。

当前开发候选为 `artifacts/dataset_development_v10`，尚未冻结。训练仍为 8 个模板各 100 条，其中 T04 按五个条件格各 20 条抽样，ID 每格 12 条使用独立种子；控制扫描固定背景邮件、初始材料大小、参与人数、时间窗口、截止/监控时间和消息上限。`workflow_count=1` 指一条会议工作流，实际 DAG 宽度另行计算：D1 为 1，D2/D3 为 2。计划、结果及分层 CSV 都保留 `dependency_width`，跨宽度差异不能解释成纯深度效应；扰动档位指预设外部事件数，不等同于所有策略实际重规划次数。


真实 tokenizer 的 CPU 探针使用独立环境（Python 3.12/Linux），不安装 Torch 或下载模型权重：

```bash
python3 -m venv .venv-tokenizer
.venv-tokenizer/bin/python -m pip install --require-hashes -r requirements-tokenizer.lock
.venv-tokenizer/bin/hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --include 'tokenizer*.json' vocab.json merges.txt config.json --local-dir artifacts/qwen-tokenizer
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-tokenizer/bin/python -m officegym.training.tokenizer_probe --tokenizer artifacts/qwen-tokenizer --revision cdbee75f17c01a7cc42f958dc650907174af0554 --out artifacts/NEW_TOKENIZER_PROBE --dataset-manifest artifacts/dataset_development_v10/manifest.json
```

[该固定 revision 的官方文件](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)中的模板没有 assistant generation 标记。SFT 入口只对已锁定模板哈希添加 generation 区间标注，仍使用 Transformers 原生 assistant mask，并逐样本检查渲染文本和 token IDs 完全不变。推理模板保持原版；未知模板继续要求其提供有效原生 mask。生成的训练 manifest 记录原始和标注模板哈希，token_alignment.jsonl 供逐 token 审阅。当前真实 tokenizer 检查覆盖 14 个模板的 19 条参考变体；全量长度审计发现 800 条训练参考轨迹中 40 条超过 8,192 token（此前为 60／159 条）。它们是手工参考解的长度，不能当作真实教师采样结果；正式数据冻结前仍须校准长度／预算并验证真实服务的一致性。


当前公开事件协议为 tools-v2-public-events：events.poll 返回可见业务事实、对象标识和游标，完整 before/after 快照留在可信审计库，读取对象正文仍须通过对应权限检查。参考解跨监控／跟进阶段共享事件游标，并缓存已观察到的纪要和负责人变更。Harness 版本为 harness-v6-public-event-observations，旧教师／准备数据须重新验证；开发候选 v10 与 v9／v7 的全部任务、初始世界、私有评分约束和事件规则输入哈希一致。


参考解版本 reference-v3-batched-availability 在单个协调线程向全部参与人询问，逐人核对 provided 回复并增量读取分页；审批先利用请求返回的 pending 状态再等待。v10 的任务输入与 v9 完全一致。当前仍有 40 条训练参考解超过 8,192 token，另有 40 条超过 24 次工具调用上限，需继续校准。

参考轨迹的真实 mask 检查可复跑：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-tokenizer/bin/python -m officegym.training.reference_tokenizer_probe --manifest artifacts/dataset_development_v10/manifest.json --tokenizer artifacts/qwen-tokenizer --revision cdbee75f17c01a7cc42f958dc650907174af0554 --out artifacts/NEW_REFERENCE_MASK_PROBE`。它每个模板／深度／扰动组合检查一条参考轨迹，并保存原始 token ID 哈希及 mask 来源；输出不作为真实教师数据或训练结果。


CUDA 基础运行时已在 A800 上验证 PyTorch 2.8.0+cu128／CUDA 12.8。独立环境安装使用固定哈希 lock 和有界安装器（需要全新环境／输出路径）：

```bash
.venv/bin/python scripts/bootstrap_serve_runtime.py --target .venv-serve --out artifacts/NEW_SERVE_BOOTSTRAP
PYTHONPATH=src .venv-serve/bin/python -m officegym.eval.cuda_probe --out artifacts/NEW_CUDA_PROBE
```

安装器总时限 600 秒，wheel／环境／临时目录上限分别为 6／12／6 GiB；[版本来源为官方 PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)。GPU 探针要求设备空闲，子进程限时 60 秒，检查 FP32／BF16 矩阵乘法误差及有限、非零反向梯度，限制 PyTorch allocator 分配至 256 MiB。该指标不包含驱动上下文，也不证明模型显存可行性。探针输出保留 model_compatibility_pending；完整模型服务、业务任务评测和优化器 smoke 仍须后续验证。当前 .venv-serve 已安装 Torch／Transformers 探针栈，本地模型 HTTP 服务与 vLLM 栈仍待搭建和验证。


固定 Qwen revision 的完整文件现已下载并通过内容哈希校验，真实 BF16 4B 模型已在 A800 上完成单次工具生成探针。可复跑入口（输出目录须全新）：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.fetch_model --out artifacts/NEW_QWEN_FETCH --seed-tokenizer artifacts/qwen-tokenizer
# 仅在确认先前下载进程已终止后，使用同一目录和 --resume 续传；模型来源与资源上限保持不变。
.venv-serve/bin/python -m pip install --require-hashes -r requirements-serve.lock
PYTHONPATH=src .venv-serve/bin/python -m officegym.eval.model_probe --fetch-root artifacts/NEW_QWEN_FETCH --out artifacts/NEW_MODEL_PROBE
```

下载上限为 10 GiB／1,800 秒，保留失败记录与分片，校验 LFS SHA256／Git blob SHA1 后才接受；结果中的 result.json 是协调器资源验收，verified.json 仅表示文件校验。模型探针使用原始模板、全部工具 schema、最多 128 个生成 token、120 秒和 16 GiB PyTorch allocator 上限。当前真实结果为 2,415 输入／16 输出 token，正确生成 board.read({})；业务工具执行、完整任务成功率和训练效果仍须另行验证。requirements-serve.lock 是实际探针栈，不替代后续 vLLM／verl／TRL 版本冻结。

本机真实服务探针：`PYTHONPATH=src .venv-serve/bin/python -m officegym.eval.vllm_probe --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 --out artifacts/NEW_VLLM_PROBE`。服务环境锁见 requirements-vllm.lock；命令要求 GPU 空闲，以有限时间／显存和临时环境密钥启动 loopback 服务。固定 vLLM 0.11 的 serial_vllm 适配层在采样前将明确的单工具请求限制为一个调用。v4 已验证 72 次真实调用的 token 计数和单工具协议，但 T01–T03 均在 24 次调用后预算停止，完整成功 0/3；这属于协议可行性记录，不能作为正式基线。当前 Harness 为 v9，历史准备数据需要重新生成。

ChatPolicy 默认 `tool_choice=auto`，使用模型原生工具格式；配置可显式选择 `required` 作诊断，选择会记录在结果与 provenance 中。原生对照探针 artifacts/vllm_native_auto_probe_v1 已消除此前的反复读板模式，T01–T03 执行 10/10/8 个工具后声明完成，独立评分仍为 0/3。30 次响应的 token 计数均一致；其中两次纯文本响应经有限重试恢复。离线审计命令：`PYTHONPATH=src .venv/bin/python -m officegym.eval.protocol_audit --runs RUN_DIRECTORY --out NEW_AUDIT.json`；严格单工具审计不会把这种可恢复的纯文本响应算成通过。

真实训练后端探针：`PYTHONPATH=src .venv-train/bin/python -m officegym.training.backend_probe --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 --out artifacts/NEW_BACKEND_PROBE`。有效依赖锁见 requirements-training.lock；当前 .venv-train 通过显式 .pth 复用服务环境，仅覆盖冲突的训练依赖。探针使用清楚标注的手写协议样本，真实完成两次 TRL/PEFT LoRA 更新，并核对 assistant-only loss、非零有限梯度、参数变化与原始文件哈希；不产生教师数据或正式 SFT baseline。v1 实测 wall 38.79s、峰值分配约 12.60 GiB；16 样本过拟合、任务效果与导出等价仍待完成。

真实合并／数值探针：`PYTHONPATH=src .venv-train/bin/python -m officegym.training.export_probe --profile fp32 --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 --backend artifacts/real_optimizer_backend_probe_v1 --out artifacts/NEW_EXPORT_PROBE`。它只使用已标注的后端 checkpoint，不能登记为正式 SFT 导出。BF16 对照已发现：两项生成完全一致，但最大 logit 差异 0.4375/0.375 超过预设 0.25，故保留为未通过。FP32 是独立精度配置，容差预设为 1e-4；完整验收还需要任务级工具等价。

独立 FP32 v1 实测也未通过其预设容差：两项生成完全一致，但最大 logit 差异约 1.50e-4/1.38e-4，高于 1e-4。完整资源测量已完成（203.53s、观察峰值 22,540 MiB），数值／工具等价门禁仍关闭；命令在数值拒绝时返回非零退出码并保留全部证据。

数值报告现为 numeric-v2：保存可用标准库重算的原始 JSON logits／token IDs，工具等价门禁不再直接信任 `passed` 或最大误差字段。显式 `logit_rtol` 默认 0；启用时按每个 reference logit 计算 `atol + rtol*abs(reference)`，可配真实禁用 adapter 的负对照。有界复测入口：`PYTHONPATH=src .venv-train/bin/python -m officegym.training.numeric_probe --plan FROZEN_PLAN.json --out NEW_DIRECTORY`。新三项输入的前瞻性 FP32 检查有两项通过、四个 logit 超限，整体仍未通过；CPU 原始数据复算与负对照验证均已完成。旧失败记录不变，正式导出门禁仍关闭。

正式训练现在先记录 formal-training-admission-v1，绑定准入数据、实际样本、解析后的配置、父权重、阶段和优化器步数上限。正式导出使用 checkpoint-export-v3-formal-admission；选模、RS-SFT 分叉和分支校验都会重新检查来源文件。后端探针仍可做数值诊断，但缺少准入事件的真实 checkpoint-2 已确认无法登记为正式 SFT。GRPO 也必须实际加载声明的已验收 SFT 路径；原生 verl checkpoint 事件与运行时验证仍待完成。

verl 保存事件桥接已实现，绑定固定 commit 的 8 个源码哈希，要求 `trainer.use_v1=false`、同步保存、保留全部 checkpoint，以及 run 内的绝对输出路径。首版每个训练步只允许一个 mini-batch／PPO epoch，并核对有限梯度和独立更新计数，避免把 global_steps 直接当成优化器调用次数。控制器测试已验证完整分片记录和缺失分片拒绝；测试 worker 是明确标注的 CPU fixture，尚不代表真实 FSDP/GRPO 验收。具体必需字段见 configs/grpo_plan.json 的 intent 合同和 IMPLEMENTATION_STATUS.md。

现代 RL 环境单独位于 .venv-rl（Torch 2.13/CUDA 13、Transformers 5.16、vLLM 0.32），依赖与源码配方见 requirements-rl.lock、configs/rl_runtime_source.json；旧服务／SFT 环境保持不变。真实单 GPU FSDP2 保存恢复探针：`PYTHONPATH=src .venv-rl/bin/python -m officegym.training.fsdp_probe --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 --checkout artifacts/verl_checkout_v1/repo --out artifacts/NEW_FSDP_PROBE`。v3 实测 4B 权重完成原生 checkpoint 保存、故意扰动、恢复及下一步更新，恢复分支与未中断分支逐字节一致（44.14s）。这是两条手写协议样本上的后端连续性验证，尚非真实 GRPO 或正式训练结论。

原生 verl 循环的 CPU 协议探针（需要已锁定的 `.venv-rl`）：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_loop_probe \
  --instance artifacts/dataset_development_v10/train_atomic/T01_100000 \
  --model artifacts/qwen3_4b_base_cdbee75_v1/model \
  --out artifacts/NEW_NATIVE_PROTOCOL
```

该入口运行真实上游 AgentLoop、tokenizer、工具和评分器，但生成端提供人工 token/logprob，不是模型 rollout。它验证并发世界隔离、原始 IDs/mask/logprob 保留、容量拒绝与资源释放，限时 180 秒；当前 v9 的十四项检查通过。正式 GRPO、actor/reference 与权重同步验收仍待完成，详见 IMPLEMENTATION_STATUS.md。

新 RL 环境中的真实基模/vLLM 原始 token 接入探针：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v10/train_atomic/T01_100000 \
  --out artifacts/NEW_NATIVE_MODEL_PROBE
```

要求 GPU 空闲，限时 600 秒／36 GiB 显存／2 GiB 产物；最多两条轨迹、十次生成、1,024 个生成 token，无付费 API。当前 v2 完成十次真实生成，原始 IDs/logprobs 与 mask 对齐，两条轨迹均在诊断轮次上限结束、奖励为零。它不运行 Ray rollout server、参数更新或正式 GRPO，也不证明 worker 文件系统隔离。

原生控制器版本绑定探针（真实 DataProto／固定上游分批方法，更新计数为人工 fixture）：

```bash
PYTHONPATH=src .venv-rl/bin/python scripts/probe_native_actor_dispatch.py \
  --checkout artifacts/verl_checkout_v1/repo --out artifacts/NEW_ACTOR_DISPATCH
```

正式启动配置须将 `trainer.officegym_starting_actor_hash` 绑定到已验收的 SFT 导出哈希。控制器按完成更新次数标记每条轨迹，循环逐次检查 backend 版本，更新前拒绝混合版本 batch。这些检查不证明实际权重已同步；真实 Ray／权重同步／KL／GRPO 更新验收仍待执行。

原生有效组／补采样控制流探针：

```bash
PYTHONPATH=src .venv-rl/bin/python scripts/probe_native_effective_groups.py \
  --checkout artifacts/verl_checkout_v1/repo --out artifacts/NEW_NATIVE_GROUPS
```

该探针使用真实 DataProto 与固定上游 `fit`，奖励和后端为明确标注的人工构造，限时 180 秒。当前 v5 验证了四倍候选上限、同版本补采样、零方差组跳过且不推进更新计数、有效组原始 token 保留和重复轨迹拒绝。它没有执行真实 optimizer／KL 或证明物理权重同步；完整联训验收仍待完成。

使用真实上游 Ray server／负载均衡器／原生客户端的同一探针：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --backend ray --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v10/train_atomic/T01_100000 \
  --out artifacts/NEW_NATIVE_RAY_PROBE
```

当前 `native_ray_vllm_loop_probe_v1` 完成十次真实生成，独立 token 审计通过，HTTP 认证检查为 401／200，结束后 GPU 空闲。仍使用固定基模、standalone 模式和声明的第 0 步标签；不证明训练权重同步、KL、optimizer 更新或 worker 文件系统隔离。上限与 direct 探针相同，Ray 日志也计入产物预算，无付费 API。

已归档真实轨迹的原生 worker 后处理／数学验收（CPU，不重新生成）：

```bash
PYTHONPATH=src RAY_USAGE_STATS_ENABLED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-rl/bin/python -m officegym.training.worker_postprocess_probe \
  --source artifacts/native_ray_vllm_loop_probe_v1 \
  --model artifacts/qwen3_4b_base_cdbee75_v1/model --out artifacts/NEW_WORKER_REPLAY
```

当前 v3 验证真实 worker 构造、24 工具加载、原始 IDs/logprobs、padding、输入元数据和奖励落点。另有明确标注的人工奖励及 logprob 叶节点梯度对照；它们不是模型训练数据或模型参数更新。独立优势助手现与固定上游的样本标准差／epsilon 约定一致。完整 Ray worker 生成、实际权重同步与 GRPO 联训仍待验收。

完整原生 worker 生成路径（真实 Ray CPU worker + 实际 GPU server）：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --backend ray-worker --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v10/train_atomic/T01_100000 \
  --out artifacts/NEW_LIVE_WORKER_PROBE
```

当前 v2 完成两个并发原生循环、十次真实生成，返回的原生 batch 与逐轨迹请求／终局归档匹配；独立审计通过。仍是固定基模的零更新诊断：奖励均为零，无有效组，不证明 KL、实际权重同步或 GRPO 学习。默认 AgentLoopManager、文件系统隔离和恢复仍待验收。上限为 600 秒／36 GiB／2 GiB／1,024 个累计生成 token，无付费 API。

默认 AgentLoopManager 的完整调度／合并路径（两个原生 worker）：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --backend ray-manager --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v10/train_atomic/T01_100000 \
  --out artifacts/NEW_NATIVE_MANAGER_PROBE
```

当前 v2 通过默认 manager 分批、两个真实 worker 生成、结果合并与独立 token 审计；十次生成、奖励均为零，无 optimizer 更新。上限与前述原生探针相同。实际权重同步、KL、正向 GRPO 联训、文件系统隔离及恢复仍待验收。

Native GRPO operational curve data can be reconstructed without GPU work:

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.curves \
  --events /absolute/run/training_events.jsonl \
  --usage /absolute/run/rollout_token_usage.json \
  --out /absolute/run/curve_data
```

This writes `summary.json`, `updates.csv`, `sampling_batches.csv` and `checkpoints.csv`. Sampling counters are cumulative within each refill window and are differenced before summing. The summary distinguishes reported sampling tokens, ledger usage bounds, actor loss tokens, context tokens, protocol-verified updates and unfinished attempts/windows. Omit `--usage` when no ledger summary exists; missing cost remains explicitly unrecorded. These are operational records, not evaluation success curves, formal training admission or evidence of learning improvement. GPU-hours/API/tool costs are not inferred from absent records.

Native sampling records also carry known terminal-reported tool calls and the number of trajectories whose tool counter is missing. Curve summaries difference these window-cumulative counters and label exactness only for recorded sampling trajectories. Selected-batch tool costs are a subset of sampling cost, not additional tool executions; calls from trajectories that never return a batch remain outside this accounting.

`python -m officegym.training.merge_weight_audit --plan /absolute/merge_plan.json --result /absolute/merge_result.json --out /absolute/weight_audit` audits supported FP32 vanilla linear-LoRA weights on CPU. It streams tensor rows, rechecks source fingerprints, verifies untouched weights and reports FP64 merge-reference error against a normal-FP32 roundoff envelope. Its 300s/4GiB-RSS/10MiB-output supervisor bounds are fixed. Passing this diagnostic does not pass logits/generation/tool equivalence or formal export admission.

A merge plan may explicitly set `merge_accumulation: "float64"` for supported default-adapter vanilla linear LoRA with FP32 output. This computes delta and base addition in FP64 before one final FP32 conversion; the default remains the native PEFT merge. `officegym.training.precise_merge_probe` adds owned GPU/disk/artifact supervision for this candidate path. The existing three-input numerical gate still fails on the inspected backend fixture even with single-round weights; neither this option nor the weight audit is an accepted SFT export.


教师 API 费用网关已提供配置模板 [teacher_gateway.template.json](configs/teacher_gateway.template.json)。操作者必须先填写固定 HTTPS 地址、请求模型与供应商响应中的精确模型版本、有限请求/token/费用/墙钟预算，以及带日期和来源的十进制价格上界；模板不能直接启动。上游密钥和本地网关密钥分别通过模板指定的两个环境变量提供，密钥值不写入配置或账本。启动命令：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.teacher_gateway --config /absolute/operator-gateway.json
```

将打印的 loopback endpoint、相同 model/model_version 和本地 gateway_key_env 填入已有 ChatPolicy 采样配置。上游凭据只由可信网关读取。认证后的 GET `/budget` 可读取账本汇总。SQLite 账本在发出请求前原子预留全额输入上限及本次输出上限的费用；并发调用共享账本，重启保留消耗且拒绝改变原账本策略。确认的 usage 可释放未用预留；超时、缺失 usage 等未知消耗保留全额预留。模型版本或 token 上限违约记录后阻止后续请求。网关不自动重试、不跟随重定向，并限制启动后的服务寿命。

此费用口径是所声明 token 价格的上界，不是供应商账单。供应商必须以 prompt_tokens/completion_tokens 返回全部计费 token（包括推理 token），且遵守所填上下限；字节及 framing 检查不是通用 tokenizer 证明。价格需覆盖所选服务档位等 token 费用，税费、非 token 费用和供应商违约不在保证范围内。这里只验证了软件上游、真实本地 HTTP、并发及重启协议；尚未核实真实供应商兼容性。软件 fixture 网关会标记来源，真实 ChatPolicy 拒绝用它生成 teacher 数据。真实 teacher 选择、费用预算及采样仍待提供，当前没有付费调用。


费用账本可在网关运行时或结束后导出，无需提供任何密钥，也不会启动服务或发出请求：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.teacher_gateway --config /absolute/operator-gateway.json --snapshot-out /absolute/NEW_SNAPSHOT_DIRECTORY
```

命令只读打开原账本，通过 SQLite backup 导出已提交的一致视图（含 WAL 中的提交），保存 `ledger.sqlite`、`pricing_policy.json` 及含文件 SHA256 的 `report.json`。输出目录必须全新；源账本缺失或配置指纹不符会拒绝执行。快照保留请求哈希、已确认 token usage、错误类型和未知预留，不含请求正文或密钥值。它用于核对声明价格下的消耗；仅有快照不能证明真实模型调用、商业价格或账单。后续调用不会改变已导出的快照。


首批 teacher 采样按手册先选择 40 个训练任务。离线选样无需模型服务、密钥或费用预算：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.pilot --manifest artifacts/dataset_development_v10/manifest.json --out artifacts/NEW_TEACHER_PILOT
```

选样器先按模板轮询，再在模板内按深度／扰动层轮询，用显式 seed 与任务 ID 的 SHA256 确定层内顺序。它只读取训练任务身份及层信息，不以参考成绩或长度筛选，不保证有限名额覆盖所有层。`selection.json` 保存任务清单、选择规则、seed、模板计数和源 manifest 哈希；当前 40 任务计划及独立核对位于 `artifacts/teacher_pilot_selection_v1`，T01–T08 各 5 个。此计划基于尚未冻结的开发数据，不表示数据已获正式准入。

`collect-teacher` CLI 默认 `--max-tasks 40 --selection-seed officegym-teacher-pilot-v1`，按同一规则选样，在首次调用模型前保存完整选择清单及其哈希。扩大采样需显式提高 `--max-tasks`，仍受必填的 episode／生成 token 预算和每任务最多三次尝试限制；较小预算可能只执行计划的一部分，汇总分别记录 selected_tasks 和 attempted_tasks。使用相同 manifest、seed 和任务上限可复现离线计划。底层 `collect()` 未传 max_tasks 时保留全训练池行为，供现有显式预算的 RS-SFT 调用使用。真实采样仍须先完成教师配置与实际费用预算；离线计划不提供通过率或费用估计。


SFT 五样本人工 mask 审查包：

```bash
PYTHONPATH=src .venv-tokenizer/bin/python -m officegym.training.mask_review --data /absolute/BUILT_SFT_DATA --tokenizer /absolute/LOCKED_TOKENIZER --revision FULL_REVISION --out /absolute/NEW_REVIEW_DIRECTORY
```

命令核对数据／tokenizer 哈希，选择五条各含至少两次工具调用及观察的会话，用原生模板重算并逐项比对保存的 ID、角色、mask 和 label。生成可独立打开的 `review.html`、逐 token 的 `tokens.jsonl`、绑定源文件与页面哈希的 `packet.json` 和空白 `human_checklist.json`。页面绿色行参与 loss；词表片段展示原始 tokenizer 符号，不能当作逐 token 独立解码的自然文本。原始会话和工具 schema 可展开核对。检查用户／系统／工具观察为 label -100、assistant 监督范围及工具调用边界；不足五条多轮样本、源会话改变或重算不一致会拒绝输出。

生成器不填写人工判断、不改数据 manifest，也不授予训练准入。当前 `artifacts/mask_review_authored_reference_v2/review/review.html` 是使用真实锁定 tokenizer 的参考夹具演示，包含 24,562 个 token，非 teacher 数据，人工审查状态仍为 pending。SFT 状态字段现已绑定人工检查记录；仅将字段改为 passed 会被训练准入拒绝。


人工检查五条样本后，在 `human_checklist.json` 填写 reviewer、带时区的 reviewed_at_utc、逐样本的三项布尔判断和具体 notes，只有全部通过时才将其 status 改为 passed。以下命令只校验并登记已经填写的记录，不生成判断：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.training.review_admission --data /absolute/BUILT_SFT_DATA --packet /absolute/REVIEW/packet.json --checklist /absolute/REVIEW/human_checklist.json --verify-only
PYTHONPATH=src .venv/bin/python -m officegym.training.review_admission --data /absolute/BUILT_SFT_DATA --packet /absolute/REVIEW/packet.json --checklist /absolute/REVIEW/human_checklist.json
```

登记以原子替换更新数据 manifest 的审查状态与 packet／checklist 路径和哈希。数据身份指纹排除这两个审查登记字段，原 manifest 副本仍随审查包保存，避免哈希循环。SFT、RS-SFT 和训练血缘准入都会重新检查五个不同多轮样本、逐 token 对齐、所有检查项、审查包及源文件哈希。旧版审查包需要重新生成，不能仅补填 passed。记录属于具名人工声明，系统不提供审查者身份认证，也不能证明实际检查行为；软件测试中合成的记录明确标为夹具，不作为真实人工审查证据。


Teacher 和 RS-SFT 采样现在会在保存的 resolved_sampling_config 中强制 `context_policy: training_full_history`，要求完整 Harness、`max_context_tokens <= 8192`。每次生成前用锁定 tokenizer 计算完整原始会话和工具 schema 的输入长度，并预留 max_output_tokens；超限直接记 CONTEXT_OVERFLOW，不调用模型、不摘要、不丢旧轮次。工具返回追加后再用无额外生成前缀的原生模板检查整条会话，最终 task.finish 的工具观察也纳入检查；超长不能 DECLARED_DONE 或进入 successful_runs。普通评测默认仍使用既有完整／简单 Harness 上下文策略。构建 SFT 时继续独立核对最终 assistant mask 及训练序列长度。

`artifacts/teacher_pilot_history_feasibility_v1` 使用真实锁定 tokenizer、8,192 上限及 512 输出预留检查首批 40 个任务的作者参考解：38 条完整放得下，2 条在后续生成前超出预留条件，后两条参考解也均超过 24 次工具调用。保留全部选中任务和失败记录，没有按参考成绩／长度重选；实际 teacher 可能产生不同长度轨迹，不能将此结果写成 teacher 成功率或不可解证明。


十六样本过拟合测量已接入 `train-sft` 的可选 `overfit_check` 配置。将 [sft_overfit_criteria.template.json](configs/sft_overfit_criteria.template.json) 填成数值对象后放入完整训练配置的 overfit_check 字段；模板本身不能运行。四项条件分别为最终 assistant 平均因果交叉熵上限、token 准确率下限、相对初始损失降幅下限，以及每次测量的有限秒数。准确率／降幅阈值范围为 0–1，必须在训练前确定并随 resolved 配置保存。

启用后，数据必须恰好十六个不同样本，仍需通过正常的 teacher 来源、长度、mask 和五样本人工审查检查。同一任务的不同轨迹可作为不同样本，不能用相同样本重复补齐数量。训练前后以 eval/no_grad 模式、相同原始会话和 assistant labels 测量，按因果移位仅累计被监督 token，逐块计算 FP32 交叉熵及 token／全序列准确率，保存 overfit_before.json、overfit_after.json 和 overfit_result.json，并写入训练证据。训练后检查还要求实际发生 optimizer update；失败阈值会如实保留为 passed=false，独立 dev 评测始终另记 pending，不自动视为环境成功。

测量在样本及计算块边界检查独立时限和训练总时限；单次模型 forward 不能被该 Python 检查抢占，运行时仍应沿用外层资源 watchdog。当前 `artifacts/overfit_measurement_cpu_fixture_v1` 只验证作者构造 logits 的数值、mask 与模式恢复，没有加载真实模型权重、没有 optimizer update，也不满足手册的真实十六样本过拟合要求。真实完整训练路径尚待获得经审查的 teacher 数据后执行。


SFT 构建 manifest 现在为每条接收样本保存源运行目录中 `result.json`、`trajectory.jsonl`、`policy_isolation.json` 的 SHA256。SFT／RS-SFT 训练预检及后续训练血缘验证会重新核对这三个文件，并通过原有 teacher 提取器重新验证当前 Harness／grader 版本、完整成功、隔离证据及无压缩／恢复等条件，要求提取出的原始会话与 conversations.jsonl 完全一致。还核对 source_run、轨迹哈希、样本 ID、mask／label、长度、token 总量，拒绝重复运行路径和重复 token 序列。

因此需要保留每条样本的原始运行证据及原路径；旧版数据 manifest 若缺来源文件清单，必须从保留的原始运行重建并重新人工审查，不能直接补一个 passed 字段。此检查证明保存的来源证据和训练数据相符，不提供独立供应商身份认证，也不能从可编辑本地文件证明真实外部调用；软件夹具的结构验收不能写成真实 teacher 证据。


训练运行时 tokenizer 也需通过全量核对。`train-sft` 在加载模型权重前，从模型目录只读加载 tokenizer，比较数据 manifest 中的 revision、词表哈希、原始／mask 模板哈希、template_kwargs 和消息规范化方式，并对全部训练会话重算原始 token ID、role、assistant mask 和 labels，不限于人工检查的五条。即使未监督的用户／工具输入 token 被改变，也会拒绝训练。结果保存在 runtime_tokenizer_check.json 及训练事件中，包含实际检查的样本、token、因果监督位置和数据哈希。

训练配置可单独指定 tokenizer_revision；未指定时使用 model_revision。例如 RS-SFT 的模型 revision 可为已合并 SFT checkpoint 的身份，而 tokenizer_revision 仍锁定原 tokenizer revision。若数据使用了 template_kwargs，训练配置也须显式匹配。检查后仅在缺少 pad token 时沿用既有 EOS padding 回退。训练墙钟预算现在从入口开始，覆盖预检及 tokenizer 核对，并在模型初始化前后检查；单次长操作仍需外层 watchdog 才能硬中断。

artifacts/runtime_tokenizer_admission_probe_v1 用真实锁定 tokenizer 对五条参考夹具重算全部 24,562 token／2,515 个因果监督位置，错误 revision 或改动模板均被拒绝。此探针没有加载模型权重、没有 teacher 调用、没有人工批准，不能作为完整 SFT 训练验收。


SFT／RS-SFT 正式导出现在还要求逐步优化器调用证据。生产训练与原生回调探针共用回调：优化器调用前检查所有可用可训练梯度的有限性，以 FP64 累加平方范数，并记录梯度 tensor／元素数；调用返回后检查 Accelerate 的 optimizer_step_was_skipped，记录配对完成事件，再核对 Trainer global_step。非有限／缺失／不支持的梯度在调用前拒绝，跳步保留失败证据并停止；有限零梯度可以记录，不据此宣称学习提升。

导出器只读取源 checkpoint 事件之前的冻结前缀，要求从 1 到声明 step 的全部尝试／完成事件按序匹配，拒绝跳步、重复、悬空尝试及 checkpoint 前的训练失败。正数 step、loss 日志或权重文件不能单独证明完成更新。旧的工程探针仍保留，缺这些证据的记录不能补写成正式训练。`artifacts/sft_update_native_cpu_fixture_v1` 仅以 1,016 参数随机微型 Transformer 验证真实 HF Trainer 两次调用及 NaN 拒绝，非 OfficeGym 正式训练，未加载预训练权重或 teacher 数据。
