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

当前开发候选为 `artifacts/dataset_development_v25`，尚未冻结。训练仍为 8 个模板各 100 条，其中 T04 按五个条件格各 20 条抽样，ID 每格 12 条使用独立种子；控制扫描固定背景邮件、初始材料大小、参与人数、时间窗口、截止/监控时间和消息上限。`workflow_count=1` 指一条会议工作流，必需 DAG 宽度另行计算：当前受控五条件格均为 1，可选材料观察不计入因果先决节点。计划、结果及分层 CSV 都保留 `dependency_width`，跨宽度差异不能解释成纯深度效应；扰动档位指预设外部事件数，不等同于所有策略实际重规划次数。


真实 tokenizer 的 CPU 探针使用独立环境（Python 3.12/Linux），不安装 Torch 或下载模型权重：

```bash
python3 -m venv .venv-tokenizer
.venv-tokenizer/bin/python -m pip install --require-hashes -r requirements-tokenizer.lock
.venv-tokenizer/bin/hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --include 'tokenizer*.json' vocab.json merges.txt config.json --local-dir artifacts/qwen-tokenizer
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-tokenizer/bin/python -m officegym.training.tokenizer_probe --tokenizer artifacts/qwen-tokenizer --revision cdbee75f17c01a7cc42f958dc650907174af0554 --out artifacts/NEW_TOKENIZER_PROBE --dataset-manifest artifacts/dataset_development_v25/manifest.json
```

[该固定 revision 的官方文件](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)中的模板没有 assistant generation 标记。SFT 入口只对已锁定模板哈希添加 generation 区间标注，仍使用 Transformers 原生 assistant mask，并逐样本检查渲染文本和 token IDs 完全不变。推理模板保持原版；未知模板继续要求其提供有效原生 mask。生成的训练 manifest 记录原始和标注模板哈希，token_alignment.jsonl 供逐 token 审阅。当前真实 tokenizer 检查覆盖 14 个模板的 19 条参考变体；全量长度审计发现 800 条训练参考轨迹中 40 条超过 8,192 token（此前为 60／159 条）。它们是手工参考解的长度，不能当作真实教师采样结果；正式数据冻结前仍须校准长度／预算并验证真实服务的一致性。


当前公开事件协议为 tools-v2-public-events：events.poll 返回可见业务事实、对象标识和游标，完整 before/after 快照留在可信审计库，读取对象正文仍须通过对应权限检查。参考解跨监控／跟进阶段共享事件游标，并缓存已观察到的纪要和负责人变更。Harness 版本为 harness-v9-native-auto-tool-choice，旧教师／准备数据须重新验证；开发候选 v16 的公开业务任务、初始世界、私有评分约束、事件规则和参考动作同 v15；外层 DAG 元数据已显式修正。


参考解版本 reference-v3-batched-availability 在单个协调线程向全部参与人询问，逐人核对 provided 回复并增量读取分页；审批先利用请求返回的 pending 状态再等待。v16 的公开业务任务同 v15，DAG 元数据与宽度标注有修正。当前仍有 40 条训练参考解超过 8,192 token，另有 40 条超过 24 次工具调用上限，需继续校准。

参考轨迹的真实 mask 检查可复跑：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv-tokenizer/bin/python -m officegym.training.reference_tokenizer_probe --manifest artifacts/dataset_development_v25/manifest.json --tokenizer artifacts/qwen-tokenizer --revision cdbee75f17c01a7cc42f958dc650907174af0554 --out artifacts/NEW_REFERENCE_MASK_PROBE`。它每个模板／深度／扰动组合检查一条参考轨迹，并保存原始 token ID 哈希及 mask 来源；输出不作为真实教师数据或训练结果。


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
  --instance artifacts/dataset_development_v25/train_atomic/T01_100000 \
  --model artifacts/qwen3_4b_base_cdbee75_v1/model \
  --out artifacts/NEW_NATIVE_PROTOCOL
```

该入口运行真实上游 AgentLoop、tokenizer、工具和评分器，但生成端提供人工 token/logprob，不是模型 rollout。它验证并发世界隔离、原始 IDs/mask/logprob 保留、容量拒绝与资源释放，限时 180 秒；当前 v9 的十四项检查通过。正式 GRPO、actor/reference 与权重同步验收仍待完成，详见 IMPLEMENTATION_STATUS.md。

新 RL 环境中的真实基模/vLLM 原始 token 接入探针：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v25/train_atomic/T01_100000 \
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
  --instance artifacts/dataset_development_v25/train_atomic/T01_100000 \
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
  --instance artifacts/dataset_development_v25/train_atomic/T01_100000 \
  --out artifacts/NEW_LIVE_WORKER_PROBE
```

当前 v2 完成两个并发原生循环、十次真实生成，返回的原生 batch 与逐轨迹请求／终局归档匹配；独立审计通过。仍是固定基模的零更新诊断：奖励均为零，无有效组，不证明 KL、实际权重同步或 GRPO 学习。默认 AgentLoopManager、文件系统隔离和恢复仍待验收。上限为 600 秒／36 GiB／2 GiB／1,024 个累计生成 token，无付费 API。

默认 AgentLoopManager 的完整调度／合并路径（两个原生 worker）：

```bash
PYTHONPATH=src .venv-rl/bin/python -m officegym.training.native_inference_probe \
  --backend ray-manager --fetch-root artifacts/qwen3_4b_base_cdbee75_v1 \
  --instance artifacts/dataset_development_v25/train_atomic/T01_100000 \
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
PYTHONPATH=src .venv/bin/python -m officegym.training.pilot --manifest artifacts/dataset_development_v25/manifest.json --out artifacts/NEW_TEACHER_PILOT
```

选样器先按模板轮询，再在模板内按深度／扰动层轮询，用显式 seed 与任务 ID 的 SHA256 确定层内顺序。它只读取训练任务身份及层信息，不以参考成绩或长度筛选，不保证有限名额覆盖所有层。`selection.json` 保存任务清单、选择规则、seed、模板计数和源 manifest 哈希；当前 40 任务计划及独立核对位于 `artifacts/teacher_pilot_selection_v13`，T01–T08 各 5 个。此计划基于尚未冻结的开发数据，不表示数据已获正式准入。

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


上一阶段导出评分器反例集扩展为 26 例：8 个合法、18 个负例，历史输出在 artifacts/grader_cases_v6 和 artifacts/grader_report_v7。新增七个确认授权案例覆盖无期限、发送在到期前一秒／恰好到期／到期后一秒、错误确认者、发送后才批准及旧版本批准。到期边界用显式 offline_snapshot_mutation 记录 expires_at 的修改位置，因为模型没有修改确认有效期的工具；其余事件通过真实业务工具和调度器执行。独立 SQL oracle 核对了实际 grant/send/expiry 时间、确认者、版本及 digest。

run_suite 的 summary.json 新增 category_counts，逐类记录合法／负例数量、已知负例检出、合法误拒和标签不一致；确认授权变体合并在 approval_authorization 类。该阶段 26 例标签均匹配，18/18 已知负例被拒绝、0/8 合法误拒。这是软件反例覆盖，不是模型发现的 reward hacking 或全面漏洞检出率；其他类别仍需继续增加合法／错误变体。


上一评分版本为 eval-v5-todo-field-scope／reward-v5-todo-field-scope。修复了“同标题／同会议即可创建额外错误负责人或截止时间待办”的误放；新增创建必须匹配任务或纪要明确指定的 title／assignee_id／due_at，公开 TodoOwnerChanged 仅授权对应会议、标题的新负责人。状态仍是结果约束，不能把正常创建但未完成的准备事项直接当作无关创建。原 v4 下两条作者反例为成功，v5 下均保留 outcome_ok=true 但因 unrelated_data_mutation 判失败；合法控制通过，完整复现与修复台账在 artifacts/todo_field_scope_reproduction_v1，模型发现漏洞数为零。

v11 是从 v10 保留的参考世界重新评分所得，尚未冻结；1,010 个实例的五类输入／动作文件哈希和参考最终状态哈希均不变。独立重新执行结果在 artifacts/dataset_v11_reference_replay：1,010 个参考解、最终哈希和操作审计全部通过。首批 40 任务计划 v2 绑定新 manifest，选中任务与 v1 相同。旧数据／评分证据仍保留，不能把 v4 结果当作 v5 正式结果复用。开发数据可用以下命令在新目录重新评分；它拒绝冻结数据、非空 WAL、损坏的来源哈希、来源目录内的输出及已存在的输出目录：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.tasks.regrade --manifest /absolute/OLD_DEVELOPMENT/manifest.json --out /absolute/NEW_DEVELOPMENT --max-wall-seconds 180
```

上一阶段反例集 artifacts/grader_cases_v7／grader_report_v8 含 30 例（9 合法／21 负例），标签全部匹配；新增 todo_creation_scope 分类包含合法待办、完全重复、额外错误负责人和额外错误截止时间四例。这仍是软件反例覆盖，真实 teacher 数据、人工审查及正式实验尚未完成。


上一评分为 eval-v6-todo-status-evidence／reward-v6-todo-status-evidence。修复“唯一所需待办被取消后仍满足任务”的误放：required_todos 未明确指定 status 时，cancelled 不计完成；明确指定 status 时按原条件判断。open／blocked／done 控制均仍可满足未限定状态的要求。重复检测按非状态业务字段单独执行，取消其中一条不能隐藏仍存在的重复待办。原 v5 反例、v6 结果和作者反例台账位于 artifacts/todo_status_reproduction_v1，未声明为模型发现。

上一开发数据 v12 已重新评分并重新执行全部 1,010 个参考解，成绩／状态哈希／操作审计全部通过；五类输入及参考动作文件哈希与 v11 相同。首批计划 v3 绑定新 manifest，任务身份不变。上一评分器反例集为 artifacts/grader_cases_v8／grader_report_v9：34 例（11 合法、23 负例），全部标签匹配。旧 v5 评分证据保留为历史，正式复用前须重新评分；数据仍未冻结，真实 teacher、人工审查及正式实验尚未完成。


上一评分为 eval-v7-meeting-creation-scope／reward-v7-meeting-creation-scope，开发数据为 v13。修复了完成任务之外额外创建／邀请无关标题会议仍获成功的漏洞：新增会议须在任务标题／组织者范围内，或有明确本地修改许可／可信环境创建证据；实际发出的邀请还单独按不可变会议版本检查任务标题。发错邀请后改标题、取消会议不能抹去外发副作用。本地创建许可不会自动授权无关邀请，同任务未邀请的备选草稿仍合法；未给定标题的旧通用 spec 不会被额外推断标题约束。

完整复现与修复台账在 artifacts/meeting_scope_reproduction_v1，包含三个误放变体和合法备选草稿控制，均为作者构造的软件反例。v13 的 1,010 个参考解重新评分及重新执行全部通过，五类输入／动作文件与最终状态哈希同 v12；计划 v4 保持原四十任务身份。上一反例集 artifacts/grader_cases_v9／grader_report_v10 含 38 例（12 合法、26 负例），标签全部匹配。旧 v6 评分证据保留为历史；真实 teacher 采样、人工审查和正式实验仍未完成。


上一评分为 eval-v8-invitation-participants／reward-v8-invitation-participants，开发数据为 v14。正式邀请的完整 participant_ids 集合必须匹配发送当时的任务授权：以原任务成员为基准，只应用同一会议且 MailSent 之前的可信 AttendeeChanged 事件。标题正确、收件人都在全局允许名单内仍不足以授权额外的部分成员会议；取消会议、恢复正确成员也不能抹去已发错的邀请。变更前的合法旧成员邀请不会被最终成员集合追溯判错。

作者反例和修复台账位于 artifacts/invitation_members_reproduction_v2，早期 v1 夹具构造错误也已保留。v14 的 1,010 个参考解重新评分及执行全部通过，输入／参考动作和最终状态哈希同 v13；首批计划 v5 的四十任务身份不变。上一反例集 artifacts/grader_cases_v10／grader_report_v11 含 42 例（14 合法、28 负例），全部匹配；独立 SQL 审计验证了变更前后两次发送的成员和事件顺序。真实 teacher、人工审查及正式实验仍待完成。


当前评分为 eval-v9-invitation-timing／reward-v9-invitation-timing，开发数据已更新为 v16。每次实际发出的邀请都按不可变版本检查允许时间窗口、要求时长、发送时尚未开始的会议时间（恰好开始时允许），并且必须遵循该会议在发送前收到的最新改期要求。最终修正不能抹去已发错的时间；后来收到改期要求也不会追溯否定此前合法邀请。

artifacts/invitation_time_reproduction_v1 保存 v8 下通过的窗口前／后各一秒、错误时长及忽略已收到改期要求后再修正的作者反例与 v9 结果；补充的过去时间夹具明确标记为未提交 v9 开发阶段检查，不是正式版本结果。最新反例集 artifacts/grader_cases_v11／grader_report_v12 共 51 例（18 合法、33 负例），全部标签匹配，包含窗口两端、恰好开始时发送、正常改期等合法控制及独立 SQL 时间审计。

v15 的 1,010 个参考解重新评分／重新执行全部通过，五类输入／参考动作及最终状态哈希同 v14，最终规则重放的评分逐条等于保存值。首批计划 v6 保留原四十任务身份。数据仍未冻结；这些均为软件验证，真实 teacher、人工审查与正式实验仍未完成。


连续改期与取消的生命周期回归探针：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.env.lifecycle_probe --out artifacts/NEW_LIFECYCLE_PROBE
```

探针通过实际工具连续创建 v1、改期到 v2／v3、发送邀请并幂等重放最后一次发送，分别验证正常结束、结束前取消、结束后等待纪要时取消。每个场景执行两遍，要求最终状态及完整工具观察日志哈希一致；旧 v1／v2 结束事件必须取消，最新结束／纪要各最多产生一次，取消后的待发纪要不能出现，环境也不能自动创建待办。重放只复用业务 data，新的 observation_id 和 30 秒模拟耗时仍须记录。

当前 artifacts/meeting_lifecycle_probe_v3 的六轮共 108 次工具调用及独立 SQL 审计通过：正常结束有一次结束／一次纪要，结束前取消二者均为零，等待纪要时取消有一次结束／零纪要。每轮四次发送请求仅有三次发送操作／三条外发效果。初稿将整个响应外壳误判为应保持相同，失败记录保留在 v1；v2 是修正后的行为验证，v3 进一步明确区分六轮场景执行与实际 RPC 数量。本次只补验证入口，评分／任务数据仍为 v9／v15；不代表真实模型任务结果或项目完成。


文档版本、共享只读和附件事务权限回归：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.env.document_acl_probe --out artifacts/NEW_DOCUMENT_ACL_PROBE
```

探针由可信测试控制身份（不是模型可调用能力），验证 owner 发布、共享只读用户读旧／新版本但不能写、未授权用户搜索／读取不可见、多人附件任一接收人无权时整体拒绝，以及拒绝后邮件／外发／会议状态／操作收据均不部分提交。失败仍消耗一次工具调用和 30 秒，返回 data=null；除时间与调用计数外全库快照保持一致。发布 v3 后，v1／v2 整行保持不变，重放 v2 发布与 v2 附件发送仍复用原业务结果、更新观察 ID，不重复创建版本或外发。

artifacts/document_acl_probe_v2 的两轮共 38 次调用，每轮 8 次预期拒绝，状态和动作日志哈希一致。独立 SQL 审计核对旧版本内容／ACL／哈希、5 个提交操作、唯一一条发给授权用户的外发及明确的 v2 附件引用。该探针直接验证工具后端，不代替 MCP／模型隔离验证；未改变评分与数据版本，也不表示真实 teacher 或正式实验完成。


当前评分下的 D3 约束跳阶段审计可完整复跑：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.graders.dependency_probe --manifest artifacts/dataset_development_v25/manifest.json --out artifacts/NEW_D3_PROBES --max-instances 142 --max-wall-seconds 170
```

入口先验证当前数据，再要求实例上限能覆盖完整 D3 池，不能静默截取一部分后声称完整覆盖。按任务选择适用约束：确认、回复收集、材料绑定、会后待办，以及 U04 的负责人修正和通知。控制器明确标记伪造观察／修改参数；缺失待办时也提供连贯的后续伪观察，使参考程序能到达 task.finish，而不是因不存在的 ID 提前崩溃。回复夹具使用既定规则事实隔离证据缺失，不给真实模型新增私有入口。记录只有在截断了对应阶段、未超截止时间、已声明完成且命中预期违规时才算通过。

artifacts/dependency_skip_probe_v15 覆盖全部 142 个 D3 实例／368 条案例：确认142、待办142、回复32、材料绑定32、负责人修正10、通知10，全部命中目标拒绝，耗时8.20秒；独立 SQL 审计逐例核对了缺失／未修正的数据库事实。它更新了旧 v4 评分下48条证据，但只证明这些选定的证据／写入约束，不能据此宣称所有 DAG 节点、所有执行路径均必需，也不构成数据冻结或模型发现漏洞。


生成器现为 generator-v4-required-dependencies。发现旧图中 read_material 不是必需操作：任务／公开版本事件已给出材料引用，合法策略可以直接绑定版本。v16 删除 298 个此类必需节点，将 documents.read 标为可选观察；170 个路径长度和128 个宽度标注得到修正，160 个受控扫描实例的必需 DAG 宽度均为1。业务材料绑定仍受原约束检查。其他 DAG 路径仍需独立因果审计。

参考策略新增具名公共引用方案 reference-v3-public-material-refs-only-v1，通过 `solve(..., material_observations=False)` 实际省略读取；默认行为保留原观察。artifacts/material_read_optional_probe_v3 以真实工具调用完成全部298个材料实例，没有伪造观察、实际 documents.read 为0；相对基准省略的读取调用数为500，不是实测模型 token 或费用节省。早期v1/v2控制器省略观察的探索证据保留，最终证明采用真实公共工具方案及独立 SQL／动作日志审计。

旧 DAG 不可直接作为当前标注复用。显式迁移命令：

```bash
PYTHONPATH=src .venv/bin/python -m officegym.tasks.regrade --manifest /absolute/OLD/manifest.json --out /absolute/NEW --max-wall-seconds 180 --repair-optional-observations
```

该模式会修改外层 task.json 的 DAG／可选观察元数据，并如实报告 task_inputs_unchanged=false；公开任务、初始数据库、私有约束、事件规则、参考动作和最终业务状态保持一致。v16 的默认参考策略重放全部1,010例通过，142个D3实例的368条选定约束审计也通过；首批计划v7保持四十任务身份。当前数据仍未冻结，真实 teacher、人工审查、过拟合及正式实验仍待完成。


2026-11-01 optional handoff observation checkpoint: 346 CPU tests pass. Current unfrozen dataset is artifacts/dataset_development_v17, generator-v5-optional-observation-scope / required-dependency-annotation-v3; scoring remains eval-v9-invitation-timing/reward-v9-invitation-timing. In all ten current U01 fixtures, real people.search yields exactly one visible person outside the principal, initial coordinator and required participants. The named reference-v3-unique-public-other-approver-v1 policy requests real approval from this candidate and legally completes without reading the handoff thread. It refuses ambiguous rosters; the default reference still reads the chain and handles the added-distractor control. This is a narrow fixture shortcut, not a general handoff algorithm.

The explicit metadata migration removes ten optional read_handoff_chain nodes and corrects ten widths; no path lengths change. Independent audit verifies all 1,010 public task payloads, initial/private/event-rule/reference-action/grade files, semantic fingerprints and reference final hashes remain unchanged. All annotated DAG widths are now one; this is not exhaustive proof of causal paths. Optional tool metadata now includes purpose scopes (mail.get_thread: handoff_chain, documents.read: material_body when applicable), so it does not declare every use of the mail tool optional. Use --repair-optional-observations for explicit migration; --repair-optional-material remains a compatibility alias.

Evidence: handoff_read_optional_probe_v1 preserves the pre-repair proof; v2 verifies all ten current instances with zero handoff reads. dataset_v17_reference_replay passes all 1,010 grades/state hashes/operation audits; dependency_skip_probe_v17 passes all 368 selected-stage interventions over 142 D3 instances. teacher_pilot_selection_v8 preserves the same forty planned tasks as v7. optional_handoff_package_v1 verifies all 143 wheel source/resource files and isolated imports. No model/teacher/API calls or training took place. Human mask review, real sixteen-sample overfit, admitted SFT/export and formal GRPO/RS-SFT and frozen scientific evaluations remain pending.


2026-11-01 历史邀请的回复证据修复：评分器升级为 eval-v10-invitation-availability / reward-v10-invitation-availability。原评分器只保证最终版本存在晚于有效回复的邀请，遗漏了历史版本已经外发的违规事实。artifacts/grader_reply_evidence_v1 保留真实工具执行的三个反例：提前邀请后补回复、邀请窗口不符合回复后改正，以及取消早期错误邀请。v9 三例均误判成功；v10 逐条读取实际发送版本的会议快照，核对该版本所有参会人的有效回复是否在发送时已到达、且覆盖该版本时间，三例均以 invitation_before_valid_availability_reply 拒绝。后续修正或取消不能抹去历史违规。

新增两个合法案例分别先建草稿后收回复、先收回复后建草稿，均通过；不能将本次评分修复理解为强制 calendar.create 的调用顺序。两个额外测试检查只有一名参会人迟回复的情形，以及回复与发送时间相等时实际先到达后外发的合法事件边界。全部348项测试通过，导出案例集56例（20合法、36负例）零标签偏差。独立SQL审计及带版本的漏洞记录保存在 grader_reply_evidence_v1；这是一个人工构造根因，模型发现数为0，没有正式训练checkpoint需要重评。

当前未冻结候选 artifacts/dataset_development_v18 在新评分器下重评全部1,010例通过（9.91秒），全量新重放的成绩、终态哈希及操作证据均匹配。独立核对确认全部任务输入／DAG标注、初始世界、私有约束、规则、参考动作和参考终态文件与v17完全一致，参考成绩仅版本字段变化。dependency_skip_probe_v18 的全部142个D3实例／368条选定干预均命中拒绝；teacher_pilot_selection_v9 的40个任务身份不变。invitation_reply_package_v1 验证143个wheel文件及隔离导入。未进行模型、教师、优化器或付费调用；真实教师采集、人工mask审查、16样本过拟合、正式SFT／GRPO／RS-SFT、通过验收的导出及冻结科学实验仍未完成。


2026-11-01 reply provenance controls: artifacts/reply_provenance_probe_v1 runs three authored scenarios twice through real Dispatcher calls (six executions, 38 calls). Genuine participant availability replies pass. A principal-authored kind=reply message with a forged sender_id inside its body remains sender a and fails; a genuine participant reply containing the same availability facts but generated by a clarification request also fails. Allowed recipients include the principal in every scenario, so these negatives do not depend on wrong-recipient rejection. No reply rows, sender fields or event logs are edited during execution. Each pair has identical final-state and action-log hashes. Independent SQL audit checks actual senders, request/thread provenance, ReplyArrived events and the still-pending genuine reply in the spoof case. This verifies existing controls; it is not a newly discovered grader bug, exhaustive provenance proof or model reward-hacking finding.

Reproduce with `PYTHONPATH=src .venv/bin/python -m officegym.env.reply_evidence_probe --out artifacts/NEW_REPLY_PROVENANCE_PROBE`. All 349 CPU tests pass. reply_provenance_package_v1 verifies all 144 wheel source/resource files and isolated imports. The current unfrozen v18 dataset, v10 scoring and v5 generator are unchanged; no model, teacher, optimizer or paid calls occurred. Human mask review, real sixteen-sample overfit, admitted SFT/export and formal GRPO/RS-SFT/frozen scientific evaluations remain pending.


2026-11-01 GradeResult completion contract: added grade-result-v2-completion-evidence with terminated_reason, termination_reason_source and completion_checks. Snapshot-only grading reports DECLARED_DONE / NOT_DECLARED_DONE / DEADLINE_EXCEEDED from database facts; it does not infer wall-clock, parsing or infrastructure termination. Checks separately expose business outcome, absence of hard/soft violations, absence of unexpected mutations, simulated deadline and declared completion. Existing business-success and reward semantics remain unchanged: a valid invited meeting can satisfy the domain before task.finish, while rollout/teacher admission still applies its existing termination gates.

Actual Harness and framework-independent rollout finalization now record their trusted runtime reason and runtime_accepted check, retaining world_terminated_reason. A budget stop can coexist with a finished business world and a valid simulated deadline; it still fails runtime admission. artifacts/grade_completion_runtime_v1 preserves two real Harness runs and all five rollout stop reasons with actual tool-executed reference worlds; independent SQL audit checks database finished flags and tool counts. Authored policies/tokens are explicitly software fixtures, with zero model calls; these are not native model-training experiments.

All351 CPU tests pass, and grade_completion_grader_report_v1 retains all56 labelled outcomes. Unfrozen artifacts/dataset_development_v19 regrades and freshly replays all1,010 instances; independent_schema_audit.json verifies every old GradeResult field, scoring identity, task/DAG input, reference action and final-world file remains unchanged, with exactly four diagnostic fields added. Pilot v10 preserves the same40 planned tasks. grade_completion_package_v2 verifies all145 wheel files and isolated imports; the earlier package v1 predates final retention of world_terminated_reason and remains historical. Current scoring stays eval-v10-invitation-availability/reward-v10-invitation-availability; the diagnostic schema identity is separate. Formal teacher collection, human mask review, real16-sample overfit, admitted SFT/export, GRPO/RS-SFT and frozen scientific results remain pending.


2026-11-01 evaluation export evidence guard: actual episode execution and offline export now share grade_episode_snapshot. result_row re-evaluates the retained world against the source public/private constraints and initial snapshot, applies recorded Harness termination/error policy, independently reconciles operations, and requires exact agreement with the recorded grade and audit before constructing a row. Export rows retain grade_schema_version, terminated_reason, termination_reason_source and completion_checks. A success flag, completion check or operation count edited in result.json cannot silently override the underlying evidence. This is consistency checking against the supplied terminal state and world, not authentication against coordinated rewriting of every artifact. Existing model/isolation/context-measurement gates remain in production.

artifacts/grade_completion_runtime_v2 preserves seven fresh actual Harness/framework-independent rollout paths under the shared implementation, using explicitly authored policies/tokens and zero model calls. export_grade_evidence_v1 verifies two recorded Harness runs, rejects all six offline evidence mutations without changing originals, and checks five terminal states against a genuinely tool-completed reference world. The export-entrypoint unit test explicitly mocks only independent model/isolation/context gates to exercise the real business-evidence guard; it cannot be used as a measured model result. Current scoring v10, grade schema v2, generator v5 and unfrozen v19 data are unchanged. All354 CPU tests pass; export_grade_package_v2 verifies all146 wheel source/resource files and isolated imports. Earlier package v1 predates final diagnostic row fields and is historical. No teacher, model, optimizer or paid activity occurred. Formal teacher data, human mask review, sixteen-sample overfit, admitted SFT/export, GRPO/RS-SFT and frozen scientific evaluation remain unfinished.


2026-11-01 reproducibility inventory completion: scripts/export_versions.py now records repository configurations (JSON/YAML/TOML), dependency lock files, Python/shell entrypoints and pyproject.toml in addition to implementation Python, SQL and contracts. inventory_groups and reproducibility-source-inventory-v2 make coverage explicit. Installed package versions remain separately observed. Configuration templates are hashed as repository inputs, not admitted as resolved native/runtime configurations. Model weights, run artifacts and environment values are not included in this source inventory.

artifacts/version_inventory_v2/independent_audit.json verifies all169 listed files:140 implementation files,2 SQL schemas,5 contracts,6 configurations,7 dependency locks,8 entrypoints and1 packaging descriptor. New tests show configuration/lock/launcher changes invalidate the source-manifest hash and artifact directories are excluded. All356 CPU tests pass. This turn changes the repository metadata script, not the installed business runtime, scoring or unfrozen v19 dataset. No model, teacher, GPU or paid activity occurs. Required teacher identity and operator API/GPU budgets in configs/project.json remain unfilled; genuine teacher collection, human mask review,16-sample overfit, admitted SFT/export, full native training smoke tests and frozen scientific evaluations remain incomplete.


2026-11-01 long output paths and native parse parity audit: a real fixture run failed to start its world service because AF_UNIX pathname length exceeded Linux limits; artifacts/harness_parse_parity_probe_v1 retains the infrastructure failure. run_episode now allocates a private short /tmp socket directory when the encoded requested socket path approaches the limit, keeps all evidence at the requested output path and cleans up the temporary socket after owned service shutdown. A regression test uses a long UTF-8 directory and verifies normal DECLARED_DONE plus retained world/runtime/result files.

The original protocol audit now has real evidence: native_parse_parity_probe_v1 executes the pinned upstream AgentLoop with the actual tokenizer and authored tokens (zero model generation). Both unparseable plain text and malformed tool JSON terminate after one native generation with no tools. harness_parse_parity_probe_v2 runs the actual ChatPolicy parser and both ordinary Harnesses with authored transport responses: each makes three requests, bills all fixture-reported generated tokens and ends PARSE_ERROR without tools. Independent SQL/actor-token replay in parity_audit.json confirms protocol_consistent=false. Matching bounded retry/history semantics in the native loop remains required; neither the native probe's passed plumbing flag nor this audit is a formal training acceptance. Native generated IDs, masks and logprobs must remain intact when that fix is implemented.

All357 CPU tests pass; long_path_package_v1 verifies146 wheel files and isolated imports. Scoring v10, grade schema v2, generator v5 and unfrozen v19 data remain unchanged. No model, teacher, optimizer or paid call occurred. Genuine teacher data, human mask review, real16-sample overfit, admitted SFT/export, complete native training smoke tests and frozen scientific experiments remain incomplete.


2026-11-01 bounded parsing retries: native no-call/malformed/multi-call/nonobject responses now use the ordinary Harness default of two bounded retries, with the same user feedback text and no tool execution for invalid responses. Validly parsed calls reset the failure streak, including calls whose tool response is a structured error. Exhaustion produces native parse_error with failed runtime admission. Launch-wide rollout row binding explicitly supplies parse_retries (default2), rejects invalid values and row overrides, and supports explicit finite nonnegative alternatives. Native retry feedback is appended through the original Continuous Token context path with actor IDs/logprobs preserved and feedback mask/logprob zero; response/context limits are checked before committing oversized feedback.

Ordinary Harness retries now retain the invalid model response as inert assistant text before feedback, preserving content and serializing an unvalidated batch as text rather than executable tool history. Raw responses remain in the trajectory; no usage/transport metadata is injected as model content. Teacher SFT extraction continues its existing rejection of trajectories containing parse failures. HARNESS_VERSION advances to harness-v10-bounded-parse-retry-history, so old prepared/evaluation artifacts require current-Harness validation; v19 task inputs, reference actions, scoring v10 and grade schema v2 do not change. Pilot v11 preserves the same40 planned tasks, without teacher sampling.

Evidence: native_parse_retry_probe_v4 runs21 actual pinned AgentLoop/tokenizer cases using authored token outputs, including persistent failures, recovery, streak reset, nonobject arguments, zero retries and feedback response-capacity overflow. Its independent audit replays every completed actor-token/logprob stream, zero-mask metadata and generation-token ledger, checks no tools were dispatched for invalid-only trajectories, and verifies owned-process cleanup. Four comparisons against harness_parse_parity_probe_v3 confirm three generations/zero tools/normalized PARSE_ERROR in both ordinary modes and native parsing exhaustion. Native protocol probes use real tokenizer IDs with authored outputs; they are not model generations or formal training. v1 failed when an authored backend-version list ran out after added retries and remains retained; v2/v3 are intermediate successful probes, v4 is current.

All359 CPU tests pass; native_parse_retry_package_v2 verifies146 wheel files and isolated imports (v1 predates final probe coverage). No model, teacher, optimizer or paid calls occurred. Real teacher data, human mask review,16-sample overfit, admitted SFT/export, full native optimizer/version-sync smoke tests and frozen scientific evaluation remain incomplete.


2026-11-01 native parse capacity acceptance: native_parse_capacity_probe_v2 extends the actual pinned upstream/tokenizer authored-output probe to26 cases. New controls stop after one generation when parse feedback exceeds context capacity, assistant/user turn caps are reached, or the finite generation allowance is consumed. The three-token case reserves and observes exactly3 tokens; invalid-only boundary cases perform zero tool calls/operations and retain failure rewards. An unknown-tool structured error is billed as an attempted call and resets the parse-failure streak, allowing the next two invalid outputs to be retried before a valid finish. No partial multi-call execution or silent actor-token truncation is introduced.

Independent raw ledger/SQL checks are in independent_boundary_audit.json; independent_audit.json replays all completed generated actor IDs/logprobs/masks, reconciles generation usage, compares the four ordinary-Harness parse cases and checks owned-process cleanup. v1 preserves the25-case intermediate run; v2 includes the structured-tool-error recovery control. native_parse_capacity_package_v1 verifies all146 wheel files and isolated imports. The last full CPU suite remains359 passing tests; this turn changes only the native diagnostic probe and directly executes its full26-case path. Harness v10, scoring v10, grade schema v2, generator v5 and unfrozen v19 data remain unchanged. Model generations, optimizer updates, teacher samples and paid calls are zero. Full native training/version-sync smoke tests, admitted SFT/export, real teacher/human review/overfit and frozen scientific evaluations remain incomplete.


2026-11-01 historical-calendar validation follow-up: all362 CPU tests pass, including send-sequence creation/cancellation, later rescheduling in both directions, initial before-images, seeded objects without mutation events, creation timestamps, organizer conflicts, unrelated participants and touching nonoverlapping intervals. dataset_v20_reference_replay passes all1010 grades/final-state hashes/operation audits. dataset_development_v20/independent_history_audit.json independently verifies original task input hashes, unchanged reference final hashes and grades unchanged except scoring versions. calendar_history_package_v1 verifies147 wheel source/resource files. This closes the full reference replay and local history-boundary follow-up; a larger exported labelled conflict suite and formal training/scientific evaluation remain pending. No model/teacher/optimizer/paid activity occurred.


2026-11-01 exported calendar-conflict cases: generate_cases now includes three genuine tool-executed, approval-backed historical calendar controls. A conflicting first invitation followed by an approved correction is rejected; a touching nonoverlapping slot and an explicitly authorized cancellation before invitation are accepted. All59 labelled cases (22 legal/37 negative) match in calendar_conflict_report_v1, with original traces, initial/final databases and specs retained in calendar_conflict_cases_v1. All362 CPU tests pass; conflict_suite_package_v1 verifies148 wheel files. This expands known-case verifier coverage; it is not model reward-hacking discovery, exhaustive conflict proof or formal training/evaluation completion. No model, teacher, optimizer or paid activity occurred.


2026-11-01 v11 preparation revalidation: dependency_skip_probe_v20 repeats all368 selected-stage interventions across142 D3 instances against dataset_development_v20 and current scoring; every intervention is rejected with its target constraint. Manifest SHA is independently checked. teacher_pilot_selection_v12 preserves exactly the same40 planned task identities as v11, with no model calls. historical_conflict_probe_v1/audit.jsonl now binds the retained genuine tool trace, minimal reproduction and old v10/new v11 grades; audit_summary.json counts one authored root cause, zero model discoveries and zero pending checkpoint reevaluations (no admitted trained checkpoints). README current commands now use v20. These are development preparation/audit records, not frozen data, teacher samples, training or scientific results. Formal gates remain incomplete.


2026-11-01 documented generation CLI contract: `python -m officegym.cli generate --config configs/project.json --out NEW_DIRECTORY` now routes to the same full configured development assembler as assemble-dataset. Without --config, existing individual-template generation is unchanged. Combining --config with template/split/seed/depth/perturbation options is rejected, including --option=value forms, rather than silently ignoring overrides. CLI route tests mock only the existing assembler; they establish argument routing and rejection, not a new end-to-end dataset generation or freeze. Formal teacher/model/budget inputs remain pending; a clarification request asks for exact teacher service/version and API/GPU caps without requesting credentials in chat. No model jobs or paid calls occur.


2026-11-01 full documented config-CLI execution: the actual `generate --config configs/project.json --out artifacts/config_cli_dataset_v1` command completed under a180-second external cap, producing1010 verified reference instances with splits800/80/60/30/40. config_cli_validation_v1/audit.py independently runs validate_dataset, checks configured counts, compares identities/fingerprints/reference final hashes and all6060 manifest-listed input/action/grade file hashes against v20. Every comparison passes; the command log, config/manifest hashes and audit result are retained. This upgrades the earlier mocked route test to genuine end-to-end generation evidence. Both candidates remain unfrozen; no teacher/model/optimizer/paid calls occur and formal data/training/scientific acceptance remains pending.


2026-11-01 version-evidence preservation: scripts/export_versions.py now exclusively creates its destination and refuses to overwrite an existing version manifest. version_preservation_probe_v1 retains the original manifest, a rejected second invocation, stderr and matching before/after SHA256 values; the CLI regression test also verifies unchanged bytes after failure. Choose a new --out path for each snapshot. No business runtime, dataset, scoring or model-job behavior changes. Formal teacher/training/scientific acceptance remains incomplete.


2026-11-01 generation-argument abbreviation fix: argparse previously accepted --te=T01 as --templates, bypassing the config/template mixed-option guard and silently invoking the full assembler. The generate subparser now requires complete option names (allow_abbrev=False). Targeted CLI tests cover exact, equals-form and abbreviated template/depth/holdout options; the actual subprocess probe in generate_abbreviation_probe_v1 confirms exit2 before an output directory is created. Documented complete commands retain their existing behavior. No dataset or model job was started by this probe; formal experiment gates remain incomplete.


2026-11-01 run-plan CLI delivery: `python -m officegym.cli freeze-run-plan --manifest DATASET/manifest.json --out NEW_PLAN.jsonl` validates the dataset, builds P0/P1 run keys with chosen --splits and writes an exclusive plan plus SHA256 through existing plan functions. It does not implement dataset freeze/admission. Actual v20 development execution in run_plan_cli_probe_v1 produces1950 unique P0 keys (ID900/combo450/unseen600), repeats0/1/2; independent hash/count checks pass, and a repeated write is rejected without changing the plan. Source manifest hash and explicit dataset_frozen=false are retained in audit.json. This is preparation evidence, not a formally admitted experiment plan or retrospective proof of experiment chronology. Dataset-freeze acceptance, teacher/model budgets, formal training and scientific results remain incomplete. No model/teacher/optimizer/paid calls occur.


2026-11-01 plan-pair failure cleanup: freeze_plan now flushes/fsyncs both exclusively created files and rolls back invocation-owned files on an exception. Cleanup checks device/inode before unlinking, preserving existing checksum/plan files and paths replaced by another writer. Five targeted plan tests pass, including a pre-existing checksum, pre-existing plan and injected second fsync failure. This is ordinary exception cleanup, not an atomic two-file commit or a guarantee against abrupt process/power loss; readers still require the matching checksum. No model, dataset or training work occurs. Formal completion gates remain pending.


2026-11-01 integrated software acceptance checkpoint: all368 CPU/protocol tests pass after recent CLI, source-inventory and run-plan rollback changes. integrated_regression_v1 verifies all148 wheel files against repository bytes, extracts the wheel and runs isolated Python outside the repository. The installed freeze-run-plan entrypoint generates1950 valid keys against v20, rejects an overwrite without changing plan bytes, and the installed v11 grader rejects the preserved historical-conflict world with reward-1. Raw command output/rejection, package SHA and report are retained; this is actual packaged-runtime verification without model inference. Dataset remains unfrozen and all genuine teacher/human-review/overfit/SFT/export/GRPO/RS-SFT/scientific result gates remain incomplete.


2026-11-01 historical-material regression acceptance: dataset_v21_reference_replay passes all1010 fresh reference grades/final hashes/operation audits. dataset_development_v21/independent_material_audit.json checks unchanged v20 task inputs and reference final hashes, with reference grades differing only in scoring versions. The full369-test regression run passes; the subsequently expanded material-history test file also passes both tests, covering document creation sequence, static requirements, seeded versions and later publications sharing a timestamp. material_history_package_v1 verifies149 wheel files. This remains development software evidence, not a freeze or scientific experiment; exported material-case expansion, real teacher/human review/overfit, admitted training/export and formal results remain pending. No model, teacher, optimizer or paid calls occur.


2026-11-01 exported historical-material controls: generate_cases now adds real tool-executed, approval-backed missing-material/corrected-version and material-bound-first-invitation controls. material_cases_v1 retains all61 labelled worlds/specs/traces; material_report_v1 matches23 legal and38 negative labels with zero mismatches. historical_material_probe_v1/audit.jsonl binds v11/v12 grades and original trace/reproduction, counting one authored root cause, zero model discoveries and no affected admitted checkpoints. All370 CPU tests pass; material_case_package_v1 verifies150 wheel files. These are software verifier checks, not teacher data or model experiments. Formal freeze, human review, overfit, admitted SFT/export, native training and scientific evaluations remain unfinished.


2026-11-01 v12 development preparation refreshed: dependency_skip_probe_v21 covers all142 D3 instances/368 selected-stage interventions with target rejections under current scoring; independent_manifest_audit.json checks the actual source SHA. teacher_pilot_selection_v13 preserves the same40 planned task identities as v12. v12_run_plan_v1 regenerates1950 P0 keys with the actual CLI, verifies byte equality with the earlier key plan and binds the v21 manifest hash while explicitly recording dataset_frozen=false. README current commands use v21. No model/teacher sampling, optimizer or paid work occurs; these preparation artifacts do not establish data freeze, admitted training or scientific outcomes.


2026-11-01 executable operator-input check: `python -m officegym.cli check-operator-inputs --config configs/project.json` reports missing/invalid teacher choice and finite positive total API/GPU caps without reading credential environment values or launching jobs. It exits2 when choices are incomplete; success only validates those three fields, not provider/pricing credentials, dataset/model admission or scientific completion. Three targeted tests cover explicit choices, placeholders, null/boolean/nonfinite/nonpositive budgets. operator_inputs_probe_v1 verifies the real current config exits2 with model.teacher/operator_budget.max_api_cost_usd/operator_budget.max_gpu_hours missing. Existing model-job admission checks remain separate; no inputs are guessed and no paid/model calls occur.


2026-11-01 full matrix key/denominator preparation: full_matrix_plan_audit_v1/audit.py validates v21 and generates four exclusive SHA-bound plans. P0 has1200 dev/1950 test keys (5 groups); P1 has1920 dev/3120 test keys (8 groups). Every group/instance has exactly repeats0/1/2, all keys are unique, and missing_runs reports the full denominator for no records, zero for complete identity rows and exactly one after removing one row. These checks exercise identity accounting only; passing plan rows as identity fixtures does not create measured results or metric records. report.json binds the source manifest and explicitly records dataset_frozen=false/model_calls=0. Runtime/model/budget admission, dev calibration, dataset freeze and formal scientific evaluation remain required.


2026-11-01 v12 integrated packaged-runtime acceptance: all373 CPU/protocol tests pass. v12_integrated_package_v1 verifies151 wheel source/resource files and loads only the extracted wheel through isolated Python outside the checkout. Installed CLI generation produces1950 run keys, refuses overwrite without modifying the plan, installed grading rejects both historical calendar-conflict and material-omission worlds, and installed check-operator-inputs reports the three current missing choices with exit2/jobs_started=false. Raw outputs, rejection and wheel SHA are retained. This validates current software packaging/integration; it does not admit development v21 as frozen data, provide real teacher samples, satisfy human/overfit/training/export gates or establish scientific results. No model, teacher, optimizer or paid calls occur.


2026-11-01 development calibration audit implementation: eval.calibration.audit_calibration requires a SHA-bound complete dev plan for api_simple/api_full, every dev instance and repeats0/1/2, matching non-synthetic result rows, one shared model/budget identity and current Harness. Every row is re-exported from original runtime artifacts through existing isolation/context/world-grading checks and compared before rates are computed. Difficulty bounds are explicit finite caller inputs; output distinguishes evidence_verified/calibration_usable and always dataset_frozen=false. A targeted test mocks only independent dataset/export gates to test six-row denominator accounting, missing results and synthetic rejection; it does not admit fixture rows as measured data. CLI wiring and genuine calibration remain follow-up; no actual calibration result, freeze or model/API/optimizer work is claimed.


2026-11-01 dev calibration CLI: `audit-dev-calibration --manifest ... --plan ... --results ... --out NEW_REPORT.json --min-success-rate ... --max-success-rate ...` exposes the evidence audit with explicit difficulty bounds. A verified but unusable difficulty report is retained and exits2; malformed or incomplete evidence fails without claiming calibration. Tests cover CLI routing, missing records, synthetic rows, mismatched model/budget identity, unusable rates and invalid/nonfinite bounds before file access. calibration_cli_probe_v1 uses the actual v21 manifest and480-key paired dev plan; empty results are rejected and no calibration report is created. Genuine calibration still requires real API simple/full results and configured model/budgets; no model or paid calls occur.


2026-11-01 calibration-repeat evidence reuse fix: audit_calibration now requires a nonempty artifact path for each planned run and rejects reuse of canonical resolved runtime directories. Without this check, the same original episode could be relabelled with multiple repeat IDs because export preserves the supplied planned repeat. Targeted calibration tests now use separate authored fixture paths for the accounting control and reject exact reuse, relative-path aliases and symlink aliases before re-export. These mocked accounting tests do not admit fixtures as model results. This guards directory reuse, not coordinated copying/forgery of every runtime file; original provenance checks remain required. No genuine calibration, freeze, model or paid job occurs.


2026-11-01 statistics runtime-evidence reuse protection: select_and_validate now rejects repeated canonical artifact directories whenever selected results carry artifact_path. Evidence paths must then be present/nonempty for every selected row, preventing one run from being relabelled with different repeat/group keys or relative-path aliases before bootstrap/summary. The21 targeted support/statistics tests pass, including valid distinct paths, normalized alias reuse and partial path coverage. Legacy path-free statistical inputs retain their original contract and do not acquire model provenance through this check; copied/forged runtime trees remain outside this directory-identity guard. No measured result or model experiment is created. Formal calibration/freeze/training/scientific gates remain incomplete.


2026-11-01 calibration input stability: audit_calibration captures manifest/plan/result-ledger SHA256 values before validation, checks them again after runtime evidence re-export and writes those original hashes into the report. Persisting an input change during audit now fails before creating a report, avoiding rates computed from earlier bytes being labelled with a later ledger hash. A targeted mocked-export fault injects result-ledger changes during per-row verification and verifies failure/no output; all3 calibration tests pass. This detects changed input files, not coordinated transient tampering or an atomic snapshot of every runtime artifact. No real calibration, data freeze, model or paid job is claimed.


2026-11-01 v22 clean regression: all377 CPU tests pass after correcting the obsolete fixed-width assertion. dataset_v22_reference_replay and independent_replay_audit.json verify all1010 fresh grades/final hashes/operation audits against retained references. v22_package_v1 verifies152 wheel files. v22_width_audit_v1 publishes actual per-split/depth/perturbation/width CSV:160 controlled instances have32 D1P1 width1 and32 each D2P0/D2P1/D2P2/D3P1 width2. It explicitly records fixed_width_depth_scan_valid=false/freeze_ready=false; matched noise/timing does not remove depth/width confounding. Scoring stays v12 and Harness v10; generator/annotation are v6/v4. Experimental redesign or width-stratified interpretation is still required before freeze, alongside real teacher/training/scientific gates. No model/teacher/optimizer/paid work occurs.


2026-11-01 proposal-bound controlled generator activated: generator-v7-proposal-bound-controls / required-dependency-annotation-v5 enables explicit proposal_bound_availability and approval_after_availability on controlled D2/D3 tasks. Their DAG is plan→collect_replies→confirm→invite followed by existing downstream stages; D1 retains its original chain. Unscoped atomic reply tasks retain truthful width2. Reference identity advances to reference-v4-proposal-bound-availability. This changes public task business requirements and reference trajectories; it is a fresh generation, not metadata-only migration.

Actual generate --config execution creates unfrozen dataset_development_v24 with1010 passing reference instances and zero exact duplicates. controlled_design_audit.json confirms160 controlled instances all width1; whole-dataset widths870×1/140×2. All381 CPU tests pass. The audit also records60 train reference trajectories above the24-call training cap, maximum36 calls; this is not proof of feasible teacher/model training under that cap. Fresh full replay, new causal skip-stage tests, budget-bounded legal strategy/token-length audits and experiment preparation refresh remain necessary. Old v22/v23 mixed-width data is retained; the new public semantics cannot be obtained by merely repairing its metadata. No model, teacher, optimizer or paid work occurs; no data freeze or scientific acceptance is claimed.


2026-11-01 v24 reference and dependency regression: dataset_v24_reference_replay freshly verifies all1010 grades, final hashes and operation audits; independent_replay_audit.json compares each saved result. dependency_skip_probe_v24 covers all142 D3 instances/368 selected-stage interventions, all rejected with target constraints, with source manifest SHA independently verified. Existing controller-authored omission probes still reach declared completion under the new recollection path; no actor observation or model result is fabricated as scientific evidence. These checks do not prove every DAG edge or budget24 feasibility. New task public semantics remain a development candidate pending bounded legal strategies, current tokenizer lengths and broader design audits. No model/teacher/optimizer/paid work occurs.


2026-11-01 bounded public change coalescing: the opt-in wakeup heuristic can wait through the public monitor horizon, read real queued events, merge time/material changes into one actual calendar.update, recollect version-bound availability and obtain a genuine approval before one reschedule. Public event data alone determines the merged patch; raw event observations are not replaced. Attendee changes explicitly remain unsupported by this narrow heuristic. Default reference behavior is unchanged.

public_wakeup_budget_probe_v3 independently runs all128 v24 proposal-bound instances with Dispatcher initialized under actual cap24. All128 finish normally with declared_done/full_success and no tool errors; maximum22 calls. independent_audit.json checks database cap/counters, all real trace responses, completion and grades, fixing the earlier domain-only success criterion. The previous v1 misconfigured-budget and v2 partial96/128 attempts remain retained. The targeted reference test now passes all eight default/bounded controlled fixture branches. This proves call-budget existence for these scoped fixtures only, not model/teacher success, full-training-token feasibility or every atomic template. Current reference trajectories remain preserved and formal data/training/scientific gates remain incomplete; no paid/model/optimizer activity occurs.


2026-11-01 real tokenizer audit of bounded public strategies: public_wakeup_token_audit_v1 audits all128 scoped v24 traces using the actual pinned Qwen tokenizer, full tool schemas, current full-Harness instructions and complete uncompressed tool histories. It retains original encoded IDs, assistant masks/roles, trace/ID hashes and per-case lengths; independent replay verifies alignment and non-actor masking. All full histories fit8192 (maximum7350), but only96 fit when every generation reserves1024 output tokens;32 fail that stricter admission test, maximum required8289. These are authored tool-only policy histories without model reasoning, not teacher data or actual generation-token measurements. Call-budget existence therefore does not yet establish all current context-budget feasibility. Model/teacher/optimizer/paid counts remain zero; further legitimate strategy/context work and formal experimental gates remain incomplete.


2026-11-01 followup polling and context reserve: opt-in public wakeup strategy waits before the first followup poll when the public meeting end is still in the future, avoiding one known premature poll. It uses concise structured availability requests without redundant body text; request/meeting/version/recipient/approval constraints remain unchanged. Default reference behavior is unchanged. public_wakeup_budget_probe_v5 completes all128 scoped v24 instances under actual Dispatcher cap24, maximum21 calls.

public_wakeup_token_audit_v3 re-encodes every complete saved trace with the actual pinned tokenizer/schema/full prompt, retains IDs/masks/roles/hashes and verifies assistant spans. All128 fit8192 including1024 next-output reserve; max history7243, max input+reserve8182 (minimum margin10). Independent audit links traces, real successful world grades/caps and token IDs. Prior v2 token audit remains retained with125/128 fitting and max8196; earlier failures are not relabelled. These are authored tool-only traces: actual model reasoning and independently assigned Harness/teacher operation IDs may change context, so this is not real teacher feasibility/admission. The381-test full regression preceding the final body shortening passed, and targeted default/bounded reference tests passed after it. Formal teacher/data/training/scientific gates remain incomplete; no model/teacher/optimizer/paid calls occur.


2026-11-01 actual full-Harness context counterevidence: wakeup_harness_context_probe_v1 executes32 v24 T04 D3P1 instances with real world-service processes, full journal UUID operation IDs, actual tokenizer and TrainingContext8192/reserve1024/call cap24. All32 stop CONTEXT_OVERFLOW before normal completion; none is counted successful. Independent audit reconstructs exact uncompressed interaction histories, verifies32-character journal keys and recalculates required next-input context, maximum8386. The prior authored reference-key trace result8182 remains valid only for its own serialization and does not establish actual Harness feasibility. All raw runtime/world/context artifacts are retained; software fixture token usage and replay policy are not model/teacher evidence. An efficient legal strategy or public-query observation design must address the additional real provenance tokens without truncating history or raising the fixed budget. No model/teacher/optimizer/paid work occurs; formal acceptance remains incomplete.


2026-11-01 public event-kind filtering: tools-v3-filtered-public-events keeps24 tools and adds optional kinds:string[] to events.poll. Filtering occurs after ACL visibility. A full matching page advances to its last returned match; an exhausted matching set advances past inspected visible rows, so irrelevant visible events do not force repeated pages. Hidden events neither appear nor advance the public cursor. Empty/duplicate/invalid kind lists are rejected. Omitted filtering preserves prior behavior. Schemas are shared across native/MCP/HTTP clients, and the tool description documents cursor semantics.

Actual Dispatcher tests cover matching pagination, irrelevant visible events, hidden matching events, invalid lists and unchanged unfiltered results. All382 CPU tests pass. Fresh config generation creates unfrozen dataset_development_v25 with1010 passing references/validation and zero exact duplicates under tool protocol v3. This implements the query primitive, not yet the reference-policy integration or proof of real Harness8192/1024 feasibility. The prior32 real Harness context failures remain authoritative until fresh successful executions. No model/teacher/optimizer/paid activity occurs.


2026-11-01 filtered public wakeup strategy and real Harness acceptance: opt-in reference polling selects visible reschedule/attendee/document/minutes/notification events through the public v3 kinds parameter; genuine returned events retain all provenance fields. No stored interaction history or operation ID is shortened or rewritten. public_wakeup_budget_probe_v6 completes128 scoped v25 instances under cap24, maximum21 calls.

wakeup_harness_context_probe_v2 first reruns the32 previously failing D3 cases successfully; v3 expands to all128 scoped instances through real full Harness, world-service processes, journal UUID keys and actual tokenizer/TrainingContext. All128 reach DECLARED_DONE/full_success with2,016 real tool calls total. Independent audit verifies every32-character write key, cap24, zero compression/removal, actual event-kind filters, per-call input+1024≤8192 and complete histories≤8192. Max input+reserve6984; max history6047. All382 CPU tests pass. Previous context-overflow artifacts and narrower reference-key token audits remain retained. Policies and generation-token usage are authored fixtures, not model/teacher sampling; natural-language reasoning can still change context. Formal calibration/freeze, human review/overfit, admitted training/export and scientific evaluation remain incomplete. No model/teacher/optimizer/paid calls occur.


2026-11-01 native filtered-event protocol checkpoint: native_event_filter_probe_v1 passes all27 cases using the actual pinned upstream AgentLoop and Qwen tokenizer with authored output tokens. The added events.poll case forwards after_seq=0 and kinds=[Notification] unchanged, receives a successful tools-v3-filtered-public-events response, then executes task.finish. Independent audit verifies two charged tool calls,42 original actor tokens/logprobs, zero non-actor logprobs, exact terminal masks/IDs, ledger observations and owned-process cleanup. This fixture returns no Notification items; nonempty filtering/pagination/ACL semantics are covered separately by the passing actual Dispatcher event-kind test. Current source hashes match the report. Prior26 retry/cap/version/isolation cases remain passing. No model generation, teacher sampling or optimizer update occurs; formal training and scientific acceptance remain incomplete.


2026-11-01 full current-protocol replay and controlled-design admission: reference_replay_v25 re-executes all1010 current v25 instances under tools-v3. All1010 reference grades, deterministic final hashes and operation audits pass. Independent audit checks every saved trace and grade,16,310 real tool calls, successful task.finish and current tool metadata, bound to manifest SHA256. These are default reference software executions, not budget24/model feasibility or teacher evidence.

Controlled scan validation now explicitly requires depth-dependent collect_availability/proposal_bound_availability/approval_after_availability booleans, required approval, and matching collection/confirmation stages. A task cannot claim the current controlled design after weakening those business conditions, even when public task/database copies and artifact hashes are updated together. New negative cases cover disabled collection, numeric rather than boolean binding, disabled approval ordering and disabled approval. All383 CPU tests pass; current1010-instance v25 validation still passes with zero exact duplicates. Formal dev calibration, dataset freeze, human mask review/real overfit, admitted training/export and scientific model evaluation remain incomplete. No paid calls, model generations or optimizer updates occur.


2026-11-01 dataset freeze admission implementation: `officegym freeze --dataset MANIFEST --grader eval-v13-proposal-bound-availability --calibration-plan DEV_PLAN --calibration-results DEV_RESULTS --min-success-rate 0.1 --max-success-rate 0.9 --out FREEZE_RECEIPT` re-runs the existing original-evidence calibration audit; it does not accept a caller-authored audit report. Bounds must be explicitly chosen for the experiment. Admission requires complete paired real API dev results and usable calibrated difficulty. The content-bound receipt/checksum binds all manifest-listed dataset files, calibration plan/results and runtime artifact files, plus packaged Python/JSON/SQL implementation hashes and grader/reward/Harness/tool identities. Inputs are compared before/after audit and existing receipt/checksum files are never overwritten. This detects persistent changes; it is not an atomic snapshot or independent chronology proof. Source files remain in place.

`officegym verify-dataset-freeze --freeze FREEZE_RECEIPT` verifies the receipt and all bound current files/code, then revalidates the dataset. `officegym evaluate ... --dataset-freeze FREEZE_RECEIPT` checks admission before model calls and records the receipt hash in batch selection/resume identity. Existing development evaluation without this option remains available and does not claim frozen evaluation. Checkpoint-selection rule freezing remains a separate existing workflow.

Actual `dataset_freeze_cli_probe_v1` rejects empty calibration results with exit1 and creates no receipt/checksum; v25 remains unfrozen. Lifecycle unit fixtures mock calibration solely to test receipt persistence, mutation detection, no-overwrite and pre-execution rejection. No genuine successful calibration/admission, model call, teacher sample or optimizer update is claimed. Formal training/export and scientific evaluation remain incomplete.

Final validation for this checkpoint: all386 CPU tests pass (20.584s); actual empty-calibration CLI rejection creates no freeze receipt.


2026-11-01 frozen evidence through exports and comparisons: batch selection now stores the canonical freeze receipt path as well as its hash. Exporting an episode from an admitted batch revalidates that receipt against the current manifest, bound files and code, checks batch identity, and carries dataset_freeze_path/dataset_freeze_sha256 into the result row. Missing partial bindings, changed receipts and dataset mismatches fail closed. Optional fields are documented identically in source and packaged result schemas.

`scripts/analyze_results.py ... --dataset-freeze FREEZE_RECEIPT` adds strict frozen-evidence admission to ordinary complete paired statistics. It verifies the current receipt and run-plan checksum, requires matching per-row receipt/plan bindings and distinct original runtime directories, checks the originating batch plan, and re-exports each selected original episode through model/isolation/context/world/grader validation before writing a report. Altered measured rows are rejected. The report records receipt/manifest/plan hashes and verified runtime-row count. Without this option the existing development/statistical input mode remains available and does not gain frozen-evidence admission. Content hashes do not independently prove chronology.

All388 CPU tests pass (20.595s); contracts validate, source/packaged schemas match and the real statistics CLI advertises the new option. New lifecycle tests mock independent evidence gates to test binding/re-export rejection only; no real frozen model result or successful scientific comparison is claimed. v25 remains unfrozen and formal teacher data/calibration/training/export/evaluation gates remain incomplete. No model/teacher/optimizer/paid activity occurs.


2026-11-01 bounded matrix execution entrypoint: `officegym evaluate-matrix --manifest MANIFEST --plan FROZEN_PLAN --configs GROUP_CONFIG_MAP --dataset-freeze FREEZE_RECEIPT --priority P0 --splits id_test,combo_test,unseen_test --out MATRIX_DIR --max-runs EXPLICIT_EPISODE_CAP --max-generated-tokens EXPLICIT_TOKEN_CAP` executes the complete predeclared P0/P1 plan through the existing serial batch runner. GROUP_CONFIG_MAP is a JSON object mapping every priority group to a service config path, resolved relative to the map. All groups must share the evaluation budget; simple/full model pairs must share model/tokenizer identities. It requires a verified dataset freeze and exact complete priority/split plan before any calls. This orchestration does not itself admit trained checkpoints or establish paid-resource authorization.

Global episode/token allocations are persisted and fsynced before each batch. An exclusive process lock prevents concurrent matrix controllers. Allocations conservatively reserve full per-episode generation caps; unused slots are not recycled. On interruption the allocation remains charged, original batch artifacts remain, failure type and current missing runs are retained, and `--resume` requires identical inputs/budgets and resumes existing batch recovery behavior. Top-level results are reconstructed from original batch ledgers; missing runs are always written, and an incomplete CLI matrix exits2. Invalid/unplanned/duplicate/over-reserved rows fail closed. Model failures do not become invented successful or infrastructure rows.

All391 CPU tests pass (21.198s). Mocked orchestration fixtures cover global slot cap, independent token cap, interruption/resume without double allocation, complete15-row fixture matrices, unequal budgets and incomplete plans. Actual matrix_unfrozen_cli_probe_v1 rejects an unfrozen receipt before creating a run directory or accessing missing plan/config files. Neither fixture success nor this rejection proves an actual model matrix. v25 remains unfrozen; real teacher/calibration, accepted training/export and scientific P0/P1 results remain incomplete. No model/teacher/optimizer/paid calls occur.


2026-11-01 current-protocol exported proposal controls: the labelled grader generator now exports67 cases (25 legal,42 negative), adding six real-tool proposal/reply/approval controls: legal version1; legal recollection after a version update; approval before replies; unbound replies; reuse of previous-version replies; and an early-approval invitation subsequently repaired with proper version2 replies/approval/reschedule. Public and independent grader semantics remain unchanged. Every new case retains an initial world, full action log, final world and explicit authored label provenance.

Actual grader_cases_proposal_v2/grader_report_proposal_v2 execute and grade all67 under tools-v3/grader-v13 with zero label mismatches or legal false positives and42/42 known negatives detected. Independent SQL checks causal reply/version bindings, approval-request timestamps, final send versions and operation consistency for the six new cases; both early-approval cases have exactly unapproved_external_send as their hard violation, including the repaired case with a valid final version2. It also verifies379 current-protocol tool records across all67. All391 CPU tests pass (21.466s). These are authored grader-coverage fixtures, not model evaluation, teacher samples or model-discovered exploits.

The initial v1 fixture attempt is retained with one legal-recollection label mismatch: coordination requests omitted outer meeting/version fields and therefore caused a real duplicate_external_effect. v2 fixes the fixture by carrying the correct public meeting/version in both envelope and structured request, preserving duplicate-effect grading. Earlier raw failures are not relabelled as passing. v25 remains unfrozen; formal data/calibration/training/export and scientific matrix results remain incomplete. No model/teacher/optimizer/paid calls occur.


2026-11-01 teacher admission independently regrades retained world evidence: episodes now retain task.json/private_spec.json/event_rules.json/initial_world.sqlite before policy execution, verify their bytes against original provenance hashes, and clone/grade using the retained initial world. Offline teacher verification binds retained task identity/split/input hashes and recomputes the complete grade and operation audit from world.sqlite plus trusted public/private task snapshots and actual runtime termination. A result.json success marker alone is insufficient. Missing/corrupt snapshots, altered final worlds, inconsistent grades/audits and non-training identities are rejected as teacher_world_evidence_invalid. Original policy-isolation/model/conversation/mask gates remain required separately.

The collector applies world verification before recording a candidate success or stopping retries; invalid candidates retain admission_error and charged resource reservations. Teacher extraction repeats verification, and SFT source admission now binds eight files (result, trajectory, isolation, task, private spec, event rules, initial world and final world) instead of three. Old sources without retained trusted snapshots cannot be silently reused. Unit fixtures use genuine toy tool worlds for world validation while remaining explicitly authored parser/isolation fixtures, never real teacher/human-review evidence; resource/selection-only tests explicitly mock the independent world gate.

Actual teacher_world_evidence_probe_v1 runs a software policy through the real full Harness/world service on current v25 T01, verifies successful independent world grading and four retained byte-identical inputs, then shows changed-world/unchanged-success-marker rejection. Authored policy extraction is separately rejected as not_a_model_trajectory. Regression cases also recompute source hashes after final-world or task-constraint changes and still fail independent verification. All392 CPU tests pass (27.472s). No model/teacher/optimizer/paid calls occur; genuine teacher data, human mask review, real overfit, calibrated dataset freeze, admitted training/export and scientific matrix evaluation remain incomplete.


2026-11-01 lossless teacher observation replay: extraction now requires the original prompt/tool schemas/task hash to equal the retained public task and current full Harness. A disposable clone of initial_world.sqlite receives the original episode tool-cap injection, then re-executes every action using its original write operation ID. It compares each complete tool observation (including provenance/time/schema metadata), enforces serial matched actions and unique write IDs, and compares the complete final database snapshot. Read/control operation IDs, prompt/schema drift, altered observations, missing interactions and final-world divergence fail as teacher_trace_replay_invalid. Original final-state comparison uses a read-only SQLite connection to avoid checkpointing retained WAL bytes. SFT source revalidation invokes this extraction gate too; collector success still applies its separate world-grade gate, with lossless trace admission occurring at extraction/build.

Actual teacher_trace_replay_probe_v1 replays five real-Harness T01 calls, rejects five tamper modes, and verifies original database bytes remain unchanged. Broader teacher_trace_replay_scoped_v2 reconstructs hash-matched trusted inputs in temporary directories for all128 previously saved authored Harness episodes, replays all2,016 actual calls and matches every final state with original bytes preserved. All80 training cases additionally pass teacher world-grade verification;48 non-training cases use ordinary grade revalidation and are not admitted as training sources. Scoped v1's non-training admission rejection is retained; the corrected probe preserves the split gate. These are offline software executions, not new model rollouts or teacher admission.

All393 CPU tests pass (33.553s), including direct extraction rejection of changed observations/prompt/schema/write keys. Review/source fixtures now carry actual toy-world tool conversations rather than unrelated parser conversations; dedicated parser-only tests explicitly mock the separate replay gate. Development probes also exposed and corrected the initial default-cap/runtime-cap mismatch and read-write final-state comparison's possible WAL checkpoint. No model/teacher/optimizer/paid calls occur. Genuine teacher data, human review/overfit, calibrated freeze, admitted training/export and scientific matrix results remain incomplete.


2026-11-01 GRPO command entrypoints and truthful launch exit status: `officegym grpo-smoke --config LOCKED_LAUNCH_CONFIG --out RUN_DIR --updates 5` now calls the existing bounded upstream launcher in smoke mode. Requested updates must be an exact integer1..5 and equal locked max_updates; this request never overrides resolved trainer settings. `officegym train-grpo --config LOCKED_LAUNCH_CONFIG --out RUN_DIR` routes to the same formal launcher, retaining its full compatibility/data/plugin/token-budget/package/source/accepted-SFT and prior semantic-smoke checks. configs/grpo_plan.json remains an intent document and is deliberately rejected, not converted into an unverified launch command. No checkpoint or admission gate is bypassed.

The primary CLI and underlying training.launch module (used by scripts/train_grpo.sh) now share process/semantic exit classification: failed exit, exhausted wall budget or unverified process-tree cleanup returns1; a successfully executed smoke still awaiting semantic verification returns2; only verified semantic smoke can return0. Formal train process success remains an execution result, not a claim of scientific improvement or accepted checkpoint export. The launcher currently records semantic_smoke_verification=pending, so a process-only smoke cannot receive a false completed-smoke CLI status. Existing independent seven-check smoke admission remains required before formal training.

Actual training_cli_preflight_v1 invokes both new commands with the project intent plan, observes rejection/exit1 before any output run directory, and records zero model/optimizer activity. Negative tests verify update mismatch fails before launch validation/output creation, and process-success/pending-semantic, timeout and cleanup failures propagate non-success exits. No native five-update smoke, formal GRPO or paid/GPU job is run; those acceptance gates remain incomplete.

Final validation for the GRPO entrypoint checkpoint: all395 CPU tests pass (33.697s); actual unresolved-intent CLI preflight rejects both commands without creating run outputs.


2026-11-01 committed fault checkpoint and independent branch foundation: the real worker-kill probe now captures world.sqlite/runtime.sqlite/harness.jsonl after verified SIGKILL and before recovery, then creates two independent clones from that same checkpoint. The content-bound checkpoint/checksum validates exactly one PREPARED/unacknowledged invitation, original request hashes in journal and committed world operation (including principal), exact committed receipt/outbox recipients/meeting/version, and an original full-Harness action without a matching interaction. Verification detects changed files and boundary inconsistencies; branching never overwrites existing outputs and revalidates its source after cloning. Snapshot backup preserves live WAL contents. Caller remains responsible for proven worker termination/quiescence and model provenance; this library does not establish a formal model fault episode.

Actual fault_checkpoint_probe_v1 kills its software-policy worker with exit-9, keeps the world server alive, saves the original committed boundary and successfully reconciles the original execution with one invitation effect. Independent audit confirms identical starting branch boundaries, reconciles branch_a through real operations.get/journal recovery, leaves branch_b and the source PREPARED, verifies one outbox effect per branch and byte-identical source checkpoint, and confirms all three owned processes have exited. Tests also reject missing committed effects and changed pending requests. All397 CPU tests pass (35.055s).

The checkpoint includes original model/Harness/task/budget metadata and trace, preserving the original wall-clock origin rather than granting fresh recovery time. It remains explicitly a software fault probe: no model calls, no simple/full paired model comparison, no second context-rebuild fault protocol and no formal fault-evaluate command are claimed. Those experiment/controller requirements, genuine teacher/calibration/training/export and scientific evaluation remain incomplete; no paid/GPU/model/optimizer activity occurs.


2026-11-01 single full-mode model fault execution path: `officegym fault-evaluate --instance INSTANCE --config LOCKED_SERVICE_CONFIG --out RUN_DIR --protocol after_world_commit` now runs one full-Harness episode under an outer owned-process subreaper. A separate controller owns the world service, kills the Harness worker only after its first invitation commits and before local ACK, verifies exit-9/world survival, retains the committed checkpoint, then creates a new worker using the same locked model/tokenizer/task/budget and original wall origin. Both initial/recovery ChatPolicy processes use the existing chroot/setuid/no-environment isolation, including compression. Journal restoration and operations.get reconciliation occur before policy continuation; injection is disabled on recovery. Short private socket paths avoid output-path limits. Trusted task snapshots, actual trace, checkpoint, both isolation/tokenizer attestations, immediate controller/PID boundary records and process-ownership cleanup artifacts are retained.

No-trigger episodes remain explicit (CLI exit2) and infrastructure failures exit1. Budget expiry is BUDGET_STOP rather than an infrastructure error. Trace usage after budget/infrastructure failure is a reported lower bound; the full generation cap remains reserved and monetary/GPU cost stays unmeasured. An incomplete outer controller writes an explicit retained incomplete/unknown-usage record. The new runner is currently full-mode/after_world_commit only: paired simple/full branches, context-rebuild faults, predeclared formal fault plans and fault-specific statistical export remain incomplete. Ordinary result export explicitly rejects fault episodes, and all fault-evaluation sources are excluded from P0 teacher extraction, preventing silent ordinary/teacher reclassification.

Actual fault_transport_probe_v4 uses three fixed-output HTTP responses with the real pinned tokenizer, real OS isolation and real SIGKILL/restart. It reaches DECLARED_DONE/full_success through four real tool calls (create, invitation, operations.get, finish); token usage counters are authored service fixtures, not LLM measurements. Independent SQL/PID audit verifies one invitation per recipient, original operation RECONCILED, unchanged checkpoint model identity and all five world/Harness/policy processes gone. Owned process-tree cleanup is verified. The initial v1 fixture incorrectly expected five calls; retained results show four, and corrected v2/v3/v4 assertions preserve the actual counter.

Actual fault_transport_boundaries_v1 retains an early-finish no-trigger episode and stops a stalled HTTP fixture under a6-second wall cap in6.07seconds, with BUDGET_STOP, usage_complete=false/full8192 reservation and verified process cleanup. All401 CPU tests pass (35.944s), including unsupported simple-mode preflight, no-trigger/infra CLI statuses and refusal to export fault records as ordinary or teacher data. These are authored HTTP/software transport tests, not actual model generations, paid calls, formal fault comparisons or optimizer work. Genuine teacher/calibration, human review/overfit, admitted training/export and complete scientific evaluation remain incomplete.


2026-11-01 matched conditional recovery: fault-evaluate now supports `--paired --max-generated-tokens CAP`, requiring a full-mode source and an explicit global cap covering three conservative generation reservations (source/simple/full) before any subprocess/model request. The controller captures one real commit boundary, clones both branches and runs isolated simple/full workers concurrently under the same original wall-clock deadline, model/tokenizer/task and per-episode budgets. Shared prefix usage appears in each branch's episode total but only once in physical reported usage; incomplete traces remain lower bounds and unused reservations are not recycled.

Shared-fault-prefix-recovery-v1 reconstructs completed logged interactions. Simple mode receives a clearly labelled user/controller uncertainty notice containing original pending request/content/operation ID; it performs no automatic query/ACK/retry and uses ordinary SimpleContext. Full mode retains pending actions and reconciles through the cloned journal. Both retain generated/input/tool counters and wall-clock origin. New branch traces explicitly bind the source checkpoint/trace, inherited usage and pending actions; ordinary restore rejects these traces to prevent accidental prefix-budget loss. Branch-start file hashes, task/model/budget/schema identity and cloned-journal path are validated. No tool confirmation is fabricated.

Actual paired_fault_transport_v1 uses fixed HTTP responses with the real tokenizer and OS isolation. Both branches complete from the same checkpoint/origin without duplicate invitations. The simple actor chooses operations.get; full uses controller reconciliation. Five authored HTTP responses produce fixture per-branch counters40/30 and physical50 after removing the duplicated20-token prefix; these are authored usage counters, not model cost measurements. Independent SQL/trace/PID audit verifies statuses PREPARED(simple)/RECONCILED(full), unchanged source checkpoint, common model/clock and all nine world/Harness/policy processes gone. All406 CPU tests pass (43.405s), including no new time from expired checkpoints and explicit paired-cap rejection before dispatch.

This is a conditional comparison from a common full-mode prefix, not end-to-end simple/full trigger comparison. Context-rebuild faults, predeclared formal fault plans/statistical export and actual model comparisons remain incomplete. Fault records remain barred from ordinary statistics and P0 teacher data. No real model generation, paid call or optimizer update occurs in these HTTP fixtures. Following the user's progress check, remaining work is prioritized around genuine data/training/export/scientific acceptance rather than further engineering expansion.


2026-11-01 critical-path correction from offline export evidence: export_critical_path_diagnosis_v1 hash-verifies the retained numeric plan, merge/result records, full-vocabulary logits and the diagnostic training/source reports without loading model weights or running a GPU. The candidate traces to real_optimizer_backend_probe_v1/trainer/checkpoint-2: formal_sft=false, two optimizer updates over authored protocol fixtures and zero teacher samples. It is not a genuine/admitted SFT model.

Replaying the unchanged atol=1e-4/rtol=1e-5 gate reproduces exactly two violating coordinates out of455,808, with all three fixed generations equal. The previously recorded FP64-accumulation weight audit reports zero elements different from the single-rounded FP64 reference across4,022,468,096 elements; this analysis reuses that hash-bound report and does not reread the model tensors. Descriptive probability differences/KL are retained in the report, not substituted for the fixed acceptance gate. Forward arithmetic/association remains a possible explanation, not an isolated cause. The diagnostic export still fails and is not admitted.

The earlier description of this diagnostic failure as a prerequisite blocker for beginning real teacher data/SFT was too broad. Real data collection should proceed once teacher/provider/resource inputs are available, and export admission must then be assessed for the actual trained checkpoint. Do not spend further GPU iterations trying to admit the old diagnostic artifact merely to unlock teacher collection. check-operator-inputs still reports missing teacher model/API USD cap/GPU-hours cap; the pending user request for those values remains unanswered. This offline diagnosis uses zero model generations, optimizer updates, GPU hours and API spend. No production training code or numerical thresholds changed; no repeated full test run was necessary.
