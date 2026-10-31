# OfficeGym

按《OFFICEGYM_IMPLEMENTATION.md》持续实施的可重置办公任务研究环境。目前具备 24 个工具、14 个模板生成器、独立评分器、官方 MCP 适配器、预算化 Harness、真实 worker-kill 恢复探针，以及尚待真实后端验证的训练入口。正式数据冻结、模型训练及完整评测尚未完成；没有模型效果或训练提升结论。

Python 3.11+，当前 CPU 实现仅依赖标准库。项目目录中执行：

```bash
python3 -m venv .venv
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m officegym.cli probe-reference --out artifacts/reference_new
PYTHONPATH=src .venv/bin/python -m officegym.cli probe-fault --out artifacts/fault_new
.venv/bin/python scripts/export_versions.py --out artifacts/versions.json
```

探针使用全新输出目录，保留原始轨迹及数据库。`probe-reference` 检查基础排期、延迟审批和确定性重放；`probe-fault` 真实终止 Agent worker，保留世界服务并核对外发效果。它们均属于软件验证，不属于真实模型实验。

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
