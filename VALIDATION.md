# 交接包验证记录

验证日期：2026-10-31。验证地点为文档制作环境，不是你的实验服务器。

- Python：3.12。
- scripts/validate_contracts.py：通过，检查工具数量、模板数量、计划数据量、seed 区间、奖励排序、训练配置与 SQLite DDL。
- python3 -m unittest discover -s tests -v：16 项通过。
- scripts/probe_host.py：成功生成制作环境的主机信息；该结果未放入交接包，避免被当作你的服务器配置。
- py_compile：辅助脚本及测试文件通过。

测试包含配对 Bootstrap 的已知输出、相同组零差异、同源场景聚类、缺失及重复记录、版本冲突、故障指标零分母、示例数据拒绝、事件日志不可改写和事务回滚。测试中的人工数据只用于软件断言，不是模型实验结果。

尚未验证：OfficeGym 业务工具、MCP server、真实任务生成器、参考解、完整 grader、Harness、GPU 推理、SFT、verl GRPO、模型表现和成本。它们是按手册实施的后续工作。

配置中的 API 费用和 GPU 时长预算尚未填写；这不代表允许无限运行。依赖组合需在服务器完成 smoke test 后锁定。

2026-11-01 native actor/base-reference diagnostic: artifacts/native_fsdp_logprob_probe_v2/report.json and independent_audit.json pass. Real native FSDP2 engine, two archived rows/one unique greedy history, initial/restored actor-reference equality, adapter perturbation isolation and independently indexed 453-position causal check all pass. Observed wall 31.83s, GPU peak 11,886MiB, no optimizer/generation/teacher/API work. This validates initial-base logprob plumbing; full KL, admitted-SFT reference, physical rollout synchronization and formal GRPO remain unverified. The historical v1 label sampled_kl_mean must be read as recorded-action logratio. All 269 stdlib tests pass.
