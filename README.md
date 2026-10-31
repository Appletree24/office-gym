# OfficeGym

按《OFFICEGYM_IMPLEMENTATION.md》持续实施的可重置办公任务研究环境。目前具备 CPU 业务环境、独立评分器初版、参考任务探针、预算化 Harness 基础及真实 worker-kill 恢复探针。完整模板、MCP、模型训练及冻结评测尚未完成；没有模型效果或训练提升结论。

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
