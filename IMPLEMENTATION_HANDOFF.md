# 给服务器编码 Agent 的实施要求

目标：按 OFFICEGYM_IMPLEMENTATION.md 实现可复现 OfficeGym，保留原始轨迹，用真实数据填写项目结果。不要一次编写庞大系统后再测试，不要生成“展示用实验数字”。

先做：
1. 运行 probe_host.py 与 validate_contracts.py，检查模型缓存、GPU、驱动和实际训练依赖。
2. 提交阶段计划、实际可用资源及版本清单。模型下载、API 花费和 GPU 训练遵守执行者配置的预算；配置里 null 不是无限预算。
3. 建立新仓库或当前工作分支；不改服务器全局环境。

阶段 A（CPU）：实现 world 数据层、dispatcher、24 个工具、时钟、事件、稳定回放；用 2–3 个手工任务跑脚本参考解。验收包括 API 层重放、版本冲突、权限、deadline、外发事件与副作用。SQLite + 工具 + 参考解稳定后再加模板。

阶段 B（数据与评分）：实现 14 个模板和公开信息参考解；数据隔离、指纹查重、留出组合排除；三层 grader 与正负反例。参考解必须从 Agent 客户端执行，不可读取数据库或私有答案。grader 不能直接信任看板或“已完成”文本。

阶段 C（工程）：完成 simple/full Harness，统一预算和模型适配器；先强 API 两组 baseline，随后 base_full。实现两类受控故障，保存触发点和世界 hash；模拟恢复必须真实杀死/重建 Agent 进程，不用模型口头声称“已恢复”。世界服务不随 Agent 重启。

阶段 D（SFT）：只采训练分区、不压缩的成功轨迹；检查 assistant mask，做过拟合小测试；导出同一 checkpoint，跑 dev，再探测组内奖励信号。

阶段 E（GRPO）：按锁定版 verl AgentLoop/BaseTool 接入；保留生成 token IDs，工具 token 不计 actor loss；每 rollout 独立世界；同一更新内补采样使用同 actor 版本。先 smoke，再有限预算训练。reference 必须为 SFT checkpoint。

阶段 F（冻结评测）：固定环境、测试 manifest、grader 和预算，对 P0 五组每实例 3 次；只在开发集选 checkpoint。统计完整运行台账，缺失结果不能静默丢弃。保留失败、预算停止和基础设施错误。

阶段 G（P1）：base_simple、grpo_simple、RS-SFT，预算匹配，报告训练收益迁移与流程对照的限度。没有完成就不写对应结论。

每个阶段交付：代码 commit、必要测试结果、命令与退出码、产物路径、已知问题、下一阶段准入结论。所有版本都可追溯；记录真实花费，不提交密钥。

最低事实边界：本包中的 officegym CLI 尚未实现；现有 scripts 是辅助工具。不得在 README 或简历写不存在的训练成功、通过率、提升、GPU 性能或 reward hacking 发现。

