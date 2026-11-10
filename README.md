# Personal Agent Runtime

可持续交互的个人助理运行时，并保留 OfficeGym 模拟评测。主入口是 `personal-agent`，原 `officegym` 的环境生成、独立评分、Harness 和训练入口继续可用。

已实现持久会话、独立任务、跨会话记忆、澄清与等待、定时任务、事件收件箱、应用操作确认和结果未知后的对账。SQLite 保存这些状态；关闭 CLI 不丢失等待条件，完成一个任务不关闭会话。首版面向本地 CLI，通过配置的 MCP 服务连接真实应用。

项目契约见 [SPEC.md](SPEC.md)，原评测与训练的完整说明见 [OFFICEGYM.md](OFFICEGYM.md)。离线演示使用脚本策略和模拟应用，没有模型效果或训练提升结论。

## 安装与离线演示

Python 3.11+。核心运行时依赖 `jsonschema`；连接 MCP 需要官方 Python SDK 1.30.0。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[mcp]'
.venv/bin/personal-agent demo --out artifacts/personal-demo
```

若虚拟环境没有 pip，可使用 `uv pip install --python .venv/bin/python -e '.[mcp]'`。演示目录必须是新的输出目录。

演示会询问称呼、保存带来源的偏好、重开数据库后恢复事件等待、在另一会话读取记忆，并验证确认前模拟文档数为 0、确认后为 1。结果写入输出目录的 `demo.json`。`kind=software_fixture`、`model_calls=0` 明确表示软件验证。

## 对话与恢复

复制 [配置示例](configs/personal-agent.example.json)，设置实际模型标识和凭证环境变量。文件路径相对于配置文件解析；`model_version` 是记录用标识，服务端未提供版本证明时不能据此声称模型版本已锁定。示例使用 DeepSeek 的 chat-completions 接口，也可填写兼容服务。运行时不会自动重试模型请求。

```bash
cp configs/personal-agent.example.json configs/personal-agent.local.json
export DEEPSEEK_API_KEY='你的凭证'
.venv/bin/personal-agent --config configs/personal-agent.local.json chat '记住我喜欢简洁的中文回复'
.venv/bin/personal-agent --config configs/personal-agent.local.json tasks
```

`chat` 返回会话和任务标识。再次对话使用 `--session SESSION_ID`；回答澄清问题使用 `--task TASK_ID`，继续同一任务。普通文本回答也能完成任务。

```bash
.venv/bin/personal-agent --config configs/personal-agent.local.json chat '请叫我小林' --task TASK_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json chat '下次也用这个称呼' --session SESSION_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json history --session SESSION_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json memories --query '称呼'
.venv/bin/personal-agent --config configs/personal-agent.local.json resume TASK_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json cancel TASK_ID
```

预算耗尽的任务进入 `paused`，需要显式 `resume`。澄清进入 `waiting_user`；等待真实时间或事件进入 `waiting_event`。工作进程只认领就绪、到期或租约过期的任务，等待状态不会忙轮询模型。

```bash
.venv/bin/personal-agent --config configs/personal-agent.local.json event --id reply-001 --kind reply --payload '{"subject":"回复已到达"}'
.venv/bin/personal-agent --config configs/personal-agent.local.json tick --limit 10
.venv/bin/personal-agent --config configs/personal-agent.local.json work --interval 5 --limit 10
```

`tick` 有界地运行一次；`work` 持续运行直到用户停止。事件来自本地管理入口，不会自动订阅外部应用。事件按用户与标识去重；同标识不同内容报冲突。只有登记等待之后收到的新事件能唤醒任务。

## 应用工具与确认

默认配置不连接应用。可以先创建独立的模拟办公数据库：

```bash
.venv/bin/personal-agent init-office --path artifacts/personal-office.sqlite
```

在配置的 `adapters` 中添加：

```json
[
  {"name": "office", "kind": "officegym", "path": "../artifacts/personal-office.sqlite"}
]
```

OfficeGym 适配器复用业务校验、权限、版本检查和幂等回执；不暴露评分、隐藏任务答案或 episode 控制工具，也不推进模拟时间或自动应用预设事件。它是模拟应用，不发送真实邮件。请使用专用世界文件，避免与运行中的评测共用。

真实应用通过已配置的 MCP 服务接入。stdio 示例：

```json
[
  {
    "name": "apps",
    "kind": "mcp",
    "transport": "stdio",
    "command": "/absolute/path/to/application-mcp",
    "args": [],
    "env_refs": {"APP_TOKEN": "PERSONAL_APP_TOKEN"},
    "readonly_tools": ["mail_search", "calendar_list"]
  }
]
```

Streamable HTTP 示例：

```json
[
  {
    "name": "apps",
    "kind": "mcp",
    "transport": "streamable-http",
    "url": "https://your-mcp-service.example/mcp",
    "header_env_refs": {"Authorization": "PERSONAL_MCP_AUTHORIZATION"},
    "readonly_tools": ["mail_search"]
  }
]
```

`PERSONAL_MCP_AUTHORIZATION` 应包含完整 Header 值，例如 `Bearer …`。只读分类由受信任配置指定，未列出的工具均视为写操作。工具名映射为 `适配器名__工具名`，远端名称中的点等字符映射为下划线，碰撞会报错。凭证不进入模型上下文。stdio 运行配置的本地程序；个人助理运行时本身不提供 OS 沙箱。

应用写入会先生成操作标识并进入 `waiting_confirmation`。检查确切参数后，通过 CLI 决定：

```bash
.venv/bin/personal-agent --config configs/personal-agent.local.json operation OPERATION_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json approve OPERATION_ID
.venv/bin/personal-agent --config configs/personal-agent.local.json reject OPERATION_ID
```

确认绑定用户、任务、参数、工具定义、适配器身份和有效期；模型没有批准工具。执行前收到同一任务的新用户消息会使旧批准失效。`chat`、`resume`、`approve`、`reject` 可加 `--no-run`，只持久化管理操作。

写后断连保存 `unknown`，任务进入 `blocked`。`resume` 会先对账，再决定能否继续。不支持查询回执的服务保持阻塞，不能自动重试。支持可靠回执的 MCP 服务可配置 `receipt_tool`、`receipt_argument` 和 `operation_meta_key`；查询工具的 `structuredContent` 必须为 `{"committed": true, "result": 原始业务结果}` 或 `{"committed": false}`。只有服务确实保证同操作标识幂等，才配置 `idempotent: true`。标识通过 MCP `_meta` 发送，默认键为 `personal_agent/operation_id`。

## 模块与边界

`src/personal_agent/` 包含持久化、上下文、工具注册、运行时、模型适配器、应用适配器和 CLI。`src/officegym/` 保留模拟世界、评测 Harness 与训练模块；两条入口共用办公业务实现。

记忆使用 SQLite 文本检索，包含版本和真实会话来源；偏好要求用户消息来源。上下文保留完整工具调用/结果组，并报告省略数量。`context_bytes` 限制发送的消息与工具定义的 UTF-8 序列化字节，不作为 token 估计；生成预算使用服务端实际报告的 completion tokens。

首版没有原生供应商 OAuth、GUI、远程多人认证、向量记忆或任意代码执行工具。本地 OS 账户是管理入口；数据库按配置的用户标识隔离。OfficeGym 原 Linux worker 沙箱仍只用于评测路径。取消任务不会撤销已发生的外部效果；在途写操作可留下待人工检查的结果未知记录。

## 验证

```bash
.venv/bin/python -m unittest tests.test_personal_store tests.test_personal_runtime tests.test_personal_protocol tests.test_personal_cli -v
.venv/bin/python -m unittest tests.test_dispatcher tests.test_budget_restart tests.test_harness tests.test_policy tests.test_context tests.test_journal tests.test_mcp
.venv/bin/python -m build --wheel
```

新增测试使用临时 SQLite、本地 HTTP 模型服务和官方 MCP 的 stdio / Streamable HTTP 服务，覆盖隔离、版本、租约、重启、等待、确认、取消与写后断连。没有向真实用户或供应商发送测试消息。原评测命令和实验限制详见 [OFFICEGYM.md](OFFICEGYM.md)。
